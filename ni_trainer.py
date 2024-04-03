# +
import os
import string
import re
import pdb
from collections import defaultdict

import pandas as pd
from transformers.trainer_seq2seq import Seq2SeqTrainer
from transformers.trainer import *
from datasets import load_metric
from transformers.trainer_callback import TrainerCallback
# -

PY_TXT = \
'''import os
import pandas as pd
pd.set_option('display.max_rows', 1000)
log_history_df = pd.read_json("log_history.json")
log_history_df
'''


class DenserEvalCallback(TrainerCallback):
    def on_step_end(self, args: TrainingArguments, state: TrainerState, control: TrainerControl, **kwargs):
        log_eval_steps = [1, 50, 100, 200]

        # Log
        if args.logging_strategy == IntervalStrategy.STEPS and state.global_step in log_eval_steps:
            control.should_log = True

        # Evaluate
        if args.evaluation_strategy == IntervalStrategy.STEPS and state.global_step in log_eval_steps:
            control.should_evaluate = True

        return control


# +
class NITrainer(Seq2SeqTrainer):
    def __init__(
        self,
        val_dataset,
        *args,
        **kwargs,
    ):
        super().__init__(
            *args,
            **kwargs,
        )
        self.val_dataset = val_dataset

        file_name = f'{self.args.output_dir.split("/")[-1].replace("run", "eval")}.py'
        with open(os.path.join(self.args.output_dir, file_name), "w") as f:
            f.write(PY_TXT)
    
    def compute_loss(
        self, model, inputs, grad_enabled=True, return_outputs=False, return_inputs=False,
    ):
        """
        How the loss is computed by Trainer. By default, all models return the loss in the first element.
        Subclass and override for custom behavior.
        """
        with torch.set_grad_enabled(grad_enabled):
            model_inputs = model.prepare_inputs(inputs)
            outputs = model(**model_inputs)
            loss = outputs["loss"]
            
        if return_outputs:
            return (loss, None, outputs)
        elif return_inputs:
            return (loss, outputs, model_inputs)
        else:
            return loss
    
    # rewrite the evaluation loop, with customized call to compute_metrics
    def evaluation_loop(
        self,
        dataloader: DataLoader,
        description: str,
        prediction_loss_only: Optional[bool] = None,
        ignore_keys: Optional[List[str]] = None,
        metric_key_prefix: str = "eval",
    ) -> EvalLoopOutput:
        """
        Prediction/evaluation loop, shared by `Trainer.evaluate()` and `Trainer.predict()`.

        Works both with or without labels.
        """
        args = self.args

        prediction_loss_only = prediction_loss_only if prediction_loss_only is not None else args.prediction_loss_only

        # if eval is called w/o train init deepspeed here
        if args.deepspeed and not self.deepspeed:
            # XXX: eval doesn't have `resume_from_checkpoint` arg but we should be able to do eval
            # from the checkpoint eventually
            deepspeed_engine, _, _ = deepspeed_init(
                self, num_training_steps=0, resume_from_checkpoint=None, inference=True
            )
            self.model = deepspeed_engine.module
            self.model_wrapped = deepspeed_engine
            self.deepspeed = deepspeed_engine

        model = self._wrap_model(self.model, training=False)

        # if full fp16 or bf16 eval is wanted and this ``evaluation`` or ``predict`` isn't called
        # while ``train`` is running, cast it to the right dtype first and then put on device
        if not self.is_in_train:
            if args.fp16_full_eval:
                model = model.to(dtype=torch.float16, device=args.device)
            elif args.bf16_full_eval:
                model = model.to(dtype=torch.bfloat16, device=args.device)

        batch_size = dataloader.batch_size

        logger.info(f"***** Running {description} *****")
        if has_length(dataloader.dataset):
            logger.info(f"  Num examples = {self.num_examples(dataloader)}")
        else:
            logger.info("  Num examples: Unknown")
        logger.info(f"  Batch size = {batch_size}")

        model.eval()

        self.callback_handler.eval_dataloader = dataloader
        # Do this before wrapping.
        eval_dataset = dataloader.dataset

        if is_torch_tpu_available():
            dataloader = pl.ParallelLoader(dataloader, [args.device]).per_device_loader(args.device)

        if args.past_index >= 0:
            self._past = None

        # Initialize containers
        # losses/preds/labels on GPU/TPU (accumulated for eval_accumulation_steps)
        losses_host = None
        losses_prefix_host = None
        losses_frozen_host = None
        losses_infer_host = None
        preds_host = None
        labels_host = None
        # losses/preds/labels on CPU (final containers)
        all_losses = None
        all_losses_prefix = None
        all_losses_frozen = None
        all_losses_infer = None
        all_preds = None
        all_labels = None
        # Will be useful when we have an iterable dataset so don't know its length.

        observed_num_examples = 0
        # Main evaluation loop
        for step, inputs in enumerate(dataloader):
            # Update the observed num examples
            observed_batch_size = find_batch_size(inputs)
            if observed_batch_size is not None:
                observed_num_examples += observed_batch_size
                # For batch samplers, batch_size is not known by the dataloader in advance.
                if batch_size is None:
                    batch_size = observed_batch_size

            # Prediction step
            loss, logits, labels = self.prediction_step(model, inputs, prediction_loss_only, ignore_keys=ignore_keys)
            loss_infer, loss_prefix, loss_frozen = None, None, None

            if is_torch_tpu_available():
                xm.mark_step()

            # Update containers on host
            if loss is not None:
                losses = self._nested_gather(loss.repeat(batch_size))
                losses_host = losses if losses_host is None else torch.cat((losses_host, losses), dim=0)
            if loss_prefix is not None:
                losses_prefix = self._nested_gather(loss_prefix.repeat(batch_size))
                losses_prefix_host = losses_prefix if losses_prefix_host is None else torch.cat((losses_prefix_host, losses_prefix), dim=0)
            if loss_frozen is not None:
                losses_frozen = self._nested_gather(loss_frozen.repeat(batch_size))
                losses_frozen_host = losses_frozen if losses_frozen_host is None else torch.cat((losses_frozen_host, losses_frozen), dim=0)
            if loss_infer is not None:
                losses_infer = self._nested_gather(loss_infer.repeat(batch_size))
                losses_infer_host = losses_infer if losses_infer_host is None else torch.cat((losses_infer_host, losses_infer), dim=0)
            if labels is not None:
                labels = self._pad_across_processes(labels)
                labels = self._nested_gather(labels)
                labels_host = labels if labels_host is None else nested_concat(labels_host, labels, padding_index=-100)
            if logits is not None:
                logits = self._pad_across_processes(logits)
                logits = self._nested_gather(logits)
                if self.preprocess_logits_for_metrics is not None:
                    logits = self.preprocess_logits_for_metrics(logits, labels)
                preds_host = logits if preds_host is None else nested_concat(preds_host, logits, padding_index=-100)
            self.control = self.callback_handler.on_prediction_step(args, self.state, self.control)

            # Gather all tensors and put them back on the CPU if we have done enough accumulation steps.
            if args.eval_accumulation_steps is not None and (step + 1) % args.eval_accumulation_steps == 0:
                if losses_host is not None:
                    losses = nested_numpify(losses_host)
                    all_losses = losses if all_losses is None else np.concatenate((all_losses, losses), axis=0)
                if losses_prefix_host is not None:
                    losses_prefix = nested_numpify(losses_prefix_host)
                    all_losses_prefix = losses_prefix if all_losses_prefix is None else np.concatenate((all_losses_prefix, losses_prefix), axis=0)
                if losses_frozen_host is not None:
                    losses_frozen = nested_numpify(losses_frozen_host)
                    all_losses_frozen = losses_frozen if all_losses_frozen is None else np.concatenate((all_losses_frozen, losses_frozen), axis=0)
                if losses_infer_host is not None:
                    losses_infer = nested_numpify(losses_infer_host)
                    all_losses_infer = losses_infer if all_losses_infer is None else np.concatenate((all_losses_infer, losses_infer), axis=0)
                if preds_host is not None:
                    logits = nested_numpify(preds_host)
                    all_preds = logits if all_preds is None else nested_concat(all_preds, logits, padding_index=-100)
                if labels_host is not None:
                    labels = nested_numpify(labels_host)
                    all_labels = (
                        labels if all_labels is None else nested_concat(all_labels, labels, padding_index=-100)
                    )

                # Set back to None to begin a new accumulation
                losses_host, preds_host, labels_host = None, None, None

        if args.past_index and hasattr(self, "_past"):
            # Clean the state at the end of the evaluation loop
            delattr(self, "_past")

        # Gather all remaining tensors and put them back on the CPU
        if losses_host is not None:
            losses = nested_numpify(losses_host)
            all_losses = losses if all_losses is None else np.concatenate((all_losses, losses), axis=0)
        if losses_prefix_host is not None:
            losses_prefix = nested_numpify(losses_prefix_host)
            all_losses_prefix = losses_prefix if all_losses_prefix is None else np.concatenate((all_losses_prefix, losses_prefix), axis=0)
        if losses_frozen_host is not None:
            losses_frozen = nested_numpify(losses_frozen_host)
            all_losses_frozen = losses_frozen if all_losses_frozen is None else np.concatenate((all_losses_frozen, losses_frozen), axis=0)
        if losses_infer_host is not None:
            losses_infer = nested_numpify(losses_infer_host)
            all_losses_infer = losses_infer if all_losses_infer is None else np.concatenate((all_losses_infer, losses_infer), axis=0)
        if preds_host is not None:
            logits = nested_numpify(preds_host)
            all_preds = logits if all_preds is None else nested_concat(all_preds, logits, padding_index=-100)
        if labels_host is not None:
            labels = nested_numpify(labels_host)
            all_labels = labels if all_labels is None else nested_concat(all_labels, labels, padding_index=-100)

        # Number of samples
        if has_length(eval_dataset):
            num_samples = len(eval_dataset)
        # The instance check is weird and does not actually check for the type, but whether the dataset has the right
        # methods. Therefore we need to make sure it also has the attribute.
        elif isinstance(eval_dataset, IterableDatasetShard) and hasattr(eval_dataset, "num_examples"):
            num_samples = eval_dataset.num_examples
        else:
            num_samples = observed_num_examples

        # Number of losses has been rounded to a multiple of batch_size and in a distributed training, the number of
        # samplers has been rounded to a multiple of batch_size, so we truncate.
        if all_losses is not None:
            all_losses = all_losses[:num_samples]
        if all_losses_prefix is not None:
            all_losses_prefix = all_losses_prefix[:num_samples]
        if all_losses_frozen is not None:
            all_losses_frozen = all_losses_frozen[:num_samples]
        if all_losses_infer is not None:
            all_losses_infer = all_losses_infer[:num_samples]
        if all_preds is not None:
            all_preds = nested_truncate(all_preds, num_samples)
        if all_labels is not None:
            all_labels = nested_truncate(all_labels, num_samples)

        # Metrics!
        if self.compute_metrics is not None and all_preds is not None and all_labels is not None:
            metrics = self.compute_metrics(dataset=eval_dataset, preds=all_preds, labels=all_labels, save_prefix=metric_key_prefix)
        else:
            metrics = {}

#         metrics["global_step"] = self.state.global_step

        # To be JSON-serializable, we need to remove numpy types or zero-d tensors
        metrics = denumpify_detensorize(metrics)

        if all_losses is not None:
            metrics[f"{metric_key_prefix}_loss"] = all_losses.mean().item()
        if all_losses_prefix is not None:
            metrics[f"{metric_key_prefix}_loss (prefix)"] = all_losses_prefix.mean().item()
        if all_losses_frozen is not None:
            metrics[f"{metric_key_prefix}_loss (frozen)"] = all_losses_frozen.mean().item()
        if all_losses_infer is not None:
            metrics[f"{metric_key_prefix}_loss_infer"] = all_losses_infer.mean().item()
            
        # Prefix all keys with metric_key_prefix + '_'
        for key in list(metrics.keys()):
            if not key.startswith(f"{metric_key_prefix}_"):
                metrics[f"{metric_key_prefix}_{key}"] = metrics.pop(key)
                
        return EvalLoopOutput(predictions=all_preds, label_ids=all_labels, metrics=metrics, num_samples=num_samples)
    
    def prediction_step(
        self,
        model: nn.Module,
        inputs: Dict[str, Union[torch.Tensor, Any]],
        prediction_loss_only: bool,
        ignore_keys: Optional[List[str]] = None,
    ) -> Tuple[Optional[float], Optional[torch.Tensor], Optional[torch.Tensor]]:

        inputs = self._prepare_inputs(inputs)
        inputs = self.model.prepare_inputs(inputs)
        has_labels = "labels" in inputs
        
        with torch.no_grad():
            if has_labels:
                with self.autocast_smart_context_manager():
                    outputs = model(**inputs)
                if self.label_smoother is not None:
                    loss = self.label_smoother(outputs, inputs["labels"]).mean().detach()
                else:
                    loss = (outputs["loss"] if isinstance(outputs, dict) else outputs[0]).mean().detach()
            else:
                loss = None
        
        if prediction_loss_only:
            return (loss, None, None)

        # XXX: adapt synced_gpus for fairscale as well
        gen_kwargs = {
            "max_length": self._max_length if self._max_length is not None else self.model.config.max_length,
            "num_beams": self._num_beams if self._num_beams is not None else self.model.config.num_beams,
            "synced_gpus": True if is_deepspeed_zero3_enabled() else False,
        }

        if "attention_mask" in inputs:
            gen_kwargs["attention_mask"] = inputs.get("attention_mask", None)

        # prepare generation inputs
        # some encoder-decoder models can have varying encder's and thus
        # varying model input names
        if hasattr(self.model, "encoder") and self.model.encoder.main_input_name != self.model.main_input_name:
            generation_inputs = inputs[self.model.encoder.main_input_name]
        else:
            generation_inputs = inputs[self.model.main_input_name]
            
        generated_tokens = self.model.generate(
            generation_inputs,
            **gen_kwargs,
        )
        # in case the batch is shorter than max length, the output should be padded
        if generated_tokens.shape[-1] < gen_kwargs["max_length"]:
            generated_tokens = self._pad_tensors_to_max_len(generated_tokens, gen_kwargs["max_length"])

        
        if has_labels:
            labels = inputs["labels"]
            if labels.shape[-1] < gen_kwargs["max_length"]:
                labels = self._pad_tensors_to_max_len(labels, gen_kwargs["max_length"])
        else:
            labels = None

        return (loss, generated_tokens, labels)
    
    def _maybe_log_save_evaluate(self, tr_loss, model, trial, epoch, ignore_keys_for_eval, tr_loss_meta=None):
        if self.control.should_log:
            if is_torch_tpu_available():
                xm.mark_step()

            logs: Dict[str, float] = {}

            # all_gather + mean() to get average loss over all processes
            # reset tr_loss to zero
            tr_loss_scalar = self._nested_gather(tr_loss).mean().item()
            tr_loss -= tr_loss
            logs["loss"] = round(tr_loss_scalar / (self.state.global_step - self._globalstep_last_logged), 4)
            self._total_loss_scalar += tr_loss_scalar
            self._globalstep_last_logged = self.state.global_step
            
            if tr_loss_meta is not None:
                tr_loss_meta_scalar = self._nested_gather(tr_loss_meta).mean().item()
                tr_loss_meta -= tr_loss_meta
                if self.state.global_step_meta > 0:
                    logs["loss_meta"] = round(tr_loss_meta_scalar / (self.state.global_step_meta - self._globalstep_meta_last_logged), 4)
                else:
                    logs["loss_meta"] = 0.
                self._total_loss_meta_scalar += tr_loss_meta_scalar
                self._globalstep_meta_last_logged = self.state.global_step_meta
            
            logs["learning_rate"] = self._get_learning_rate()
            self.store_flos()

            self.log(logs)
            self._save_log_history()

        metrics = None
        if self.control.should_evaluate:
            if self.args.do_val: self.validate(val_dataset=self.val_dataset, ignore_keys=ignore_keys_for_eval)
            
            metrics = self.evaluate(ignore_keys=ignore_keys_for_eval)
            self._report_to_hp_search(trial, epoch, metrics)

        if self.control.should_save:
            self._save_checkpoint(model, trial, metrics=metrics)
            self.control = self.callback_handler.on_save(self.args, self.state, self.control)
            
    def validate(
        self,
        val_dataset: Optional[Dataset] = None,
        ignore_keys: Optional[List[str]] = None,
        metric_key_prefix: str = "val",
        max_length: Optional[int] = None,
        num_beams: Optional[int] = None,
    ) -> Dict[str, float]:
        # memory metrics - must set up as early as possible
        self._memory_tracker.start()
        
        self._max_length = max_length if max_length is not None else self.args.generation_max_length
        self._num_beams = num_beams if num_beams is not None else self.args.generation_num_beams

        val_dataloader = self.get_val_dataloader(val_dataset)
        start_time = time.time()

        eval_loop = self.prediction_loop if self.args.use_legacy_prediction_loop else self.evaluation_loop
        output = eval_loop(
            val_dataloader,
            description="Validation",
            prediction_loss_only=True,
            ignore_keys=ignore_keys,
            metric_key_prefix=metric_key_prefix,
        )

        self.log(output.metrics)
        
    def get_val_dataloader(self, val_dataset: Optional[Dataset] = None) -> DataLoader:
        assert val_dataset is not None
        return self.get_eval_dataloader(val_dataset)
            
    def _save_log_history(self):
        log_history = defaultdict(dict)
        for row in self.state.log_history:
            log_history[row["step"]].update({"epoch": row["epoch"]})
            log_history[row["step"]].update(row)

        history_df = pd.DataFrame.from_dict(log_history, orient='index').drop(columns="step")
        history_path = os.path.join(self.args.output_dir, f"log_history.json")
        history_df.to_json(history_path)
