# +
import string
import re

import time
from typing import TYPE_CHECKING, Any, Callable, Dict, List, Optional, Tuple, Union
import math
from collections import defaultdict
import os
import copy

import torch
from torch import nn
from torch.utils.data import DataLoader, Dataset, RandomSampler, SequentialSampler
from torch.utils.data.distributed import DistributedSampler
from torch.distributions.uniform import Uniform
import datasets
from datasets import load_metric
import pandas as pd

from transformers.deepspeed import deepspeed_init, deepspeed_reinit, is_deepspeed_zero3_enabled
from transformers.trainer_callback import TrainerCallback
from transformers.trainer_utils import (
    PREFIX_CHECKPOINT_DIR,
    BestRun,
    EvalLoopOutput,
    EvalPrediction,
    HPSearchBackend,
    HubStrategy,
    IntervalStrategy,
    PredictionOutput,
    ShardedDDPOption,
    TrainerMemoryTracker,
    TrainOutput,
    default_compute_objective,
    default_hp_space,
    denumpify_detensorize,
    get_last_checkpoint,
    has_length,
    number_of_arguments,
    set_seed,
    speed_metrics,
)
from transformers.trainer_callback import (
    CallbackHandler,
    DefaultFlowCallback,
    PrinterCallback,
    ProgressCallback,
    TrainerCallback,
    TrainerControl,
    TrainerState,
)
from transformers.trainer_pt_utils import (
    DistributedLengthGroupedSampler,
    DistributedSamplerWithLoop,
    DistributedTensorGatherer,
    IterableDatasetShard,
    LabelSmoother,
    LengthGroupedSampler,
    SequentialDistributedSampler,
    ShardSampler,
    distributed_broadcast_scalars,
    distributed_concat,
    find_batch_size,
    get_parameter_names,
    nested_concat,
    nested_detach,
    nested_numpify,
    nested_truncate,
    nested_xla_mesh_reduce,
    reissue_pt_warnings,
)
from transformers.training_args import OptimizerNames, TrainingArguments
from transformers.utils import (
    CONFIG_NAME,
    WEIGHTS_NAME,
    find_labels,
    get_full_repo_name,
    is_apex_available,
    is_datasets_available,
    is_in_notebook,
    is_sagemaker_dp_enabled,
    is_sagemaker_mp_enabled,
    is_torch_tpu_available,
    logging,
)
from transformers.trainer import Trainer, _is_torch_generator_available
from transformers.optimization import get_scheduler
from transformers.debug_utils import DebugOption, DebugUnderflowOverflow

from ni_trainer import NITrainer, logger, DEFAULT_PROGRESS_CALLBACK

import pdb
# +
class IFTTrainer(NITrainer):
    def __init__(
        self,
        meta_dataset,
        *args,
        **kwargs,
    ):
        super().__init__(
            *args,
            **kwargs,
        )
        self.meta_dataset = meta_dataset
        self.temperature = self.args.temperature
    
    def compute_loss(self, model, model_prefix, inputs, grad_enabled=True, return_outputs=False):
        """
        How the loss is computed by Trainer. By default, all models return the loss in the first element.
        Subclass and override for custom behavior.
        """
        with torch.set_grad_enabled(grad_enabled):
            if self.args.prefix:
                model_frozen_inputs = self.compute_frozen_inputs_prefix(model, model_prefix, inputs)
            elif self.args.prefix_embeds:
                model_frozen_inputs = self.compute_frozen_inputs_prefix_embeds(model, model_prefix, inputs)
            elif self.args.prefix_linear:
                model_frozen_inputs = self.compute_frozen_inputs_prefix_linear(model, model_prefix, inputs)
            elif self.args.prefix_exemplar:
                model_frozen_inputs = self.compute_frozen_inputs_prefix_exemplar(model, model_prefix, inputs)
            elif self.args.exemplar:
                model_frozen_inputs = self.compute_frozen_inputs_exemplar(model, model_prefix, inputs)
            elif self.args.exemplar_embeds:
                model_frozen_inputs = self.compute_frozen_inputs_exemplar_embeds(model, model_prefix, inputs)
            elif self.args.exemplar_linear:
                model_frozen_inputs = self.compute_frozen_inputs_exemplar_linear(model, model_prefix, inputs)
            elif self.args.naturalmeta:
                model_frozen_inputs = model.prepare_inputs(inputs)
            elif self.args.nonprefix:
                model_frozen_inputs = {k.replace("_frozen", ""): v for k, v in inputs.items() if "_frozen" in k}
            
            outputs_frozen = model(**model_frozen_inputs)
            loss = outputs_frozen["loss"]
            
            if self.args.reweight:
                labels = model_frozen_inputs["labels"]
                lm_logits = outputs_frozen["logits"]
                ignore_index = -100
                loss_fct = nn.CrossEntropyLoss(ignore_index=ignore_index, reduction="none")
                losses_ = loss_fct(lm_logits.view(-1, lm_logits.size(-1)), labels.view(-1))
                losses = losses_.view(labels.size(0), -1)
                weights = torch.softmax(model_prefix[inputs["task_ids"]], -1) * labels.size(0)
                loss = torch.sum(losses * weights[:, None]) / torch.sum(labels != ignore_index)
            
        if return_outputs: return loss, outputs_frozen, model_frozen_inputs
        return loss
        
    def compute_meta_loss(self, model, inputs, grad_enabled=True):
        with torch.set_grad_enabled(grad_enabled):
            model_inputs = model.prepare_inputs(inputs)
            outputs_frozen = model(**model_inputs)
            loss = outputs_frozen["loss"]
        return loss
    
    def compute_frozen_inputs_prefix(self, model, model_prefix, inputs):
        model_prefix_inputs = model.prepare_inputs_prefix(inputs, ignore_keys=["labels", "decoder_input_ids"])
        
        prefix_embeds = model_prefix[inputs["task_ids"]] # n_batch x d_emb
        prefix_attention_mask = None
        model_frozen_inputs = model.prepare_inputs_frozen(inputs, prefix_embeds, prefix_attention_mask)
        return model_frozen_inputs
    
    def compute_frozen_inputs_prefix_embeds(self, model, model_prefix, inputs):
        model_prefix_inputs = model.prepare_inputs_prefix(inputs, ignore_keys=["labels", "decoder_input_ids"])
        
        prefix_outputs = model_prefix(**model_prefix_inputs)
        prefix_embeds = model_prefix.linear(prefix_outputs.last_hidden_state)
        if self.args.mask:
            prefix_attention_mask = model_prefix_inputs["attention_mask"]
        else:
            prefix_attention_mask = None
        model_frozen_inputs = model.prepare_inputs_frozen(inputs, prefix_embeds, prefix_attention_mask)
        return model_frozen_inputs
    
    def compute_frozen_inputs_prefix_linear(self, model, model_prefix, inputs):
        model_prefix_inputs = model.prepare_inputs_prefix(inputs, ignore_keys=["labels", "decoder_input_ids"])

        prefix_outputs = model_prefix.shared(model_prefix_inputs["input_ids"])
        prefix_hidden_state = torch.sum(prefix_outputs * model_prefix_inputs["attention_mask"][:, :, None], 1) / torch.sum(model_prefix_inputs["attention_mask"][:, :, None], 1) # n_batch x d_emb

        prefix_embeds_flat = model_prefix.linear(prefix_hidden_state)
        prefix_embeds = prefix_embeds_flat.view(prefix_embeds_flat.size(0), -1, model.config.d_model)
        prefix_attention_mask = None

        model_frozen_inputs = model.prepare_inputs_frozen(inputs, prefix_embeds, prefix_attention_mask)
        return model_frozen_inputs
    
    def compute_frozen_inputs_prefix_exemplar(self, model, model_prefix, inputs):
        model_prefix_inputs = model.prepare_inputs_prefix(inputs, ignore_keys=["labels", "decoder_input_ids"])

        prefix_outputs = model_prefix.shared(model_prefix_inputs["input_ids"])
        prefix_embeds = model_prefix.linear(prefix_outputs)
        if self.args.mask:
            prefix_attention_mask = model_prefix_inputs["attention_mask"]
        else:
            prefix_attention_mask = None
        model_frozen_inputs = model.prepare_inputs_frozen(inputs, prefix_embeds, prefix_attention_mask)
        return model_frozen_inputs
    
    def compute_frozen_inputs_exemplar(self, model, model_prefix, inputs):
        # compute probabilities over exemplars
        logits_exemplar = model_prefix[inputs["task_ids"]] # n_batch x n_exemplar x n_candidate
        probs_exemplar_soft = torch.softmax(logits_exemplar, dim=-1) # probability over n_candidate
        if self.args.hard:
            indices = torch.argmax(logits_exemplar, -1, keepdim=True)
            probs_exemplar_hard = torch.zeros_like(logits_exemplar).scatter_(-1, indices, 1.0) # binary probability over n_candidate
            probs_exemplar = probs_exemplar_hard - probs_exemplar_soft.detach() + probs_exemplar_soft
        else:
            probs_exemplar = probs_exemplar_soft

        # compute exemplar_input_embeds
        model_exemplar_inputs = model.prepare_inputs_exemplar(inputs)
        exemplar_inputs_embeds_flat = model.shared(model_exemplar_inputs["input_ids"]) # n_batch x l_seq x d_emb
        exemplar_inputs_embeds = exemplar_inputs_embeds_flat.view(probs_exemplar.size(0), -1, exemplar_inputs_embeds_flat.size(-2), exemplar_inputs_embeds_flat.size(-1)) # n_batch x n_candidate x l_seq x d_emb
        exemplar_embeds_unflat = torch.sum(probs_exemplar[:, :, :, None, None] * exemplar_inputs_embeds[:, None, :, :, :], 2) # n_batch x n_exemplar x l_seq x d_emb
        exemplar_embeds = exemplar_embeds_unflat.view(exemplar_embeds_unflat.size(0), -1, exemplar_embeds_unflat.size(-1)) # n_batch x (n_exemplar x l_seq) x d_emb

        if self.args.mask:
            # compute exemplar_attention_mask
            exemplar_inputs_attention_mask = model_exemplar_inputs["attention_mask"].view(probs_exemplar.size(0), -1, exemplar_inputs_embeds_flat.size(-2)) # n_batch x n_candidate x l_seq
            indices_exemplar = torch.argmax(probs_exemplar, -1, keepdim=True).expand(-1, -1, exemplar_inputs_attention_mask.size(-1)) # n_batch x n_exemplar x l_seq
            exemplar_attention_mask_unflat = torch.gather(exemplar_inputs_attention_mask, dim=1, index=indices_exemplar) # n_batch x n_exemplar x l_seq
            exemplar_attention_mask = exemplar_attention_mask_unflat.view(exemplar_attention_mask_unflat.size(0), -1) # n_batch x (n_exemplar x l_seq)
        else:
            exemplar_attention_mask = None

        model_frozen_inputs = model.prepare_inputs_frozen(inputs, exemplar_embeds, exemplar_attention_mask)
        return model_frozen_inputs
    
    def compute_frozen_inputs_exemplar_embeds(self, model, model_prefix, inputs):
        # compute binary probabilities over exemplars
        model_prefix_inputs = model.prepare_inputs_prefix(inputs, ignore_keys=["labels", "decoder_input_ids"])
        
        prefix_outputs = model_prefix.encoder(**model_prefix_inputs)
        prefix_embeds = torch.sum(prefix_outputs.last_hidden_state * model_prefix_inputs["attention_mask"][:, :, None], 1) / torch.sum(model_prefix_inputs["attention_mask"][:, :, None], 1) # n_batch x d_emb

        model_exemplar_inputs = model.prepare_inputs_exemplar(inputs)
        prefix_exemplar_outputs = model_prefix.encoder(**model_exemplar_inputs)
        prefix_exemplar_embeds_flat = torch.sum(prefix_exemplar_outputs.last_hidden_state * model_exemplar_inputs["attention_mask"][:, :, None], 1) / torch.sum(model_exemplar_inputs["attention_mask"][:, :, None], 1)
        prefix_exemplar_embeds = prefix_exemplar_embeds_flat.view(prefix_embeds.size(0), -1, prefix_exemplar_embeds_flat.size(-1)) # n_batch x n_candidate x d_emb_

        logits_exemplar_ = model_prefix.bilinear_layer(prefix_exemplar_embeds, prefix_embeds[:, None, :].expand(-1, prefix_exemplar_embeds.size(1), -1)) # n_batch x n_candidate x n_exemplar
        logits_exemplar = logits_exemplar_.transpose(-1, -2) # n_batch x n_exemplar x n_candidate
        probs_exemplar_soft = torch.softmax(logits_exemplar, dim=-1) # probability over n_candidate
        if self.args.hard:
            indices = torch.argmax(logits_exemplar, -1, keepdim=True)
            probs_exemplar_hard = torch.zeros_like(logits_exemplar).scatter_(-1, indices, 1.0) # binary probability over n_candidate
            probs_exemplar = probs_exemplar_hard - probs_exemplar_soft.detach() + probs_exemplar_soft
        else:
            probs_exemplar = probs_exemplar_soft

        # compute exemplar_embeds
        exemplar_inputs_embeds_flat = model.shared(model_exemplar_inputs["input_ids"])
        exemplar_inputs_embeds = exemplar_inputs_embeds_flat.view(prefix_embeds.size(0), -1, exemplar_inputs_embeds_flat.size(-2), exemplar_inputs_embeds_flat.size(-1)) # n_batch x n_candidate x l_seq x d_emb
        exemplar_embeds_unflat = torch.sum(probs_exemplar[:, :, :, None, None] * exemplar_inputs_embeds[:, None, :, :, :], 2) # n_batch x n_exemplar x l_seq x d_emb
        exemplar_embeds = exemplar_embeds_unflat.view(exemplar_embeds_unflat.size(0), -1, exemplar_embeds_unflat.size(-1)) # n_batch x (n_exemplar x l_seq) x d_emb

        if self.args.mask:
            # compute exemplar_attention_mask
            exemplar_inputs_attention_mask = model_exemplar_inputs["attention_mask"].view(prefix_embeds.size(0), -1, exemplar_inputs_embeds_flat.size(-2)) # n_batch x n_candidate x l_seq
            indices_exemplar = torch.argmax(probs_exemplar, -1, keepdim=True).expand(-1, -1, exemplar_inputs_attention_mask.size(-1)) # n_batch x n_exemplar x l_seq
            exemplar_attention_mask_unflat = torch.gather(exemplar_inputs_attention_mask, dim=1, index=indices_exemplar) # n_batch x n_exemplar x l_seq
            exemplar_attention_mask = exemplar_attention_mask_unflat.view(exemplar_attention_mask_unflat.size(0), -1) # n_batch x (n_exemplar x l_seq)
        else:
            exemplar_attention_mask = None

        model_frozen_inputs = model.prepare_inputs_frozen(inputs, exemplar_embeds, exemplar_attention_mask)
        return model_frozen_inputs
    
    def compute_frozen_inputs_exemplar_linear(self, model, model_prefix, inputs):
        # compute binary probabilities over exemplars
        model_prefix_inputs = model.prepare_inputs_prefix(inputs, ignore_keys=["labels", "decoder_input_ids"])

        prefix_outputs = model_prefix.shared(model_prefix_inputs["input_ids"])
        prefix_embeds = torch.sum(prefix_outputs * model_prefix_inputs["attention_mask"][:, :, None], 1) / torch.sum(model_prefix_inputs["attention_mask"][:, :, None], 1) # n_batch x d_emb
        if model_prefix.linear is not None: prefix_embeds = model_prefix.linear(prefix_embeds)

        model_exemplar_inputs = model.prepare_inputs_exemplar(inputs)
        prefix_exemplar_outputs = model_prefix.shared(model_exemplar_inputs["input_ids"])
        prefix_exemplar_embeds_flat = torch.sum(prefix_exemplar_outputs * model_exemplar_inputs["attention_mask"][:, :, None], 1) / torch.sum(model_exemplar_inputs["attention_mask"][:, :, None], 1)
        prefix_exemplar_embeds = prefix_exemplar_embeds_flat.view(prefix_embeds.size(0), -1, prefix_exemplar_embeds_flat.size(-1)) # n_batch x n_candidate x d_emb_
        if model_prefix.linear is not None: prefix_exemplar_embeds = model_prefix.linear(prefix_exemplar_embeds)

        logits_exemplar_ = model_prefix.bilinear(prefix_exemplar_embeds, prefix_embeds[:, None, :].expand(-1, prefix_exemplar_embeds.size(1), -1)) # n_batch x n_candidate x n_exemplar
        logits_exemplar = logits_exemplar_.transpose(-1, -2) # n_batch x n_exemplar x n_candidate
        probs_exemplar_soft = torch.softmax(logits_exemplar, dim=-1) # probability over n_candidate
        if self.args.hard:
            indices = torch.argmax(logits_exemplar, -1, keepdim=True)
            probs_exemplar_hard = torch.zeros_like(logits_exemplar).scatter_(-1, indices, 1.0) # binary probability over n_candidate
            probs_exemplar = probs_exemplar_hard - probs_exemplar_soft.detach() + probs_exemplar_soft
        else:
            probs_exemplar = probs_exemplar_soft

        # compute exemplar_embeds
        exemplar_inputs_embeds_flat = model.shared(model_exemplar_inputs["input_ids"])
        exemplar_inputs_embeds = exemplar_inputs_embeds_flat.view(prefix_embeds.size(0), -1, exemplar_inputs_embeds_flat.size(-2), exemplar_inputs_embeds_flat.size(-1)) # n_batch x n_candidate x l_seq x d_emb
        exemplar_embeds_unflat = torch.sum(probs_exemplar[:, :, :, None, None] * exemplar_inputs_embeds[:, None, :, :, :], 2) # n_batch x n_exemplar x l_seq x d_emb
        exemplar_embeds = exemplar_embeds_unflat.view(exemplar_embeds_unflat.size(0), -1, exemplar_embeds_unflat.size(-1)) # n_batch x (n_exemplar x l_seq) x d_emb

        if self.args.mask:
            # compute exemplar_attention_mask
            exemplar_inputs_attention_mask = model_exemplar_inputs["attention_mask"].view(prefix_embeds.size(0), -1, exemplar_inputs_embeds_flat.size(-2)) # n_batch x n_candidate x l_seq
            indices_exemplar = torch.argmax(probs_exemplar, -1, keepdim=True).expand(-1, -1, exemplar_inputs_attention_mask.size(-1)) # n_batch x n_exemplar x l_seq
            exemplar_attention_mask_unflat = torch.gather(exemplar_inputs_attention_mask, dim=1, index=indices_exemplar) # n_batch x n_exemplar x l_seq
            exemplar_attention_mask = exemplar_attention_mask_unflat.view(exemplar_attention_mask_unflat.size(0), -1) # n_batch x (n_exemplar x l_seq)
        else:
            exemplar_attention_mask = None

        model_frozen_inputs = model.prepare_inputs_frozen(inputs, exemplar_embeds, exemplar_attention_mask)
        return model_frozen_inputs
    
    def approx_inverse_HVP(self, v, f, learning_rate=None, k_ift=None, debug=False):
        if learning_rate is None: learning_rate = self.args.learning_rate
            
        p = tuple(v_item.clone().detach() for v_item in v)
        if debug: print(f"step0: p={p[0]}, v={v[0]}")

        for j in range(k_ift):
            f_gradv = torch.autograd.grad(f, self.model.get_frozen_parameters(), grad_outputs=v, retain_graph=True)
            assert len(p) == len(v) == len(f_gradv)
            for p_item, v_item, f_gradv_item in zip(p, v, f_gradv):
                v_item -= learning_rate*f_gradv_item
                p_item += v_item
            if debug: 
                print(f"step{j+1}: p={p[0]}, v={v[0]}")

        v2 = tuple(-learning_rate*p_item for p_item in p)
        return v2
    
    def train(
        self,
        resume_from_checkpoint: Optional[Union[str, bool]] = None,
        trial: Union["optuna.Trial", Dict[str, Any]] = None,
        ignore_keys_for_eval: Optional[List[str]] = None,
        **kwargs,
    ):
        # memory metrics - must set up as early as possible
        self._memory_tracker.start()

        args = self.args

        self.is_in_train = True

        self._move_model_to_device(self.model, args.device)

        # Data loader and number of training steps
        train_dataloader = self.get_train_dataloader(self.train_dataset, self.args.train_batch_size)

        # Setting up training control variables:
        # number of training epochs: num_train_epochs
        # number of training steps per epoch: num_update_steps_per_epoch
        # total number of training steps to execute: max_steps
        total_train_batch_size = args.per_device_train_batch_size * args.world_size * args.gradient_accumulation_steps

        len_dataloader = None
        if has_length(train_dataloader):
            len_dataloader = len(train_dataloader)
            num_update_steps_per_epoch = len_dataloader // args.gradient_accumulation_steps
            num_update_steps_per_epoch = max(num_update_steps_per_epoch, 1)
            num_examples = self.num_examples(train_dataloader)
            if args.max_steps > 0:
                max_steps = args.max_steps
                num_train_epochs = args.max_steps // num_update_steps_per_epoch + int(
                    args.max_steps % num_update_steps_per_epoch > 0
                )
                # May be slightly incorrect if the last batch in the training dataloader has a smaller size but it's
                # the best we can do.
                num_train_samples = args.max_steps * total_train_batch_size
            else:
                max_steps = math.ceil(args.num_train_epochs * num_update_steps_per_epoch)
                num_train_epochs = math.ceil(args.num_train_epochs)
                num_train_samples = self.num_examples(train_dataloader) * args.num_train_epochs
        elif args.max_steps > 0:  # Rely on max_steps when dataloader does not have a working size
            max_steps = args.max_steps
            # Setting a very large number of epochs so we go as many times as necessary over the iterator.
            num_train_epochs = sys.maxsize
            num_update_steps_per_epoch = max_steps
            num_examples = total_train_batch_size * args.max_steps
            num_train_samples = args.max_steps * total_train_batch_size
        else:
            raise ValueError(
                f"args.max_steps must be set to a positive value if dataloader does not have a length, was {args.max_steps}"
            )

        meta_max_steps = max_steps
        
        self.optimizer, self.lr_scheduler = self.create_optimizer_and_scheduler(self.model.get_frozen_named_parameters(), num_training_steps=max_steps)
        meta_optimizer, meta_lr_scheduler = self.create_optimizer_and_scheduler(self.model.get_prefix_named_parameters(), num_training_steps=max_steps, learning_rate=args.learning_rate_meta, warmup_steps=args.warmup_steps_meta)
        
        if args.noise_prefix > 0:
            self.uniform = Uniform(torch.tensor(-args.noise_prefix).to(self.model.device), torch.tensor(args.noise_prefix).to(self.model.device))

        self.state = TrainerState()

        logger.info("***** Running training *****")
        logger.info(f"  Num examples = {num_examples}")
        logger.info(f"  Num Epochs = {num_train_epochs}")
        logger.info(f"  Instantaneous batch size per device = {args.per_device_train_batch_size}")
        logger.info(f"  Total train batch size (w. parallel, distributed & accumulation) = {total_train_batch_size}")
        logger.info(f"  Gradient Accumulation steps = {args.gradient_accumulation_steps}")
        logger.info(f"  Total optimization steps = {max_steps}")

        self.state.epoch = 0
    
        start_time = time.time()
        epochs_trained = 0
        steps_trained_in_current_epoch = 0
        steps_trained_progress_bar = None

        # Update the references
        self.callback_handler.model = self.model
        self.callback_handler.optimizer = self.optimizer
        self.callback_handler.lr_scheduler = self.lr_scheduler
        self.callback_handler.train_dataloader = train_dataloader
        self.state.trial_name = self.hp_name(trial) if self.hp_name is not None else None
        self.state.trial_params = None
        
        # This should be the same if the state has been saved but in case the training arguments changed, it's safer
        # to set this after the load.
        self.state.max_steps = max_steps
        self.state.num_train_epochs = num_train_epochs
        self.state.is_local_process_zero = self.is_local_process_zero()
        self.state.is_world_process_zero = self.is_world_process_zero()

        # tr_loss is a tensor to avoid synchronization of TPUs through .item()
        # _total_loss_scalar is updated everytime .item() has to be called on tr_loss and stores the sum of all losses
        tr_loss = torch.tensor(0.0).to(args.device)
        self._total_loss_scalar = 0.0
        
        self._globalstep_last_logged = self.state.global_step
        
        tr_loss_meta = torch.tensor(0.0).to(args.device)
        self._total_loss_meta_scalar = 0.0
        
        self.state.global_step_meta = 0
        self._globalstep_meta_last_logged = self.state.global_step_meta
        
        self.optimizer.zero_grad()

        self.control = self.callback_handler.on_train_begin(args, self.state, self.control)

        for epoch in range(epochs_trained, num_train_epochs):
            epoch_iterator = train_dataloader
            meta_dataloader = iter(self.get_train_dataloader(self.meta_dataset, self.args.meta_batch_size)) # TODO

            # Reset the past mems state at the beginning of each epoch if necessary.
            if args.past_index >= 0:
                self._past = None

            steps_in_epoch = (
                len(epoch_iterator)
                if len_dataloader is not None
                else args.max_steps * args.gradient_accumulation_steps
            )
            self.control = self.callback_handler.on_epoch_begin(args, self.state, self.control)

            for step, inputs in enumerate(epoch_iterator):
                if steps_trained_in_current_epoch > 0:
                    steps_trained_in_current_epoch -= 1
                    if steps_trained_progress_bar is not None:
                        steps_trained_progress_bar.update(1)
                    if steps_trained_in_current_epoch == 0:
                        self._load_rng_state(resume_from_checkpoint)
                    continue
                elif steps_trained_progress_bar is not None:
                    steps_trained_progress_bar.close()
                    steps_trained_progress_bar = None

                if step % args.gradient_accumulation_steps == 0:
                    self.control = self.callback_handler.on_step_begin(args, self.state, self.control)
                    
                self.current_flos += float(self.floating_point_ops(inputs))
                
#                 if args.debug_trainer:
#                     model_param = self.model.shared.weight
# #                     model_prefix_param = self.model.model_prefix[inputs["task_ids"], :, 0]
#                     print(f"before: model = {model_param}, model_prefix = {self.model.model_prefix[inputs['task_ids'], :, 0]}")
                
                self.model.train()
                inputs = self._prepare_inputs(inputs)
                ###### meta training ######
                if args.meta_steps > 0 and (step+1) % args.meta_steps == 0 and not args.nonprefix:
                    for _ in range(args.meta_gradient_accumulation_steps):
                        try: # TODO
                            meta_inputs = next(meta_dataloader)
                        except StopIteration as e:
                            meta_dataloader = iter(self.get_train_dataloader(self.meta_dataset, self.args.meta_batch_size)) # TODO
                            meta_inputs = next(meta_dataloader)
                        meta_inputs = self._prepare_inputs(meta_inputs)

                        with self.autocast_smart_context_manager():
                            loss_meta = self.compute_meta_loss(model=self.model, inputs=meta_inputs)
                            loss_meta /= args.meta_gradient_accumulation_steps
                            
                        tr_loss_meta += loss_meta.detach()

                        v1 = torch.autograd.grad(loss_meta, self.model.get_frozen_parameters())
                        del loss_meta

                        with self.autocast_smart_context_manager():
                            loss = self.compute_loss(model=self.model, model_prefix=self.model.model_prefix, inputs=inputs)

                        f = torch.autograd.grad(loss, self.model.get_frozen_parameters(), create_graph=True)
                        del loss
                        
                        if args.debug_trainer: pdb.set_trace()

                        v2 = self.approx_inverse_HVP(v1, f, learning_rate=args.learning_rate_ift, k_ift=args.k_ift, debug=args.debug_hvp)
                        del v1

    #                     grads_prefix = torch.autograd.grad(f, self.model.get_prefix_parameters(), grad_outputs=v2, allow_unused=True)
    #                     name_None = [name for (name, param), grad in zip(self.model.get_prefix_named_parameters(), grads_prefix) if grad is None]
    #                     pdb.set_trace()
                        grads_prefix = torch.autograd.grad(f, self.model.get_prefix_parameters(), grad_outputs=v2)
                        if args.debug_hvp:
                            print(grads_prefix[0][inputs['task_ids'], :, 0])
                            pdb.set_trace()
                        del f, v2
                        
                        assert len(self.model.get_prefix_parameters()) == len(grads_prefix)
                        for param, grad in zip(self.model.get_prefix_parameters(), grads_prefix):
                            param.backward(grad)
                        del grads_prefix
                    
                    # Gradient clipping
                    if args.max_grad_norm_meta is not None and args.max_grad_norm_meta > 0 and not self.deepspeed:
                        self.clip_gradient(self.optimizer, self.model, args.max_grad_norm_meta)

                    meta_optimizer.step()
                    meta_lr_scheduler.step()

#                     if args.debug_trainer: 
#                         print(f"after meta step: model = {model_param}, model_prefix = {self.model.model_prefix[inputs['task_ids'], :, 0]}")
#                         pdb.set_trace()

                    self.model.zero_grad()
                    
                    self.state.global_step_meta += 1
                
                ###### main training ######
                with self.autocast_smart_context_manager():
                    if args.noise_prefix > 0: self.add_noise(model=self.model, noise_prefix=args.noise_prefix)
                    loss = self.compute_loss(model=self.model, model_prefix=self.model.model_prefix, inputs=inputs)
                    loss = loss / args.gradient_accumulation_steps
                    
                tr_loss += loss.detach()
                loss.backward()
                del loss

                if (step + 1) % args.gradient_accumulation_steps == 0 or (
                    # last step in epoch but step is always smaller than gradient_accumulation_steps
                    steps_in_epoch <= args.gradient_accumulation_steps
                    and (step + 1) == steps_in_epoch
                ):
                    # Gradient clipping
                    if args.max_grad_norm is not None and args.max_grad_norm > 0 and not self.deepspeed:
                        self.clip_gradient(self.optimizer, self.model, args.max_grad_norm)
                        
                    self.optimizer.step()
                    self.lr_scheduler.step()

#                     if args.debug_trainer: 
#                         print(f"after main step: model = {model_param}, model_prefix = {self.model.model_prefix[inputs['task_ids'], :, 0]}")
                        
                    self.model.zero_grad()
                    
                    self.state.global_step += 1
                    self.state.epoch = epoch + (step + 1) / steps_in_epoch
                    self.control = self.callback_handler.on_step_end(args, self.state, self.control)

                    self._maybe_log_save_evaluate(tr_loss, self.model, trial, epoch, ignore_keys_for_eval, tr_loss_meta=tr_loss_meta)
                else:
                    self.control = self.callback_handler.on_substep_end(args, self.state, self.control)
                                        
                if self.control.should_epoch_stop or self.control.should_training_stop:
                    break
                    
            self.control = self.callback_handler.on_epoch_end(args, self.state, self.control)
            if self.control.should_training_stop:
                break

        # add remaining tr_loss
        self._total_loss_scalar += tr_loss.item()
        train_loss = self._total_loss_scalar / self.state.global_step

        metrics = speed_metrics("train", start_time, num_samples=num_train_samples, num_steps=self.state.max_steps)
        self.store_flos()
        metrics["total_flos"] = self.state.total_flos
        metrics["train_loss"] = train_loss

        self.is_in_train = False

        self._memory_tracker.stop_and_update_metrics(metrics)

        self.log(metrics)

        self.control = self.callback_handler.on_train_end(args, self.state, self.control)

        return TrainOutput(self.state.global_step, train_loss, metrics)
    
    def add_noise(self, model, noise_prefix):
        with torch.no_grad():
            for param in model.get_prefix_parameters():
                param += self.uniform.sample(param.size())
    
    def clip_gradient(self, optimizer, model, max_grad_norm):
        # deepspeed does its own clipping
        if hasattr(optimizer, "clip_grad_norm"):
            # Some optimizers (like the sharded optimizer) have a specific way to do gradient clipping
            optimizer.clip_grad_norm(max_grad_norm)
        elif hasattr(model, "clip_grad_norm_"):
            # Some models (like FullyShardedDDP) have a specific way to do gradient clipping
            model.clip_grad_norm_(max_grad_norm)
        else:
            # Revert to normal clipping otherwise, handling Apex or full precision
            nn.utils.clip_grad_norm_(
                amp.master_params(optimizer) if self.use_apex else model.parameters(),
                max_grad_norm,
            )

    def get_train_dataloader(self, train_dataset, batch_size) -> DataLoader:
        """
        Returns the training [`~torch.utils.data.DataLoader`].
        Will use no sampler if `train_dataset` does not implement `__len__`, a random sampler (adapted to distributed
        training if necessary) otherwise.
        Subclass and override this method if you want to inject some custom behavior.
        """
        if train_dataset is None:
            raise ValueError("Trainer: training requires a train_dataset.")
        
        if batch_size is None:
            batch_size = self.args.train_batch_size

        if is_datasets_available() and isinstance(train_dataset, datasets.Dataset):
            train_dataset = self._remove_unused_columns(train_dataset, description="training")

        train_sampler = self._get_train_sampler(train_dataset)

        return DataLoader(
            train_dataset,
            batch_size=batch_size,
            sampler=train_sampler,
            collate_fn=self.data_collator,
            drop_last=self.args.dataloader_drop_last,
            num_workers=self.args.dataloader_num_workers,
            pin_memory=self.args.dataloader_pin_memory,
        )
    
    def _get_train_sampler(self, train_dataset) -> Optional[torch.utils.data.Sampler]:
        if train_dataset is None or not has_length(train_dataset):
            return None

        generator = None
        if self.args.world_size <= 1 and _is_torch_generator_available:
            generator = torch.Generator()
            # for backwards compatibility, we generate a seed here (which is sampled from a generator seeded with
            # `args.seed`) if data_seed isn't provided.
            # Further on in this method, we default to `args.seed` instead.
            if self.args.data_seed is None:
                seed = int(torch.empty((), dtype=torch.int64).random_().item())
            else:
                seed = self.args.data_seed
            generator.manual_seed(seed)

        seed = self.args.data_seed if self.args.data_seed is not None else self.args.seed

        # Build the sampler.
        if self.args.group_by_length:
            if is_datasets_available() and isinstance(train_dataset, datasets.Dataset):
                lengths = (
                    train_dataset[self.args.length_column_name]
                    if self.args.length_column_name in train_dataset.column_names
                    else None
                )
            else:
                lengths = None
            model_input_name = self.tokenizer.model_input_names[0] if self.tokenizer is not None else None
            if self.args.world_size <= 1:
                return LengthGroupedSampler(
                    self.args.train_batch_size * self.args.gradient_accumulation_steps,
                    dataset=train_dataset,
                    lengths=lengths,
                    model_input_name=model_input_name,
                    generator=generator,
                )
            else:
                return DistributedLengthGroupedSampler(
                    self.args.train_batch_size * self.args.gradient_accumulation_steps,
                    dataset=train_dataset,
                    num_replicas=self.args.world_size,
                    rank=self.args.process_index,
                    lengths=lengths,
                    model_input_name=model_input_name,
                    seed=seed,
                )
        else:
            if self.args.world_size <= 1:
                if _is_torch_generator_available:
                    return RandomSampler(train_dataset, generator=generator)
                return RandomSampler(train_dataset)
            elif (
                self.args.parallel_mode in [ParallelMode.TPU, ParallelMode.SAGEMAKER_MODEL_PARALLEL]
                and not self.args.dataloader_drop_last
            ):
                # Use a loop for TPUs when drop_last is False to have all batches have the same size.
                return DistributedSamplerWithLoop(
                    train_dataset,
                    batch_size=self.args.per_device_train_batch_size,
                    num_replicas=self.args.world_size,
                    rank=self.args.process_index,
                    seed=seed,
                )
            else:
                return DistributedSampler(
                    train_dataset,
                    num_replicas=self.args.world_size,
                    rank=self.args.process_index,
                    seed=seed,
                )
            
    def create_optimizer_and_scheduler(self, named_parameters, num_training_steps: int, learning_rate=None, warmup_steps=None):
        """
        Setup the optimizer and the learning rate scheduler.
        We provide a reasonable default that works well. If you want to use something else, you can pass a tuple in the
        Trainer's init through `optimizers`, or subclass and override this method (or `create_optimizer` and/or
        `create_scheduler`) in a subclass.
        """
        optimizer = self.create_optimizer(named_parameters, learning_rate=learning_rate)
        scheduler = self.create_scheduler(num_training_steps=num_training_steps, optimizer=optimizer, num_warmup_steps=warmup_steps)
        return optimizer, scheduler

    def create_optimizer(self, named_parameters, learning_rate):
        """
        Setup the optimizer.
        We provide a reasonable default that works well. If you want to use something else, you can pass a tuple in the
        Trainer's init through `optimizers`, or subclass and override this method in a subclass.
        """
        decay_parameters = get_parameter_names(self.model, [nn.LayerNorm])
        decay_parameters = [name for name in decay_parameters if "bias" not in name]
        optimizer_grouped_parameters = [
            {
                "params": [p for n, p in named_parameters if n in decay_parameters],
                "weight_decay": self.args.weight_decay,
            },
            {
                "params": [p for n, p in named_parameters if n not in decay_parameters],
                "weight_decay": 0.0,
            },
        ]

        optimizer_cls, optimizer_kwargs = Trainer.get_optimizer_cls_and_kwargs(self.args)
        if learning_rate is not None: optimizer_kwargs["lr"] = learning_rate

        if self.sharded_ddp == ShardedDDPOption.SIMPLE:
            optimizer = OSS(
                params=optimizer_grouped_parameters,
                optim=optimizer_cls,
                **optimizer_kwargs,
            )
        else:
            optimizer = optimizer_cls(optimizer_grouped_parameters, **optimizer_kwargs)

        if is_sagemaker_mp_enabled():
            optimizer = smp.DistributedOptimizer(optimizer)

        return optimizer

    def create_scheduler(self, num_training_steps: int, optimizer: torch.optim.Optimizer, num_warmup_steps=None):
        """
        Setup the scheduler. The optimizer of the trainer must have been set up either before this method is called or
        passed as an argument.
        Args:
            num_training_steps (int): The number of training steps to do.
        """
        if num_warmup_steps is None:
            num_warmup_steps = self.args.get_warmup_steps(num_training_steps)
        
        lr_scheduler = get_scheduler(
            self.args.lr_scheduler_type,
            optimizer=optimizer,
            num_warmup_steps=num_warmup_steps,
            num_training_steps=num_training_steps,
        )
        return lr_scheduler
# -


