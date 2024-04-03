# +
# #!/usr/bin/env python
# coding=utf-8
import logging
import os
import sys
import json
import random
import pdb
from collections import defaultdict
import string

import pandas as pd
import datasets
import nltk  # Here to have a nice missing dependency error message early on
import numpy as np
from datasets.utils import set_progress_bar_enabled
from datasets import load_dataset, load_metric, DownloadConfig
import torch
import torch.nn as nn

import transformers
from filelock import FileLock
from transformers import (
    AutoConfig,
    AutoModelForSeq2SeqLM,
    AutoTokenizer,
    DataCollatorForSeq2Seq,
    HfArgumentParser,
    MBart50Tokenizer,
    MBart50TokenizerFast,
    MBartTokenizer,
    MBartTokenizerFast,
    Seq2SeqTrainer,
    set_seed,
)
from transformers.file_utils import is_offline_mode, is_in_notebook
from transformers.trainer_utils import get_last_checkpoint
from transformers.models.t5.modeling_t5 import T5EncoderModel

from modeling_t5_frozen import T5FrozenForConditionalGeneration
from ni_collator import DataCollatorForNI
from ni_trainer import NITrainer, DenserEvalCallback
from ift_trainer import IFTTrainer
from compute_metrics import compute_metrics, compute_grouped_metrics
from arguments import DataTrainingArguments, ModelArguments, NITrainingArguments, args_to_output_dir

PROXIES = {
    'http': os.environ.get("PROXY"),
    'https': os.environ.get("PROXY"),
}

set_progress_bar_enabled(False)
logger = logging.getLogger(__name__)

try:
    nltk.data.find("tokenizers/punkt")
except (LookupError, OSError):
    if is_offline_mode():
        raise LookupError(
            "Offline mode: run this script without TRANSFORMERS_OFFLINE first to download nltk data files"
        )
    with FileLock(".lock") as lock:
        nltk.download("punkt", quiet=True)
        
# A list of all multilingual tokenizer which require lang attribute.
MULTILINGUAL_TOKENIZERS = [MBartTokenizer, MBartTokenizerFast, MBart50Tokenizer, MBart50TokenizerFast]


# -

def configure(argv=None):
    parser = HfArgumentParser((ModelArguments, DataTrainingArguments, NITrainingArguments))

    if is_in_notebook():
        model_args, data_args, training_args = parser.parse_args_into_dataclasses(args=argv.split())        
    else:
        if len(sys.argv) == 2 and sys.argv[1].endswith(".json"):
            # If we pass only one argument to the script and it's the path to a json file,
            # let's parse it to get our arguments.
            model_args, data_args, training_args = parser.parse_json_file(json_file=os.path.abspath(sys.argv[1]))
        else:
            model_args, data_args, training_args = parser.parse_args_into_dataclasses()
        argv = " ".join(sys.argv[1:])
    
    dataset_name = "ni"
    dir_name = args_to_output_dir(argv)
    model_dir = os.path.join(training_args.output_dir, dataset_name)
    training_args.output_dir = os.path.join(model_dir, dir_name)
    
    if training_args.final:
        data_args.max_num_instances_per_eval_task = 100
        training_args.eval_steps = 10000
        training_args.logging_steps = 10000
        training_args.save_steps = 10000
    
    data_args.data_dir = os.path.join("splits", data_args.data_dir)
    with open(os.path.join(data_args.data_dir, "train_tasks.txt"), "r") as f:
        tasks = [task for task in f.read().split("\n") if task != ""]
        data_args.n_task = len(tasks)
        
    if model_args.model_name_or_path is not None:
        model_name_or_path = model_args.model_name_or_path
    else:
        model_name_or_path = f"google/t5-{model_args.model}-lm-adapt"
        
    if model_args.model_name_or_path_prefix is not None:
        model_name_or_path_prefix = model_args.model_name_or_path_prefix
    else:
        model_name_or_path_prefix = f"google/t5-{model_args.model_prefix}-lm-adapt"
    
    # Setup logging
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        handlers=[logging.StreamHandler(sys.stdout)],
    )
    log_level = training_args.get_process_log_level()
    logger.setLevel(log_level)
    datasets.utils.logging.set_verbosity(log_level)
    transformers.utils.logging.set_verbosity(log_level)
    transformers.utils.logging.enable_default_handler()
    transformers.utils.logging.enable_explicit_format()

    # Log on each process the small summary:
    logger.warning(
        f"Process rank: {training_args.local_rank}, device: {training_args.device}, n_gpu: {training_args.n_gpu}"
        + f"distributed training: {bool(training_args.local_rank != -1)}, 16-bits training: {training_args.fp16}"
    )
    logger.info(f"Training/evaluation parameters {training_args}")

    if data_args.source_prefix is None and model_name_or_path in [
        "t5-small",
        "t5-base",
        "t5-large",
        "t5-3b",
        "t5-11b",
    ]:
        logger.warning(
            "You're running a t5 model but didn't provide a source prefix, which is the expected, e.g. with "
            "`--source_prefix 'summarize: ' `"
        )

    # Set seed before initializing model.
    set_seed(training_args.seed)

    # Load pretrained model and tokenizer
    tokenizer = AutoTokenizer.from_pretrained(
        model_args.tokenizer_name if model_args.tokenizer_name else model_name_or_path,
        cache_dir=model_args.cache_dir,
        use_fast=model_args.use_fast_tokenizer,
        revision=model_args.model_revision,
        use_auth_token=True if model_args.use_auth_token else None,
        proxies=PROXIES,
    )

    config = AutoConfig.from_pretrained(
        model_args.config_name if model_args.config_name else model_name_or_path,
        cache_dir=model_args.cache_dir,
        revision=model_args.model_revision,
        use_auth_token=True if model_args.use_auth_token else None,
        proxies=PROXIES,
    )
    
    if training_args.naturalinstruct or training_args.naturalmeta or training_args.nonprefix:
        if training_args.reweight:
            assert training_args.naturalmeta or training_args.nonprefix
            model_prefix = torch.nn.Parameter(torch.ones(data_args.n_task))
        else:
            model_prefix = None
    else:
        if training_args.prefix:
            model_prefix = torch.nn.Parameter(torch.randn(data_args.n_task, data_args.max_prefix_length, config.d_model))
        elif training_args.prefix_embeds:
            config_prefix = AutoConfig.from_pretrained(
                model_name_or_path_prefix,
                cache_dir=model_args.cache_dir,
                revision=model_args.model_revision,
                use_auth_token=True if model_args.use_auth_token else None,
                proxies=PROXIES,
            )
            if training_args.pretrained_prefix:
                model_prefix = T5EncoderModel.from_pretrained(
                    model_name_or_path_prefix,
                    from_tf=False,
                    config=config_prefix,
                    cache_dir=model_args.cache_dir,
                    revision=model_args.model_revision,
                    use_auth_token=True if model_args.use_auth_token else None,
                    proxies=PROXIES,
                )
            else:
                model_prefix = T5EncoderModel(
                    config=config_prefix,
                )
            model_prefix.resize_token_embeddings(len(tokenizer))
            model_prefix.linear = nn.Linear(config_prefix.d_model, config.d_model, bias=False)
        elif training_args.prefix_linear:
            config_prefix = AutoConfig.from_pretrained(
                model_name_or_path_prefix,
                cache_dir=model_args.cache_dir,
                revision=model_args.model_revision,
                use_auth_token=True if model_args.use_auth_token else None,
                proxies=PROXIES,
            )
            if training_args.pretrained_prefix:
                model_prefix = T5EncoderModel.from_pretrained(
                    model_name_or_path_prefix,
                    from_tf=False,
                    config=config_prefix,
                    cache_dir=model_args.cache_dir,
                    revision=model_args.model_revision,
                    use_auth_token=True if model_args.use_auth_token else None,
                    proxies=PROXIES,
                )
            else:
                model_prefix = T5EncoderModel(
                    config=config_prefix,
                )
            model_prefix.resize_token_embeddings(len(tokenizer))
            del model_prefix.encoder
            if training_args.dense:
                model_prefix.linear = nn.Sequential(
                    nn.ReLU(),
                    nn.Linear(config_prefix.d_model, config.d_model, bias=False),
                )
            else:
                model_prefix.linear = nn.Linear(config_prefix.d_model, config.d_model*data_args.max_prefix_length, bias=False)
        elif training_args.prefix_exemplar:
            config_prefix = AutoConfig.from_pretrained(
                model_name_or_path_prefix,
                cache_dir=model_args.cache_dir,
                revision=model_args.model_revision,
                use_auth_token=True if model_args.use_auth_token else None,
                proxies=PROXIES,
            )
            if training_args.pretrained_prefix:
                model_prefix = T5EncoderModel.from_pretrained(
                    model_name_or_path_prefix,
                    from_tf=False,
                    config=config_prefix,
                    cache_dir=model_args.cache_dir,
                    revision=model_args.model_revision,
                    use_auth_token=True if model_args.use_auth_token else None,
                    proxies=PROXIES,
                )
            else:
                model_prefix = T5EncoderModel(
                    config=config_prefix,
                )
            model_prefix.resize_token_embeddings(len(tokenizer))
            del model_prefix.encoder
            if training_args.dense:
                model_prefix.linear = nn.Sequential(
                    nn.ReLU(),
                    nn.Linear(config_prefix.d_model, config.d_model, bias=False),
                )
            else:
                model_prefix.linear = nn.Linear(config_prefix.d_model, config.d_model, bias=True)
        elif training_args.exemplar:
            model_prefix = torch.nn.Parameter(torch.ones(data_args.n_task, data_args.num_pos_examples, data_args.max_num_instances_per_prefix))
        elif training_args.exemplar_embeds:
            config_prefix = AutoConfig.from_pretrained(
                model_name_or_path_prefix,
                cache_dir=model_args.cache_dir,
                revision=model_args.model_revision,
                use_auth_token=True if model_args.use_auth_token else None,
                proxies=PROXIES,
            )
            if training_args.pretrained_prefix:
                model_prefix = T5EncoderModel.from_pretrained(
                    model_name_or_path_prefix,
                    from_tf=False,
                    config=config_prefix,
                    cache_dir=model_args.cache_dir,
                    revision=model_args.model_revision,
                    use_auth_token=True if model_args.use_auth_token else None,
                    proxies=PROXIES,
                )
            else:
                model_prefix = T5EncoderModel(
                    config=config_prefix,
                )
            model_prefix.resize_token_embeddings(len(tokenizer))
            model_prefix.bilinear_layer = nn.Bilinear(config_prefix.d_model, config_prefix.d_model, data_args.num_pos_examples, bias=False)
        elif training_args.exemplar_linear:
            config_prefix = AutoConfig.from_pretrained(
                model_name_or_path_prefix,
                cache_dir=model_args.cache_dir,
                revision=model_args.model_revision,
                use_auth_token=True if model_args.use_auth_token else None,
                proxies=PROXIES,
            )
            if training_args.pretrained_prefix:
                model_prefix = T5EncoderModel.from_pretrained(
                    model_name_or_path_prefix,
                    from_tf=False,
                    config=config_prefix,
                    cache_dir=model_args.cache_dir,
                    revision=model_args.model_revision,
                    use_auth_token=True if model_args.use_auth_token else None,
                    proxies=PROXIES,
                )
            else:
                model_prefix = T5EncoderModel(
                    config=config_prefix,
                )
            model_prefix.resize_token_embeddings(len(tokenizer))
            del model_prefix.encoder
            if training_args.dense:
                model_prefix.linear = nn.Sequential(
                    nn.Linear(config_prefix.d_model, config_prefix.d_ff, bias=False),
                    nn.ReLU(),
                    nn.Dropout(config.dropout_rate),
                )
                model_prefix.bilinear = nn.Bilinear(config_prefix.d_ff, config_prefix.d_ff, data_args.num_pos_examples, bias=False)
            else:
                model_prefix.linear = None
                model_prefix.bilinear = nn.Bilinear(config_prefix.d_model, config_prefix.d_model, data_args.num_pos_examples, bias=False)
    
    model = T5FrozenForConditionalGeneration.from_pretrained(
        model_name_or_path,
        model_prefix=model_prefix,
        from_tf=bool(".ckpt" in model_name_or_path),
        config=config,
        cache_dir=model_args.cache_dir,
        revision=model_args.model_revision,
        use_auth_token=True if model_args.use_auth_token else None,
        proxies=PROXIES,
    )
    model.resize_token_embeddings(len(tokenizer))
    
    if model.config.decoder_start_token_id is None and isinstance(tokenizer, (MBartTokenizer, MBartTokenizerFast)):
        if isinstance(tokenizer, MBartTokenizer):
            model.config.decoder_start_token_id = tokenizer.lang_code_to_id[data_args.lang]
        else:
            model.config.decoder_start_token_id = tokenizer.convert_tokens_to_ids(data_args.lang)

    if model.config.decoder_start_token_id is None:
        raise ValueError("Make sure that `config.decoder_start_token_id` is correctly defined")

    if (
        hasattr(model.config, "max_position_embeddings")
        and model.config.max_position_embeddings < data_args.max_source_length
    ):
        if model_args.resize_position_embeddings is None:
            logger.warning(
                f"Increasing the model's number of position embedding vectors from {model.config.max_position_embeddings} "
                f"to {data_args.max_source_length}."
            )
            model.resize_position_embeddings(data_args.max_source_length)
        elif model_args.resize_position_embeddings:
            model.resize_position_embeddings(data_args.max_source_length)
        else:
            raise ValueError(
                f"`--max_source_length` is set to {data_args.max_source_length}, but the model only has {model.config.max_position_embeddings}"
                f" position encodings. Consider either reducing `--max_source_length` to {model.config.max_position_embeddings} or to automatically "
                "resize the model's position encodings by passing `--resize_position_embeddings`."
            )

    if isinstance(tokenizer, tuple(MULTILINGUAL_TOKENIZERS)):
        assert (
            data_args.lang is not None
        ), f"{tokenizer.__class__.__name__} is a multilingual tokenizer which requires --lang argument"

        tokenizer.src_lang = data_args.lang
        tokenizer.tgt_lang = data_args.lang

        # For multilingual translation models like mBART-50 and M2M100 we need to force the target language token
        # as the first generated token. We ask the user to explicitly provide this as --forced_bos_token argument.
        forced_bos_token_id = (
            tokenizer.lang_code_to_id[data_args.forced_bos_token] if data_args.forced_bos_token is not None else None
        )
        model.config.forced_bos_token_id = forced_bos_token_id


    if training_args.label_smoothing_factor > 0 and not hasattr(model, "prepare_decoder_input_ids_from_labels"):
        logger.warning(
            "label_smoothing is enabled but the `prepare_decoder_input_ids_from_labels` method is not defined for"
            f"`{model.__class__.__name__}`. This will lead to loss being calculated twice and will take up more memory"
        )
        
    if training_args.init_instruction:
        model.init_instruction(data_args, tokenizer, prefix_ids=training_args.prefix_ids, prefix_embeds=training_args.prefix_embeds)
    elif training_args.init_exemplar:
        model.init_exemplar(data_args, tokenizer, prefix_ids=training_args.prefix_ids, prefix_embeds=training_args.prefix_embeds)
    elif training_args.init_vocab:
        model.init_vocab(training_args.n_vocab_init, prefix_ids=training_args.prefix_ids, prefix_embeds=training_args.prefix_embeds)
        
    # Get the NaturalInstructions dataset
    download_config = DownloadConfig(proxies=PROXIES)
    if training_args.other:
        raw_datasets = load_dataset(
            os.path.join(os.environ['HOME'], "latentprompt/blank_dataset.py"),
            data_dir=data_args.data_dir, 
            task_dir=data_args.task_dir, 
            cache_dir=model_args.cache_dir,
            max_num_instances_per_task=data_args.max_num_instances_per_task,
            max_num_instances_per_eval_task=data_args.max_num_instances_per_eval_task,
            max_num_instances_per_prefix=data_args.max_num_instances_per_prefix,
            download_config=download_config,
        )
    else:
        raw_datasets = load_dataset(
            os.path.join(os.environ['HOME'], "latentprompt/ni_dataset.py"),
            data_dir=data_args.data_dir, 
            task_dir=data_args.task_dir, 
            cache_dir=model_args.cache_dir,
            max_num_instances_per_task=data_args.max_num_instances_per_task,
            max_num_instances_per_eval_task=data_args.max_num_instances_per_eval_task,
            max_num_instances_per_prefix=data_args.max_num_instances_per_prefix,
            download_config=download_config,
        )

    if training_args.do_train:
        if "train" not in raw_datasets:
            raise ValueError("--do_train requires a train dataset")
        train_dataset = raw_datasets["train"]
        if data_args.max_train_samples is not None:
            train_dataset = train_dataset.select(range(data_args.max_train_samples))
        train_dataset = train_dataset.shuffle(training_args.seed) # NEW

    if not training_args.naturalinstruct:
        if "meta" not in raw_datasets:
            raise ValueError("--do_train requires a meta dataset")
        meta_dataset = raw_datasets["meta"]
        if data_args.max_train_samples is not None:
            meta_dataset = meta_dataset.select(range(data_args.max_train_samples))
        meta_dataset = meta_dataset.shuffle(training_args.seed) # NEW

    if training_args.do_val:
        if "validation" not in raw_datasets:
            raise ValueError("--do_eval requires a validation dataset")
        val_dataset = raw_datasets["validation"]
        if data_args.max_eval_samples is not None:
            val_dataset = val_dataset.select(range(data_args.max_eval_samples))

    if training_args.do_eval:
        if "test" not in raw_datasets:
            raise ValueError("--do_eval requires a test dataset")
        eval_dataset = raw_datasets["test"]
        if data_args.max_eval_samples is not None:
            eval_dataset = eval_dataset.select(range(data_args.max_eval_samples))

    # if training_args.do_eval:
    #     if "validation" not in raw_datasets:
    #         raise ValueError("--do_eval requires a validation dataset")
    # eval_dataset = raw_datasets["validation"]
    # if data_args.max_eval_samples is not None:
    #     eval_dataset = eval_dataset.select(range(data_args.max_eval_samples))

    if training_args.do_predict:
        if "test" not in raw_datasets:
            raise ValueError("--do_predict requires a test dataset")
        predict_dataset = raw_datasets["test"]
        if data_args.max_predict_samples is not None:
            predict_dataset = predict_dataset.select(range(data_args.max_predict_samples))

    # Data collator
    label_pad_token_id = -100 if data_args.ignore_pad_token_for_loss else tokenizer.pad_token_id
    data_collator = DataCollatorForNI(
        tokenizer=tokenizer,
        model=model,
        data_args=data_args,
        label_pad_token_id=label_pad_token_id,
        pad_to_multiple_of=8 if training_args.fp16 else None,
    )
    # we don't want to remove unused columns because we will prepare each batch during training, 
    # and some of the information will aslo be used in evaluation.
    training_args.remove_unused_columns = False

    # for debug in other
    def compute_acc_exemplar_embeds(dataset):
        dataloader = trainer.get_train_dataloader(dataset, batch_size=trainer.args.per_device_eval_batch_size)
        dataloader = iter(dataloader)
        
        acc_exemplars = []
        for _ in range(50):
            inputs = next(dataloader)
            inputs = trainer._prepare_inputs(inputs)
            model_prefix_inputs = model.prepare_inputs_prefix(inputs, ignore_keys=["labels", "decoder_input_ids"])

            prefix_outputs = model_prefix.encoder(**model_prefix_inputs)
            prefix_embeds = torch.sum(prefix_outputs.last_hidden_state * model_prefix_inputs["attention_mask"][:, :, None], 1) / torch.sum(model_prefix_inputs["attention_mask"][:, :, None], 1) # n_batch x d_emb

            model_exemplar_inputs = model.prepare_inputs_exemplar(inputs)
            prefix_exemplar_outputs = model_prefix.encoder(**model_exemplar_inputs)
            prefix_exemplar_embeds_flat = torch.sum(prefix_exemplar_outputs.last_hidden_state * model_exemplar_inputs["attention_mask"][:, :, None], 1) / torch.sum(model_exemplar_inputs["attention_mask"][:, :, None], 1)
            prefix_exemplar_embeds = prefix_exemplar_embeds_flat.view(prefix_embeds.size(0), -1, prefix_exemplar_embeds_flat.size(-1)) # n_batch x n_candidate x d_emb_

            logits_exemplar_ = model_prefix.bilinear_layer(prefix_exemplar_embeds, prefix_embeds[:, None, :].expand(-1, prefix_exemplar_embeds.size(1), -1)) # n_batch x n_candidate x n_exemplar
            logits_exemplar = logits_exemplar_.transpose(-1, -2) # n_batch x n_exemplar x n_candidate

            indices_exemplar = torch.argmax(logits_exemplar, -1)
            acc_exemplar = ((indices_exemplar == (logits_exemplar.size(-1)-1)).sum() / len(indices_exemplar)).item()
            acc_exemplars.append(acc_exemplar)
            
        acc_exemplar = np.mean(acc_exemplars)
        return acc_exemplar 
    
    # for debug in other
    def compute_acc_exemplar_linear(dataset):
        dataloader = trainer.get_train_dataloader(dataset, batch_size=trainer.args.per_device_eval_batch_size)
        dataloader = iter(dataloader)
        
        acc_exemplars = []
        for _ in range(50):
            inputs = next(dataloader)
            inputs = trainer._prepare_inputs(inputs)
            model_prefix_inputs = model.prepare_inputs_prefix(inputs, ignore_keys=["labels", "decoder_input_ids"])

            prefix_outputs = model_prefix.shared(model_prefix_inputs["input_ids"])
            prefix_embeds = torch.sum(prefix_outputs * model_prefix_inputs["attention_mask"][:, :, None], 1) / torch.sum(model_prefix_inputs["attention_mask"][:, :, None], 1) # n_batch x d_emb

            model_exemplar_inputs = model.prepare_inputs_exemplar(inputs)
            prefix_exemplar_outputs = model_prefix.shared(model_exemplar_inputs["input_ids"])
            prefix_exemplar_embeds_flat = torch.sum(prefix_exemplar_outputs * model_exemplar_inputs["attention_mask"][:, :, None], 1) / torch.sum(model_exemplar_inputs["attention_mask"][:, :, None], 1)
            prefix_exemplar_embeds = prefix_exemplar_embeds_flat.view(prefix_embeds.size(0), -1, prefix_exemplar_embeds_flat.size(-1)) # n_batch x n_candidate x d_emb_

            logits_exemplar_ = model_prefix.bilinear(prefix_exemplar_embeds, prefix_embeds[:, None, :].expand(-1, prefix_exemplar_embeds.size(1), -1)) # n_batch x n_candidate x n_exemplar
            logits_exemplar = logits_exemplar_.transpose(-1, -2) # n_batch x n_exemplar x n_candidate

            indices_exemplar = torch.argmax(logits_exemplar, -1)
            acc_exemplar = ((indices_exemplar == (logits_exemplar.size(-1)-1)).sum() / len(indices_exemplar)).item()
            acc_exemplars.append(acc_exemplar)
            
        acc_exemplar = np.mean(acc_exemplars)
        return acc_exemplar 
    
    # Metric
    def compute_ni_metrics(dataset, preds, labels=None, save_prefix=None):
        decoded_preds = tokenizer.batch_decode(preds, skip_special_tokens=True)
        references = [e["Instance"]["output"] for e in dataset]
        result = compute_metrics(predictions=decoded_preds, references=references)
        
        with torch.no_grad():
            if (not training_args.naturalinstruct) and training_args.other: 
                if training_args.exemplar_embeds:
                    result["acc"] = compute_acc_exemplar_embeds(trainer.train_dataset)
                elif training_args.exemplar_linear:
                    result["acc"] = compute_acc_exemplar_linear(trainer.train_dataset)
                elif training_args.exemplar:
                    result["acc"] = ((torch.argmax(model.model_prefix, -1) == (data_args.max_num_instances_per_prefix - 1)).sum() / len(model.model_prefix)).item()
        result_per_task = compute_grouped_metrics(predictions=decoded_preds, references=references, groups=dataset["Task"])
        result.update(result_per_task)
        categories = ["_".join(it[0].lower().split()) for it in dataset["Categories"]]
        result_per_category = compute_grouped_metrics(predictions=decoded_preds, references=references, groups=categories)
        result.update(result_per_category)
        prediction_lens = [np.count_nonzero(pred != tokenizer.pad_token_id) for pred in preds]
        result["gen_len"] = np.mean(prediction_lens)
        result = {k: round(v, 4) for k, v in result.items()}
        if save_prefix is not None:
            with open(os.path.join(training_args.output_dir, f"{save_prefix}_eval_predictions.jsonl"), "w") as fout:
                for example, pred in zip(dataset, decoded_preds):
                    fout.write(json.dumps({
                        "Task": example["Task"],
                        "Definition": example["Definition"],
                        "Instance": example["Instance"],
                        "Prediction": pred
                    }) + "\n")
        return result
    
    # Initialize our Trainer
    if training_args.naturalinstruct:
        trainer = NITrainer(
            val_dataset=val_dataset if training_args.do_val else None,
            model=model,
            args=training_args,
            train_dataset=train_dataset if training_args.do_train else None,
            eval_dataset=eval_dataset if training_args.do_eval else None,
            tokenizer=tokenizer,
            data_collator=data_collator,
            compute_metrics=compute_ni_metrics if training_args.predict_with_generate else None,
            callbacks=[DenserEvalCallback] if training_args.denser_evaluation else None
        )
    elif training_args.ift or training_args.naturalmeta or training_args.nonprefix:
        trainer = IFTTrainer(
            meta_dataset=meta_dataset,
            val_dataset=val_dataset if training_args.do_val else None,
            model=model,
            args=training_args,
            train_dataset=train_dataset if training_args.do_train else None,
            eval_dataset=eval_dataset if training_args.do_eval else None,
            tokenizer=tokenizer,
            data_collator=data_collator,
            compute_metrics=compute_ni_metrics if training_args.predict_with_generate else None,
            callbacks=[DenserEvalCallback] if training_args.denser_evaluation else None
        )

    return trainer, data_args


# +
def run(trainer, data_args):
    all_metrics = {"run_name": trainer.args.run_name}
    # Detecting last checkpoint.
    last_checkpoint = None
    if os.path.isdir(trainer.args.output_dir) and trainer.args.do_train and not trainer.args.overwrite_output_dir:
        last_checkpoint = get_last_checkpoint(trainer.args.output_dir)
        if last_checkpoint is None and len(os.listdir(trainer.args.output_dir)) > 0:
            raise ValueError(
                f"Output directory ({trainer.args.output_dir}) already exists and is not empty. "
                "Use --overwrite_output_dir to overcome."
            )
        elif last_checkpoint is not None and trainer.args.resume_from_checkpoint is None:
            logger.info(
                f"Checkpoint detected, resuming training at {last_checkpoint}. To avoid this behavior, change "
                "the `--output_dir` or add `--overwrite_output_dir` to train from scratch."
            )
            
    # Training
    if trainer.args.do_train:
        checkpoint = None
        if trainer.args.resume_from_checkpoint is not None:
            checkpoint = trainer.args.resume_from_checkpoint
        elif last_checkpoint is not None:
            checkpoint = last_checkpoint
        train_result = trainer.train(resume_from_checkpoint=checkpoint)
        trainer.save_model()  # Saves the tokenizer too for easy upload

        metrics = train_result.metrics
        max_train_samples = (
            data_args.max_train_samples if data_args.max_train_samples is not None else len(trainer.train_dataset)
        )
        metrics["train_samples"] = min(max_train_samples, len(trainer.train_dataset))

        trainer.log_metrics("train", metrics)
        trainer.save_metrics("train", metrics)
        trainer.save_state()

        all_metrics.update(metrics)
        
    # Evaluation
    if trainer.args.do_eval:
        download_config = DownloadConfig(proxies=PROXIES)
        raw_datasets = load_dataset(
            os.path.join(os.environ['HOME'], "latentprompt/ni_dataset.py"),
            data_dir=data_args.data_dir, 
            task_dir=data_args.task_dir, 
            cache_dir=None,
            max_num_instances_per_task=data_args.max_num_instances_per_task,
            max_num_instances_per_eval_task=100,
            max_num_instances_per_prefix=data_args.max_num_instances_per_prefix,
            download_config=download_config,
        )

        if "test" not in raw_datasets:
            raise ValueError("--do_eval requires a test dataset")
        eval_dataset = raw_datasets["test"]
        if data_args.max_eval_samples is not None:
            eval_dataset = eval_dataset.select(range(data_args.max_eval_samples))
            
        trainer.eval_dataset = eval_dataset
        
        max_length = (
        trainer.args.generation_max_length
        if trainer.args.generation_max_length is not None
        else data_args.max_target_length
        )
        num_beams = data_args.num_beams if data_args.num_beams is not None else trainer.args.generation_num_beams

        logger.info("*** Evaluate ***")
        metrics = trainer.evaluate(max_length=max_length, num_beams=num_beams, metric_key_prefix="eval")
        max_eval_samples = data_args.max_eval_samples if data_args.max_eval_samples is not None else len(trainer.eval_dataset)
        metrics["eval_samples"] = min(max_eval_samples, len(trainer.eval_dataset))

        trainer.log_metrics("eval", metrics)
        trainer.save_metrics("eval", metrics)

        all_metrics.update(metrics)
        
#     # Predict
#     if trainer.args.do_predict:
#         logger.info("*** Predict ***")

#         predict_results = trainer.predict(
#             predict_dataset, metric_key_prefix="predict", max_length=max_length, num_beams=num_beams
#         )
#         metrics = predict_results.metrics
#         max_predict_samples = (
#             data_args.max_predict_samples if data_args.max_predict_samples is not None else len(predict_dataset)
#         )
#         metrics["predict_samples"] = min(max_predict_samples, len(predict_dataset))

#         trainer.log(metrics)
#         trainer.log_metrics("predict", metrics)
#         trainer.save_metrics("predict", metrics)

#         all_metrics.update(metrics)

    return all_metrics


# -

if __name__ == '__main__':
    trainer, data_args = configure()
    run(trainer, data_args)
