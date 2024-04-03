# +
# %load_ext autoreload
# %autoreload
import logging
import os
os.environ["CUDA_VISIBLE_DEVICES"] = "2"
import sys
import json
import random
import pdb
from collections import defaultdict

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

from modeling_t5_frozen import T5FrozenForConditionalGeneration
from ni_collator import DataCollatorForNI
from ni_trainer import NITrainer, DenserEvalCallback
from toy_trainer import ToyTrainer
from toy_ho_trainer import ToyHOTrainer
from toy_ift_trainer import ToyIFTTrainer
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


# +
class Model(nn.Module):
    def __init__(self):
        super().__init__()
        self.w = nn.Parameter(torch.tensor(-10.), requires_grad=True)
        
    def get_prefix_named_parameters(self):
        return [(k, v) for k, v in self.named_parameters() if "model_prefix" in k]
    
    def get_prefix_parameters(self):
        return [v for k, v in self.get_prefix_named_parameters()]
    
    def get_frozen_named_parameters(self):
        return [(k, v) for k, v in self.named_parameters() if "model_prefix" not in k]
    
    def get_frozen_parameters(self):
        return [v for k, v in self.get_frozen_named_parameters()]
    
class ModelPrefix(nn.Module):
    def __init__(self):
        super().__init__()
        self.a = nn.Parameter(torch.tensor(3.), requires_grad=True)


# -

argv = '''
--ift
--model base
--learning_rate 0.1
--learning_rate_ift 0.4
--k_ift 5
--meta_steps 100
--logging_steps 10
--eval_steps 10
--save_steps 10
'''

# +
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

dataset_name = "toy"
dir_name = args_to_output_dir(argv)
training_args.output_dir = os.path.join(training_args.output_dir, dataset_name, dir_name)

data_args.data_dir = os.path.join("data/splits", data_args.data_dir)
with open(os.path.join(data_args.data_dir, "train_tasks.txt"), "r") as f:
    tasks = [task for task in f.read().split("\n") if task != ""]
    data_args.n_task = len(tasks)

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

if model_args.model_name_or_path is not None:
    model_name_or_path = model_args.model_name_or_path
else:
    model_name_or_path = f"google/t5-{model_args.model}-lm-adapt"

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
tokenizer.add_special_tokens({"sep_token": "<sep>"})

if training_args.naturalinstruct or training_args.naturalmeta or training_args.nonprefix:
    model_prefix = None
else:
    model_prefix = ModelPrefix()

model = Model()
model.model_prefix = model_prefix

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

# Get the NaturalInstructions dataset
if training_args.naturalinstruct:
    path_dataset_py = os.path.join(os.environ['HOME'], "latentprompt/ni_dataset.py")
else:
    path_dataset_py = os.path.join(os.environ['HOME'], "latentprompt/meta_dataset.py")

download_config = DownloadConfig(proxies=PROXIES)
raw_datasets = load_dataset(
    path_dataset_py,
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

# if training_args.do_predict:
#     if "test" not in raw_datasets:
#         raise ValueError("--do_predict requires a test dataset")
# predict_dataset = raw_datasets["test"]
# if data_args.max_predict_samples is not None:
#     predict_dataset = predict_dataset.select(range(data_args.max_predict_samples))

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

# Metric
def compute_ni_metrics(dataset, preds, labels=None, save_prefix=None):
    decoded_preds = tokenizer.batch_decode(preds, skip_special_tokens=True)
    references = [e["Instance"]["output"] for e in dataset]
    result = compute_metrics(predictions=decoded_preds, references=references)
#     result_per_task = compute_grouped_metrics(predictions=decoded_preds, references=references, groups=dataset["Task"])
#     result.update(result_per_task)
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
# trainer = ToyTrainer(
#     meta_dataset=meta_dataset,
#     val_dataset=val_dataset if training_args.do_val else None,
#     model=model,
#     args=training_args,
#     train_dataset=train_dataset if training_args.do_train else None,
#     eval_dataset=eval_dataset if training_args.do_eval else None,
#     tokenizer=tokenizer,
#     data_collator=data_collator,
#     compute_metrics=compute_ni_metrics if training_args.predict_with_generate else None,
#     callbacks=[DenserEvalCallback] if training_args.denser_evaluation else None
# )
# trainer = ToyHOTrainer(
#     meta_dataset=meta_dataset,
#     val_dataset=val_dataset if training_args.do_val else None,
#     model=model,
#     args=training_args,
#     train_dataset=train_dataset if training_args.do_train else None,
#     eval_dataset=eval_dataset if training_args.do_eval else None,
#     tokenizer=tokenizer,
#     data_collator=data_collator,
#     compute_metrics=compute_ni_metrics if training_args.predict_with_generate else None,
#     callbacks=[DenserEvalCallback] if training_args.denser_evaluation else None
# )
trainer = ToyIFTTrainer(
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


all_metrics = {"run_name": training_args.run_name}

# Detecting last checkpoint.
last_checkpoint = None
if os.path.isdir(training_args.output_dir) and training_args.do_train and not training_args.overwrite_output_dir:
    last_checkpoint = get_last_checkpoint(training_args.output_dir)
    if last_checkpoint is None and len(os.listdir(training_args.output_dir)) > 0:
        raise ValueError(
            f"Output directory ({training_args.output_dir}) already exists and is not empty. "
            "Use --overwrite_output_dir to overcome."
        )
    elif last_checkpoint is not None and training_args.resume_from_checkpoint is None:
        logger.info(
            f"Checkpoint detected, resuming training at {last_checkpoint}. To avoid this behavior, change "
            "the `--output_dir` or add `--overwrite_output_dir` to train from scratch."
        )

max_length = (
    training_args.generation_max_length
    if training_args.generation_max_length is not None
    else data_args.max_target_length
)
num_beams = data_args.num_beams if data_args.num_beams is not None else training_args.generation_num_beams
# -

# %autoreload
from toy_ift_trainer import ToyIFTTrainer

# Training
if training_args.do_train:
    checkpoint = None
    if training_args.resume_from_checkpoint is not None:
        checkpoint = training_args.resume_from_checkpoint
    elif last_checkpoint is not None:
        checkpoint = last_checkpoint
    train_result = trainer.train(resume_from_checkpoint=checkpoint)
    trainer.save_model()  # Saves the tokenizer too for easy upload

    metrics = train_result.metrics
    max_train_samples = (
        data_args.max_train_samples if data_args.max_train_samples is not None else len(train_dataset)
    )
    metrics["train_samples"] = min(max_train_samples, len(train_dataset))

    trainer.log_metrics("train", metrics)
    trainer.save_metrics("train", metrics)
    trainer.save_state()

    all_metrics.update(metrics)

# Evaluation
if training_args.do_eval:
    logger.info("*** Evaluate ***")
    metrics = trainer.evaluate(max_length=max_length, num_beams=num_beams, metric_key_prefix="eval")
    max_eval_samples = data_args.max_eval_samples if data_args.max_eval_samples is not None else len(eval_dataset)
    metrics["eval_samples"] = min(max_eval_samples, len(eval_dataset))

    trainer.log_metrics("eval", metrics)
    trainer.save_metrics("eval", metrics)

    all_metrics.update(metrics)
