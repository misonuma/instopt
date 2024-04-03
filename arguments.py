# +
import os
from dataclasses import dataclass, field
from typing import Optional, Union
from transformers import Seq2SeqTrainingArguments
# from transformers.training_args import OptimizerNames
from transformers.utils import ExplicitEnum
from transformers.trainer_utils import EvaluationStrategy, HubStrategy, IntervalStrategy, SchedulerType, ShardedDDPOption

ABBR_DICT = {
        "per_device_train_batch_size": "train_batch",
        "per_device_eval_batch_size": "eval_batch",
        "per_device_eval_batch_size": "eval_batch",
        "learning_rate": "lr",
        "warmup_steps": "warmup",
        "logging_steps": "log",
        "save_steps": "save",
        "eval_steps": "eval",
        "max_num_instances_per_eval_task": "num_eval",
    }


# -

def args_to_output_dir(argv, ignore_arg=None):
    args = argv.strip().split("--")[1:]
    
    if "model_name_or_path_prefix" in argv:
        model_name_or_path_prefix = [arg for arg in args if "model_name_or_path_prefix" in arg][0].split()[-1]
        checkpoint = [arg for arg in args if "checkpoint" in arg][0].split()[-1]
        output_dir = f"{model_name_or_path_prefix}-checkpoint={checkpoint}"
    else:
        output_dir = "-".join([arg.strip().replace(" ", "=").replace("model=", "") for arg in args])        
        for arg, abbr in ABBR_DICT.items():
            output_dir = output_dir.replace(arg, abbr)
    return output_dir


class OptimizerNames(ExplicitEnum):
    """
    Stores the acceptable string identifiers for optimizers.
    """
    ADAMW_HF = "adamw_hf"
    ADAMW_TORCH = "adamw_torch"
    ADAMW_TORCH_XLA = "adamw_torch_xla"
    ADAMW_APEX_FUSED = "adamw_apex_fused"
    ADAFACTOR = "adafactor"
    ADAMW_BNB = "adamw_bnb_8bit"
    SGD = "sgd"
    ADAGRAD = "adagrad"


@dataclass
class ModelArguments:
    """
    Arguments pertaining to which model/config/tokenizer we are going to fine-tune from.
    """
    model_name_or_path: Optional[str] = field(
        default=None, 
        metadata={"help": "Path to pretrained model or model identifier from huggingface.co/models"}
    )
    model_name_or_path_prefix: str = field(
        default=None,
        metadata={"help": "Path to pretrained model or model identifier from huggingface.co/models"}
    )
    checkpoint: str = field(
        default=None,
    )
    model: Optional[str] = field(
        default=None, 
    )
    model_prefix: Optional[str] = field(
        default="small", 
    )
    config_name: Optional[str] = field(
        default=None, metadata={"help": "Pretrained config name or path if not the same as model_name"}
    )
    tokenizer_name: Optional[str] = field(
        default=None, metadata={"help": "Pretrained tokenizer name or path if not the same as model_name"}
    )
    cache_dir: Optional[str] = field(
        default=None,
        metadata={"help": "Where to store the pretrained models downloaded from huggingface.co"},
    )
    use_fast_tokenizer: bool = field(
        default=True,
        metadata={"help": "Whether to use one of the fast tokenizer (backed by the tokenizers library) or not."},
    )
    model_revision: str = field(
        default="main",
        metadata={"help": "The specific model version to use (can be a branch name, tag name or commit id)."},
    )
    use_auth_token: bool = field(
        default=False,
        metadata={
            "help": "Will use the token generated when running `transformers-cli login` (necessary to use this script "
            "with private models)."
        },
    )
    resize_position_embeddings: Optional[bool] = field(
        default=None,
        metadata={
            "help": "Whether to automatically resize the position embeddings if `max_source_length` exceeds "
            "the model's position embeddings."
        },
    )


@dataclass
class DataTrainingArguments:
    """
    Arguments pertaining to what data we are going to input our model for training and eval.
    """
    lang: str = field(default=None, metadata={"help": "Language id for multilingual model."})
    data_dir: str = field(
        default="default", metadata={"help": "The directory for saving the NaturalInstructions train/dev/test splits."}
    )
    task_dir: str = field(
        default="data/tasks", metadata={"help": "The directory for saving the NaturalInstructions tasks json files."}
    )
    n_task: Optional[int] = field(
        default=None,
    )
        
    overwrite_cache: bool = field(
        default=False, metadata={"help": "Overwrite the cached training and evaluation sets"}
    )
    preprocessing_num_workers: Optional[int] = field(
        default=None,
        metadata={"help": "The number of processes to use for the preprocessing."},
    )
    max_source_length: Optional[int] = field(
        default=1024,
        metadata={
            "help": "The maximum total input sequence length after tokenization. Sequences longer "
            "than this will be truncated, sequences shorter will be padded."
        },
    )
    max_frozen_source_length: Optional[int] = field(
        default=None,
    )
    max_target_length: Optional[int] = field(
        default=128,
        metadata={
            "help": "The maximum total sequence length for target text after tokenization. Sequences longer "
            "than this will be truncated, sequences shorter will be padded."
        },
    )
    pad_to_max_length: bool = field(
        default=False,
        metadata={
            "help": "Whether to pad all samples to model maximum sentence length. "
            "If False, will pad the samples dynamically when batching to the maximum length in the batch. More "
            "efficient on GPU but very bad for TPU."
        },
    )
    max_num_instances_per_task: int = field(
        default=100, metadata={"help": "The maximum number of instances we will consider for each training task."}
    )
    max_num_instances_per_eval_task: int = field(
        default=10, metadata={"help": "The maximum number of instances we will consider for each validation/test task."}
    )
    max_num_instances_per_meta_task: int = field(
        default=None, metadata={"help": "The maximum number of instances we will consider for each training task."}
    )
    max_num_instances_per_meta_category: int = field(
        default=None, metadata={"help": "The maximum number of instances we will consider for each training task."}
    )
    max_num_instances_per_prefix: int = field(
        default=32, metadata={"help": "The maximum number of instances we will consider for prefix generation."}
    )
    max_train_samples: Optional[int] = field(
        default=None,
        metadata={
            "help": "For debugging purposes or quicker training, truncate the number of training examples to this "
            "value if set."
        },
    )
    max_eval_samples: Optional[int] = field(
        default=None,
        metadata={
            "help": "For debugging purposes or quicker training, truncate the number of evaluation examples to this "
            "value if set."
        },
    )
    max_predict_samples: Optional[int] = field(
        default=None,
        metadata={
            "help": "For debugging purposes or quicker training, truncate the number of prediction examples to this "
            "value if set."
        },
    )
    num_beams: Optional[int] = field(
        default=None,
        metadata={
            "help": "Number of beams to use for evaluation. This argument will be passed to ``model.generate``, "
            "which is used during ``evaluate`` and ``predict``."
        },
    )
    ignore_pad_token_for_loss: bool = field(
        default=True,
        metadata={
            "help": "Whether to ignore the tokens corresponding to padded labels in the loss computation or not."
        },
    )
    source_prefix: Optional[str] = field(
        default="", metadata={"help": "A prefix to add before every source text (useful for T5 models)."}
    )

    forced_bos_token: Optional[str] = field(
        default=None,
        metadata={
            "help": "The token to force as the first generated token after the decoder_start_token_id."
            "Useful for multilingual models like mBART where the first generated token"
            "needs to be the target language token (Usually it is the target language token)"
        },
    )
    add_task_name: Optional[bool] = field(
        default=False,
        metadata={"help": "whether to preappend task name before the task input."}
    )
    add_task_definition: Optional[bool] = field(
        default=True,
        metadata={"help": "whether to preappend task definition before the task input."}
    )
    num_pos_examples: Optional[int] = field(
        default=0,
        metadata={"help": "number of in-context positive examples."}
    )
    num_neg_examples: Optional[int] = field(
        default=0,
        metadata={"help": "number of in-context negative examples."}
    )
    add_explanation: Optional[bool] = field(
        default=False,
        metadata={"help": "whether to add explanation for both the postive examples and negtive examples."}
    )
    add_task_definition_train: Optional[bool] = field(
        default=None,
        metadata={"help": "whether to preappend task definition before the task input."}
    )
    num_pos_examples_train: Optional[int] = field(
        default=None,
        metadata={"help": "number of in-context positive examples."}
    )
    num_neg_examples_train: Optional[int] = field(
        default=None,
        metadata={"help": "number of in-context negative examples."}
    )
    add_explanation_train: Optional[bool] = field(
        default=None,
        metadata={"help": "whether to add explanation for both the postive examples and negtive examples."}
    )
        
    random_examples: Optional[bool] = field(
        default=False,
    )
    random_instance_examples: Optional[bool] = field(
        default=False,
    )
    random_instance_exemplars: Optional[bool] = field(
        default=False,
    )
    example_index: Optional[int] = field(
        default=None,
    )
    random_text: Optional[bool] = field(
        default=False,
    )
    
    max_prefix_length: Optional[int] = field(
        default=128,
    )
    max_exemplar_length: Optional[int] = field(
        default=128,
    )
    instructtune: bool = field(
        default=False, 
    )
    instructadd: bool = field(
        default=False, 
    )
        
    train_dir: str = field(
        default="t0_train_dataset",
    )
    meta_dir: str = field(
        default="t0_train_dataset",
    )
    val_dir: str = field(
        default="t0_validation_dataset_tmp",
    )
    eval_dir: str = field(
        default="t0_eval_dataset_tmp",
    )
        
    dataset_name: str = field(
        default=None,
    )
    dataset_config_name: str = field(
        default=None,
    )
    template_name: str = field(
        default=None,
    )
            
    def __post_init__(self):
        pass


@dataclass
class NITrainingArguments(Seq2SeqTrainingArguments):
    # super arguments
    output_dir: str = field(
        default="model",
        metadata={"help": "The output directory where the model predictions and checkpoints will be written."},
    )
    overwrite_output_dir: bool = field(
        default=True,
        metadata={
            "help": (
                "Overwrite the content of the output directory. "
                "Use this to continue training if output_dir points to a checkpoint directory."
            )
        },
    )
    do_train: bool = field(default=True, metadata={"help": "Whether to run training."})
    do_eval: bool = field(default=True, metadata={"help": "Whether to run eval on the dev set."})
    do_predict: bool = field(default=False, metadata={"help": "Whether to run predictions on the test set."})
    predict_with_generate: bool = field(
        default=True, metadata={"help": "Whether to use generate to calculate generative metrics (ROUGE, BLEU)."}
    )
    generation_max_length: Optional[int] = field(
        default=128,
        metadata={
            "help": (
                "The `max_length` to use on each evaluation loop when `predict_with_generate=True`. Will default "
                "to the `max_length` value of the model configuration."
            )
        },
    )
    per_device_train_batch_size: int = field(
        default=2, metadata={"help": "Batch size per GPU/TPU core/CPU for training."}
    )
    per_device_eval_batch_size: int = field(
        default=16, metadata={"help": "Batch size per GPU/TPU core/CPU for evaluation."}
    )
    meta_batch_size: Optional[int] = field(
        default=None, metadata={"help": "Batch size per GPU/TPU core/CPU for training."}
    )
#     lr_scheduler_type: SchedulerType = field(
#         default="constant",
#         metadata={"help": "The scheduler type to use."},
#     )
    logging_steps: int = field(default=1000, metadata={"help": "Log every X updates steps."})
    save_steps: int = field(default=1000, metadata={"help": "Save checkpoint every X updates steps."})
    save_total_limit: Optional[int] = field(
        default=10,
        metadata={
            "help": (
                "Limit the total amount of checkpoints. "
                "Deletes the older checkpoints in the output_dir. Default is unlimited checkpoints"
            )
        },
    )
    eval_steps: int = field(default=1000, metadata={"help": "Run an evaluation every X steps."})
    evaluation_strategy: IntervalStrategy = field(
        default="steps",
        metadata={"help": "The evaluation strategy to use."},
    )
        
    # original arguments
    t0: bool = field(
        default=False, 
    )
        
    fometa: bool = field(
        default=False, 
    )
    bilevel: bool = field(
        default=False, 
    )
    naturalinstruct: bool = field(
        default=False, 
        metadata={"help": "naturalinstruction or not"}
    )
    ho: bool = field(
        default=False, 
    )
    ift: bool = field(
        default=False, 
    )

    prefix: bool = field(
        default=False,
    )
    prefix_embeds: bool = field(
        default=False,
    )
    prefix_linear: bool = field(
        default=False, 
    )
    prefix_exemplar: bool = field(
        default=False, 
    )
    exemplar: bool = field(
        default=False, 
    )
    exemplar_embeds: bool = field(
        default=False, 
    )
    exemplar_linear: bool = field(
        default=False, 
    )
    reweight: bool = field(
        default=False, 
    )
    naturalmeta: bool = field(
        default=False, 
        metadata={"help": "naturalinstruction or not"}
    )
    nonprefix: bool = field(
        default=False, 
        metadata={"help": "naturalinstruction or not"}
    )
        
    pretrained_prefix: bool = field(
        default=False,
    )
    dense: bool = field(
        default=False,
    )
        
    init_instruction: bool = field(
        default=False,
    )
    init_exemplar: bool = field(
        default=False,
    )
    init_vocab: bool = field(
        default=False,
    )
    init_category: bool = field(
        default=False,
    )
    init_category_vocab: bool = field(
        default=False,
    )
    n_vocab_init: int = field(
        default=5000,
    )
        
    temperature: float = field(
        default=1.,
    )
    hard: bool = field(
        default=True,
    )
    mask: bool = field(
        default=False,
    )
    
    meta_gradient_accumulation_steps: int = field(
        default=1,
    )
#     inner_gradient_accumulation_steps: int = field(
#         default=1,
#     )
        
    do_val: bool = field(default=False, metadata={"help": "Whether to run validate on the dev set."})
        
    denser_evaluation: Optional[bool] = field(
        default=False,
        metadata={"help": "If specifid, the model will do more evaluation at the beginning of training."}
    )
    do_demo: bool = field(
        default=False, 
        metadata={"help": "Whether to run the model as a demo in the terminal."}
    )
        
    optim: OptimizerNames = field(
        default="adamw_torch",
        metadata={"help": "The optimizer to use."},
    )
    learning_rate: float = field(
        default=1e-6, 
        metadata={"help": "The initial learning rate for AdamW."}
    )
    max_grad_norm: float = field(
        default=1.0, metadata={"help": "Max gradient norm."}
    )
    learning_rate_meta: Optional[float] = field(
        default=None, 
    )
    warmup_steps_meta: int = field(default=0)
    max_grad_norm_meta: float = field(
        default=1.0, metadata={"help": "Max gradient norm."}
    )
    noise_prefix: float = field(
        default=0., 
    )
    eps: float = field(
        default=1e-2, 
    )
#     num_train_epochs: float = field(default=5.0, metadata={"help": "Total number of training epochs to perform."})
#     max_steps: int = field(
#         default=-1,
#         metadata={"help": "If > 0: set total number of training steps to perform. Override num_train_epochs."},
#     )
        
    train_frozen_only: bool = field(
        default=False,
    )
    train_prefix_only: bool = field(
        default=False,
    )
    bilinear_only: bool = field(
        default=False,
    )
        
    parallelize: bool = field(
        default=False,
    )
        
    meta_steps: int = field(
        default=1,
    )
    k_ift: int = field(
        default=1,
    )
    learning_rate_ift: Optional[float] = field(
        default=None, 
    )
    main_steps: int = field(
        default=0,
    )
    
    other: bool = field(
        default=False, 
    )
    blank: bool = field(
        default=False, 
    )
        
    final: bool = field(
        default=False, 
    )
        
    debug_trainer: bool = field(
        default=False, 
    )
    debug_hvp: bool = field(
        default=False, 
    )
    tmp: bool = field(
        default=False, 
    )
