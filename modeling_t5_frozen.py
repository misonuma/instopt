# +
# coding=utf-8
# Copyright 2018 Mesh TensorFlow authors, T5 Authors and HuggingFace Inc. team.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import copy
import math
import os
import warnings
from typing import Optional, Tuple, Union, Dict, Any
import json
import string
import pdb

import numpy as np
import torch
from torch import nn
from torch.nn import CrossEntropyLoss
from torch.utils.checkpoint import checkpoint

from transformers.activations import ACT2FN
from transformers.modeling_outputs import (
    BaseModelOutput,
    BaseModelOutputWithPastAndCrossAttentions,
    Seq2SeqLMOutput,
    Seq2SeqModelOutput,
)
from transformers.modeling_utils import PreTrainedModel, find_pruneable_heads_and_indices, prune_linear_layer
from transformers.utils import (
    DUMMY_INPUTS,
    DUMMY_MASK,
    add_start_docstrings,
    add_start_docstrings_to_model_forward,
    is_torch_fx_proxy,
    logging,
    replace_return_docstrings,
)
from transformers.utils.model_parallel_utils import assert_device_map, get_device_map
from transformers.models.t5.modeling_t5 import T5PreTrainedModel, T5Stack, T5ForConditionalGeneration
from transformers.models.t5.configuration_t5 import T5Config


logger = logging.get_logger(__name__)

_CONFIG_FOR_DOC = "T5Config"
_TOKENIZER_FOR_DOC = "T5Tokenizer"
_CHECKPOINT_FOR_DOC = "t5-small"

T5_PRETRAINED_MODEL_ARCHIVE_LIST = [
    "t5-small",
    "t5-base",
    "t5-large",
    "t5-3b",
    "t5-11b",
    # See all T5 models at https://huggingface.co/models?filter=t5
]


# -

class T5PrefixEncoderModel(T5PreTrainedModel):
    def __init__(self, config: T5Config, n_task: int, max_prefix_length: int):
        super().__init__(config)
        self.model_dim = config.d_model
        
        self.prefix_embeds = torch.nn.Parameter(torch.randn(n_task, max_prefix_length, config.d_model))

        encoder_config = copy.deepcopy(config)
        encoder_config.is_decoder = False
        encoder_config.use_cache = False
        encoder_config.is_encoder_decoder = False
        self.encoder = T5Stack(encoder_config)

        self.lm_head = nn.Linear(config.d_model, config.vocab_size, bias=False)

        # Initialize weights and apply final processing
        self.post_init()

        # Model parallel
        self.model_parallel = False
        self.device_map = None

    def set_output_embeddings(self, new_embeddings):
        self.lm_head = new_embeddings

    def get_output_embeddings(self):
        return self.lm_head
        
    def _resize_token_embeddings(self, new_num_tokens):
        # if word embeddings are not tied, make sure that lm head is resized as well
        if self.get_output_embeddings() is not None and not self.config.tie_word_embeddings:
            old_lm_head = self.get_output_embeddings()
            new_lm_head = self._get_resized_lm_head(old_lm_head, new_num_tokens)
            self.set_output_embeddings(new_lm_head)

        return self.get_output_embeddings()


class T5FrozenForConditionalGeneration(T5ForConditionalGeneration):
    def __init__(self, config: T5Config, model_prefix: Union[PreTrainedModel, nn.Module]):
        super().__init__(config)
        self.model_prefix = model_prefix
        self.encoder.main_input_name = "inputs_embeds"
        pdb.set_trace()
        
    def prepare_inputs(self, inputs):
        model_inputs = {
            "inputs_embeds": self.shared(inputs["input_ids"]),
            "attention_mask": inputs["attention_mask"],
            "labels": inputs["labels_frozen"],
            "decoder_input_ids": inputs["decoder_input_ids_frozen"],
        }
        return model_inputs
        
    def prepare_inputs_choice(self, inputs) -> torch.Tensor:
        model_inputs = {
            "inputs_embeds": self.shared(inputs["input_ids"].repeat_interleave(inputs["n_choice"], dim=0)),
            "attention_mask": inputs["attention_mask"].repeat_interleave(inputs["n_choice"], dim=0),
            "decoder_input_ids": inputs["decoder_input_ids_choices"],
        }
        assert len(model_inputs["inputs_embeds"]) == len(model_inputs["attention_mask"]) == len(model_inputs["decoder_input_ids"])
        return model_inputs
        
    def prepare_inputs_prefix(self, inputs, ignore_keys=[]):
        model_prefix_inputs = {k.replace("_prefix", ""): v for k, v in inputs.items() if "_prefix" in k}
        model_prefix_inputs = {k: v for k, v in model_prefix_inputs.items() if k not in ignore_keys}
        return model_prefix_inputs
    
    def prepare_inputs_exemplar(self, inputs):
        model_exemplar_inputs = {k.replace("_exemplar", ""): v for k, v in inputs.items() if "_exemplar" in k}
        return model_exemplar_inputs
    
    def prepare_inputs_frozen(self, inputs, prefix_embeds, prefix_attention_mask=None):
        frozen_embeds = self.shared(inputs["input_ids_frozen"]) # batch_l x inputs_l x n_emb
        inputs_embeds = torch.concat([prefix_embeds, frozen_embeds], 1)

        frozen_attention_mask = inputs["attention_mask_frozen"]
        if prefix_attention_mask is None:
            prefix_attention_mask = torch.ones(prefix_embeds.size(0), prefix_embeds.size(1), dtype=frozen_attention_mask.dtype, device=frozen_attention_mask.device)
        attention_mask = torch.concat([prefix_attention_mask, frozen_attention_mask], -1)
        
        model_frozen_inputs = {k.replace("_frozen", ""): v for k, v in inputs.items() if "_frozen" in k and k != "input_ids_frozen"}
        model_frozen_inputs["inputs_embeds"] = inputs_embeds
        model_frozen_inputs["attention_mask"] = attention_mask
        return model_frozen_inputs
                
    def get_prefix_named_parameters(self):
        return [(k, v) for k, v in self.named_parameters() if "model_prefix" in k]
    
    def get_prefix_parameters(self):
        return [v for k, v in self.get_prefix_named_parameters()]
    
    def get_frozen_named_parameters(self):
        return [(k, v) for k, v in self.named_parameters() if "model_prefix" not in k]
    
    def get_frozen_parameters(self):
        return [v for k, v in self.get_frozen_named_parameters()]
    
    def init_instruction(self, data_args, tokenizer, prefix_ids, prefix_embeds):
        assert sum([prefix_ids, prefix_embeds]) == 1
        
        task_dir = data_args.task_dir
        split_path = os.path.join(data_args.data_dir, "train_tasks.txt")
        task_instructions = []
        with open(split_path, encoding="utf-8") as split_f:
            for task_id, line in enumerate(split_f):
                task_name = line.strip()
                task_path = os.path.join(task_dir, task_name + ".json")
                with open(task_path, encoding="utf-8") as task_f:
                    s = task_f.read()
                    task_data = json.loads(s)

                    if isinstance(task_data["Definition"], list):
                        definition = "Definition: " + task_data["Definition"][0].strip()
                    else:
                        definition = "Definition: " + task_data["Definition"].strip()
                    if not definition[-1] in string.punctuation:
                        definition += "."
                    definition += "\n\n"

                    task_instructions.append(definition)

        prefix_inits = tokenizer(
            task_instructions,
            max_length=data_args.max_prefix_length,
            padding="max_length",
            return_tensors="pt",
            truncation=True,
            return_attention_mask=False,
        )
        model_prefix_inits = self.shared(prefix_inits["input_ids"].to(self.device))
        
        if prefix_ids:
            assert self.model_prefix.prefix_embeds.data.size() == model_prefix_inits.size()
            self.model_prefix.prefix_embeds.data = model_prefix_inits
            
            assert self.model_prefix.shared.weight.data.size() == self.shared.weight.data.size()
            self.model_prefix.shared.weight.data = self.shared.weight.data
        elif prefix_embeds:
            assert self.model_prefix.data.size() == model_prefix_inits.size()
            self.model_prefix.data = model_prefix_inits
        
    def init_exemplar(self, data_args, tokenizer, prefix_ids, prefix_embeds):
        assert sum([prefix_ids, prefix_embeds]) == 1
        
        task_dir = data_args.task_dir
        split_path = os.path.join(data_args.data_dir, "train_tasks.txt")
        pos_examples = []
        with open(split_path, encoding="utf-8") as split_f:
            for task_id, line in enumerate(split_f):
                task_name = line.strip()
                task_path = os.path.join(task_dir, task_name + ".json")
                with open(task_path, encoding="utf-8") as task_f:
                    s = task_f.read()
                    task_data = json.loads(s)
                    pos_examples_str = []
                    for idx, pos_example in enumerate(task_data["Positive Examples"][:data_args.num_pos_examples]):
                        pos_example_str = f" Positive Example {idx+1} -\n"
                        pos_example_str += f"Input: {pos_example['input'].strip()}"
                        if not pos_example_str[-1] in string.punctuation:
                            pos_example_str += "."
                        pos_example_str += "\n"
                        if isinstance(pos_example['output'], list):
                            pos_example_str += f" Output: {random.choice(pos_example['output']).strip()}"
                        else:
                            pos_example_str += f" Output: {pos_example['output'].strip()}"
                        if not pos_example_str[-1] in string.punctuation:
                            pos_example_str += "."
                        pos_example_str += "\n" 

                        if data_args.add_explanation and "explanation" in pos_example:
                            pos_example_str += f" Explanation: {pos_example['explanation'].strip()}"
                            if not pos_example_str[-1] in string.punctuation:
                                pos_example_str += "."
                            pos_example_str += "\n"
                        pos_example_str += "\n"
                        pos_examples_str.append(pos_example_str)

                pos_examples.append("".join(pos_examples_str))

        prefix_inits = tokenizer(
            pos_examples,
            max_length=data_args.max_prefix_length,
            padding="max_length",
            return_tensors="pt",
            truncation=True,
            return_attention_mask=False,
        )
        model_prefix_inits = self.shared(prefix_inits["input_ids"].to(self.device))

        if prefix_ids:
            assert self.model_prefix.prefix_embeds.data.size() == model_prefix_inits.size()
            self.model_prefix.prefix_embeds.data = model_prefix_inits
            
            assert self.model_prefix.shared.weight.data.size() == self.shared.weight.data.size()
            self.model_prefix.shared.weight.data = self.shared.weight.data
        elif prefix_embeds:
            assert self.model_prefix.data.size() == model_prefix_inits.size()
            self.model_prefix.data = model_prefix_inits
        
    def init_vocab(self, n_vocab_init, prefix_ids, prefix_embeds):
        if prefix_ids:
            n_task = self.model_prefix.prefix_embeds.size(0)
            max_prefix_length = self.model_prefix.prefix_embeds.size(1)
        elif prefix_embeds:
            n_task = self.model_prefix.size(0)
            max_prefix_length = self.model_prefix.size(1)
        indices = [np.random.choice(n_vocab_init, size=max_prefix_length, replace=False).tolist() for _ in range(n_task)]
        model_prefix_inits = self.shared.weight[indices]
        
        if prefix_ids:
            assert self.model_prefix.prefix_embeds.data.size() == model_prefix_inits.size()
            self.model_prefix.prefix_embeds.data = model_prefix_inits
            
            assert self.model_prefix.shared.weight.data.size() == self.shared.weight.data.size()
            self.model_prefix.shared.weight.data = self.shared.weight.data
        elif prefix_embeds:
            assert self.model_prefix.data.size() == model_prefix_inits.size()
            self.model_prefix.data = model_prefix_inits

    def init_category(self, n_vocab_init, task_category_dict, prefix_ids, prefix_embeds):
        if prefix_ids:
            n_task = self.model_prefix.prefix_embeds.size(0)
            max_prefix_length = self.model_prefix.prefix_embeds.size(1)
            d_emb = self.model_prefix.prefix_embeds.size(2)
        elif prefix_embeds:
            n_task = self.model_prefix.size(0)
            max_prefix_length = self.model_prefix.size(1)
            d_emb = self.model_prefix.size(2)

        assert n_task == len(task_category_dict)
        n_category = len(set(task_category_dict.values()))
        model_prefix_inits_category = torch.randn(n_category, max_prefix_length, d_emb)
        indices = [task_category_dict[task_id] for task_id in range(n_task)]
        model_prefix_inits = model_prefix_inits_category[indices]
        
        if prefix_ids:
            assert self.model_prefix.prefix_embeds.data.size() == model_prefix_inits.size()
            self.model_prefix.prefix_embeds.data = model_prefix_inits
            
            assert self.model_prefix.shared.weight.data.size() == self.shared.weight.data.size()
            self.model_prefix.shared.weight.data = self.shared.weight.data
        elif prefix_embeds:
            assert self.model_prefix.data.size() == model_prefix_inits.size()
            self.model_prefix.data = model_prefix_inits
            
    def init_category_vocab(self, n_vocab_init, task_category_dict, prefix_ids, prefix_embeds):
        if prefix_ids:
            n_task = self.model_prefix.prefix_embeds.size(0)
            max_prefix_length = self.model_prefix.prefix_embeds.size(1)
        elif prefix_embeds:
            n_task = self.model_prefix.size(0)
            max_prefix_length = self.model_prefix.size(1)

        assert n_task == len(task_category_dict)
        n_category = len(set(task_category_dict.values()))
        indices_category = [np.random.choice(n_vocab_init, size=max_prefix_length, replace=False).tolist() for _ in range(n_category)]
        indices = [indices_category[task_category_dict[task_id]] for task_id in range(n_task)]
        model_prefix_inits = self.shared.weight[indices]
        
        if prefix_ids:
            assert self.model_prefix.prefix_embeds.data.size() == model_prefix_inits.size()
            self.model_prefix.prefix_embeds.data = model_prefix_inits
            
            assert self.model_prefix.shared.weight.data.size() == self.shared.weight.data.size()
            self.model_prefix.shared.weight.data = self.shared.weight.data
        elif prefix_embeds:
            assert self.model_prefix.data.size() == model_prefix_inits.size()
            self.model_prefix.data = model_prefix_inits
