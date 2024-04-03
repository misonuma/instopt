# +
import logging
import random
import string
import pdb
from itertools import chain

import torch
import numpy as np

from transformers.data.data_collator import *
from arguments import DataTrainingArguments
# -

logger = logging.getLogger(__name__)


@dataclass
class DataCollatorForNI:
    def __init__(
        self,
        tokenizer: PreTrainedTokenizerBase,
        model: Optional[Any],
        data_args: DataTrainingArguments,
        pad_to_multiple_of: Optional[int] = None,
        label_pad_token_id: int = -100,
        return_tensors: str = "pt",
        text_only: bool=False,
    ):
        self.tokenizer = tokenizer
        self.model = model
        self.pad_to_multiple_of = pad_to_multiple_of
        self.label_pad_token_id = label_pad_token_id
        self.return_tensors = return_tensors
        self.text_only = text_only
        
        self.padding="max_length" if data_args.pad_to_max_length else "longest"
        self.max_source_length=data_args.max_source_length
        self.max_target_length=data_args.max_target_length
        self.add_task_name=data_args.add_task_name
        self.add_task_definition=data_args.add_task_definition
        self.num_pos_examples=data_args.num_pos_examples
        self.num_neg_examples=data_args.num_neg_examples
        self.add_explanation=data_args.add_explanation
        self.add_task_definition_train=data_args.add_task_definition_train
        self.num_pos_examples_train=data_args.num_pos_examples_train
        self.num_neg_examples_train=data_args.num_neg_examples_train
        self.add_explanation_train=data_args.add_explanation_train
        
        self.random_examples=data_args.random_examples
        self.random_instance_examples=data_args.random_instance_examples
        self.random_instance_exemplars=data_args.random_instance_exemplars
        self.example_index=data_args.example_index
        self.random_text=data_args.random_text
        
        self.instructtune = data_args.instructtune
        self.instructadd = data_args.instructadd
        self.max_exemplar_length = data_args.max_exemplar_length
        if data_args.max_frozen_source_length is not None:
            self.max_frozen_source_length = data_args.max_frozen_source_length
        else:
            if self.num_pos_examples > 0:
                self.max_frozen_source_length = self.max_source_length - (self.max_exemplar_length * self.num_pos_examples)
            else:
                self.max_frozen_source_length = self.max_source_length - self.max_exemplar_length 
            
    def __call__(self, batch, return_tensors=None):
        if return_tensors is None:
            return_tensors = self.return_tensors

        # for training
        task_sources = [] # inputs for frozen task predictor
        task_targets = [] # labels for frozen task predictor
        
        task_input_exemplars = [] # exemplars for frozen task predictor
        task_target_exemplars = [] # exemplars for frozen task predictor
        task_ids = []
        
        prefix_sources = []
        prefix_input_sources = [] # input sources for prefix generator
        prefix_target_sources = [] # target sources for prefix generator
                
        # for inference
        sources = []
        for instance in batch:
            add_task_name = self.add_task_name
            if instance["Subset"] == "train" and self.add_task_definition_train is not None:
                add_task_definition = self.add_task_definition_train
            else:
                add_task_definition = self.add_task_definition
            if instance["Subset"] == "train" and self.num_pos_examples_train is not None:
                num_pos_examples = self.num_pos_examples_train
            else:
                num_pos_examples = self.num_pos_examples
            if instance["Subset"] == "train" and self.num_neg_examples_train is not None:
                num_neg_examples = self.num_neg_examples_train
            else:
                num_neg_examples = self.num_neg_examples
            if instance["Subset"] == "train" and self.add_explanation_train is not None:
                add_explanation = self.add_explanation_train
            else:
                add_explanation = self.add_explanation 
            
            task_id = instance["TaskID"]
            task_ids.append(task_id)
            
            task_input = instance['Instance']['input'].strip()
            task_target = random.choice(instance["Instance"]["output"])
            
            # get task_instance
            task_instance = ""
            task_instance += "Now complete the following example -\n"
            task_instance += f"Input: {task_input}"
            if not task_instance[-1] in string.punctuation:
                task_instance += "."
            task_instance += "\n"
            task_instance += "Output: "
            
            # get task_instruction
            task_name = ""
            if add_task_name:
                task_name += instance["Task"] + ". "

            definition = ""
            if add_task_definition:
                if isinstance(instance["Definition"], list):
                    definition = "Definition: " + instance["Definition"][0].strip() # TODO: should we use <Definition>?
                else:
                    definition = "Definition: " + instance["Definition"].strip()
                if not definition[-1] in string.punctuation:
                    definition += "."
                definition += "\n\n"
            
            # try to add positive examples.
            pos_examples = []
            assert sum([self.random_examples, self.random_instance_examples, (self.example_index is not None)]) <= 1
            if (instance["Subset"] == "train" or instance["Subset"] == "meta") and self.random_examples:
                pos_examples_str = instance["Exemplar"] # random example for each task
            elif (instance["Subset"] == "train" or instance["Subset"] == "meta") and self.random_instance_examples:
                pos_examples_str = instance["Instance Exemplar"] # random example for each instance
            elif (instance["Subset"] == "train" or instance["Subset"] == "meta") and self.example_index is not None:
                # fixed example specified by example_index 
                if isinstance(self.example_index, int):
                    pos_examples_str = instance["Exemplar"][self.example_index*num_pos_examples: (self.example_index+1)*num_pos_examples] 
                elif isinstance(self.example_index, torch.Tensor):
                    example_indices = self.example_index[task_id]
                    assert len(example_indices) == num_pos_examples
                    pos_examples_str = [instance["Exemplar"][example_index] for example_index in example_indices]
            else:
                pos_examples_str = instance["Positive Examples"] # fixed example for each instance/task
            
            for idx, pos_example in enumerate(pos_examples_str[:num_pos_examples]):
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
                
                if add_explanation and "explanation" in pos_example:
                    pos_example_str += f" Explanation: {pos_example['explanation'].strip()}"
                    if not pos_example_str[-1] in string.punctuation:
                        pos_example_str += "."
                    pos_example_str += "\n"
                pos_example_str += "\n"
                if len(self.tokenizer(task_name + definition + " ".join(pos_examples) + pos_example_str + task_instance)["input_ids"]) <= self.max_source_length:
                    pos_examples.append(pos_example_str)
                else:
                    break
            
            # try to add negative examples.
            neg_examples = []
            for idx, neg_example in enumerate(instance["Negative Examples"][:num_neg_examples]):
                neg_example_str = f" Negative Example {idx+1} -\n"
                neg_example_str += f"Input: {neg_example['input'].strip()}"
                if not neg_example_str[-1] in string.punctuation:
                    neg_example_str += "."
                neg_example_str += "\n"
                neg_example_str += f" Output: {neg_example['output'].strip()}"
                if not neg_example_str[-1] in string.punctuation:
                    neg_example_str += "."
                neg_example_str += "\n"
                if add_explanation and "explanation" in neg_example:
                    neg_example_str += f" Explanation: {neg_example['explanation'].strip()}"
                    if not neg_example_str[-1] in string.punctuation:
                        neg_example_str += "."
                    neg_example_str += "\n"
                neg_example_str += "\n"
                if len(self.tokenizer(task_name + definition + " ".join(pos_examples) + " ".join(neg_examples) + neg_example_str + task_instance)["input_ids"]) <= self.max_source_length:
                    neg_examples.append(neg_example_str)
                else:
                    break
                    
            task_instruction = task_name + definition + "".join(pos_examples) + "".join(neg_examples)
            
            prefix_sources.append(task_instruction)
            prefix_input_source = f"Input: {task_input}"
            prefix_input_sources.append(prefix_input_source)
            prefix_target_source = f"Output: {task_target}"
            prefix_target_sources.append(prefix_target_source)
        
            # sources & targets for model_frozen
            if self.instructadd:
                task_source = task_instruction + task_instance
            else:
                task_source = task_instance
            task_sources.append(task_source)
                        
            task_targets.append(task_target)
            
            if "Exemplar" in instance or "Instance Exemplar" in instance:
                if self.random_instance_exemplars:
                    exemplars = instance["Instance Exemplar"]
                else:
                    exemplars = instance["Exemplar"]
                    
                input_exemplars = []
                target_exemplars = []
                for exemplar in exemplars:
                    exemplar_str = f" Positive Example -\n"
                    exemplar_str += f"Input: {exemplar['input'].strip()}"
                    if not exemplar_str[-1] in string.punctuation:
                        exemplar_str += "."
                    exemplar_str += "\n"
                    input_exemplars.append(exemplar_str)
                    
                    exemplar_str = f" Output: {random.choice(exemplar['output']).strip()}"
                    if not exemplar_str[-1] in string.punctuation:
                        exemplar_str += "."
                    exemplar_str += "\n" 
                    exemplar_str += "\n"
                    target_exemplars.append(exemplar_str)

                task_input_exemplars.append(input_exemplars)
                task_target_exemplars.append(target_exemplars)

            # sources for inference
            source = task_instruction + task_instance
            sources.append(source)
        
        # for inference
        if self.text_only:
            model_inputs = {"inputs": sources}
        else:
            model_inputs = self.tokenizer(
                sources,
                max_length=self.max_source_length,
                padding=self.padding,
                return_tensors=self.return_tensors, 
                truncation=True,
                pad_to_multiple_of=self.pad_to_multiple_of)
            
        # for prefix generator
        if self.text_only:
            model_inputs = {"inputs_prefix": prefix_sources}
        else:
            if self.instructtune:
                prefix_inputs = self.tokenizer(
                    text=prefix_sources,
                    max_length=self.max_exemplar_length,
                    padding=self.padding,
                    return_tensors=self.return_tensors, 
                    truncation=True,
                    pad_to_multiple_of=self.pad_to_multiple_of,
                    add_special_tokens=False,
                )
            else:
                prefix_inputs = self.tokenizer(
                    text=prefix_input_sources,
                    text_pair=prefix_target_sources,
                    max_length=self.max_exemplar_length,
                    padding=self.padding,
                    return_tensors=self.return_tensors, 
                    truncation=True,
                    pad_to_multiple_of=self.pad_to_multiple_of,
                    add_special_tokens=False,
                )
            model_inputs["input_ids_prefix"] = prefix_inputs["input_ids"]
            model_inputs["attention_mask_prefix"] = prefix_inputs["attention_mask"]
                   
        # for task predictor
        if self.text_only:
            model_inputs["inputs_frozen"] = task_sources
        else:
            task_inputs = self.tokenizer(
                task_sources,
                max_length=self.max_frozen_source_length, # as generated prefix_labels are concatenated to task_inputs
                padding=self.padding,
                return_tensors=self.return_tensors,
                truncation=True,
                pad_to_multiple_of=self.pad_to_multiple_of
            )
            model_inputs["input_ids_frozen"] = task_inputs["input_ids"]
            model_inputs["attention_mask_frozen"] = task_inputs["attention_mask"]
            
        if "output" in batch[0]["Instance"] and batch[0]["Instance"]["output"]:
            if self.text_only:
                model_inputs["labels_frozen"] = task_targets
            else:
                with self.tokenizer.as_target_tokenizer():
                    task_labels = self.tokenizer(
                        task_targets,
                        max_length=self.max_target_length,
                        padding=self.padding,
                        return_tensors=self.return_tensors,
                        truncation=True,
                        pad_to_multiple_of=self.pad_to_multiple_of
                    )
                label_mask = task_labels["attention_mask"].bool()
                model_inputs["labels_frozen"] = task_labels["input_ids"].masked_fill(~label_mask, self.label_pad_token_id)
                
            if self.model is not None and hasattr(self.model, "prepare_decoder_input_ids_from_labels") and not self.text_only:
                decoder_input_ids_frozen = self.model.prepare_decoder_input_ids_from_labels(labels=model_inputs["labels_frozen"])
                model_inputs["decoder_input_ids_frozen"] = decoder_input_ids_frozen
        else:
            model_inputs["labels_frozen"] = None
            
        if "Exemplar" in batch[0]:
            task_input_exemplars_flat = list(chain(*task_input_exemplars)) # 2d array (batch x n_exemplar) -> 1d array for tokenizer
            task_target_exemplars_flat = list(chain(*task_target_exemplars))
            
            if self.text_only:
                model_inputs["exemplars"] = task_exemplars_flat
            else:
                exemplar_inputs = self.tokenizer(
                    text=task_input_exemplars_flat,
                    text_pair=task_target_exemplars_flat, 
                    max_length=self.max_exemplar_length,
                    padding=self.padding,
                    return_tensors=self.return_tensors,
                    truncation=True,  # only_first sometimes yields error
                    pad_to_multiple_of=self.pad_to_multiple_of,
                    add_special_tokens=False,
                )
                model_inputs["input_ids_exemplar"] = exemplar_inputs["input_ids"]
                model_inputs["attention_mask_exemplar"] = exemplar_inputs["attention_mask"]
                
                if self.random_text:
                    input_ids_exemplar = model_inputs["input_ids_exemplar"].view(model_inputs["input_ids"].size(0), -1, model_inputs["input_ids_exemplar"].size(-1))
                    input_ids_random = torch.concat([torch.randint_like(input_ids_exemplar[:, :-1, :], low=3, high=self.tokenizer.vocab_size), input_ids_exemplar[:, -1, :].unsqueeze(1)], 1)
                    assert input_ids_random.size() == input_ids_exemplar.size()

                    attention_mask_exemplar = model_inputs["attention_mask_exemplar"].view(model_inputs["attention_mask"].size(0), -1, model_inputs["attention_mask_exemplar"].size(-1))
                    attention_mask_random = torch.concat([torch.ones_like(attention_mask_exemplar[:, :-1, :]), attention_mask_exemplar[:, -1, :].unsqueeze(1)], 1)
                    assert attention_mask_random.size() == attention_mask_exemplar.size()

                    model_inputs["input_ids_exemplar"] = input_ids_random.view(model_inputs["input_ids_exemplar"].size(0), model_inputs["input_ids_exemplar"].size(1))
                    model_inputs["attention_mask_exemplar"] = attention_mask_random.view(model_inputs["attention_mask_exemplar"].size(0), model_inputs["attention_mask_exemplar"].size(1))
        
        model_inputs["task_ids"] = torch.tensor(task_ids)
        return model_inputs


