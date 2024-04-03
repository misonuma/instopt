import re
import pdb

import datasets
import promptsource.utils


def apply_both_template(dataset, template_prompt, template, num_proc=None):
    def map_prompt_fn(ex):
        ex = promptsource.utils.removeHyphen(ex)
        answer_choices = template_prompt.get_answer_choices_list(ex)
        
        try:
            inputs_and_targets = template_prompt.apply(ex)
            
            if len(inputs_and_targets) == 2:
                inputs, targets = inputs_and_targets
                ex = {"inputs_pretokenized": inputs, "targets_pretokenized": targets}
            else:
                ex = {"inputs_pretokenized": "", "targets_pretokenized": ""}

        except Exception as e:
            print(template_prompt.name, e)
            ex = {"inputs_pretokenized": "", "targets_pretokenized": ""}

        if answer_choices:
            ex["choices_pretokenized"] = answer_choices

        return ex
    
    def map_fn(ex):
        ex = promptsource.utils.removeHyphen(ex)
        
        try:
            prefix_inputs_and_targets = template.apply(ex)
            
            if len(prefix_inputs_and_targets) == 2:
                prefix_inputs, targets = prefix_inputs_and_targets
                ex = {"prefix_inputs_pretokenized": prefix_inputs}
            # When template results in an empty example, template.apply returns [""]
            # Also, if the template gets split wrong, len can be > 2
            # We will filter these out later
            else:
                ex = {"prefix_inputs_pretokenized": ""}

        except Exception as e:
            print(template.name, e)
            ex = {"prefix_inputs_pretokenized": ""}

        return ex
    
    def filter_fn(ex):
        return len(ex["inputs_pretokenized"]) > 0 and len(ex["prefix_inputs_pretokenized"]) > 0 and len(ex["targets_pretokenized"]) > 0

    original_columns = dataset.column_names
    dataset = dataset.map(map_fn, num_proc=num_proc)
    dataset = dataset.map(map_prompt_fn, num_proc=num_proc)
    dataset = dataset.filter(filter_fn, num_proc=num_proc)
    # map keeps original columns, remove them
    return dataset.remove_columns(set(original_columns) - {"inputs_pretokenized", "targets_pretokenized", "prefix_inputs_pretokenized", "choices_pretokenized"})


def get_dataset_splits(dataset_name, subset_name=None):
    info = datasets.get_dataset_infos(dataset_name)
    subset_name = subset_name or list(info.keys())[0]
    return info[subset_name].splits


def task_clean(text):
    # Clean the text according to allowed characters for a task name
    return re.sub(r"[^\w\d\._]+", "_", text)


def get_task_name(dataset_name, subset_name, template_name):
    return task_clean(dataset_name + (f"_{subset_name}_" if subset_name is not None else "_") + template_name)
