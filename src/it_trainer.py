from functools import wraps

import pickle
import sys
import random
import torch
import warnings
import datasets
from datasets import Dataset
from datasets.arrow_writer import SchemaInferenceError
from datasets.builder import DatasetGenerationError
from transformers import Trainer
from trl import SFTTrainer
from torch.utils.data import IterableDataset


def pack_tensor_2D(raw_lst, default, dtype, length=None):
    batch_size = len(raw_lst)
    length = length if length is not None else max(len(raw) for raw in raw_lst)
    tensor = default * torch.ones((batch_size, length), dtype=dtype)
    for i, raw in enumerate(raw_lst):
        tensor[i, :len(raw)] = torch.tensor(raw, dtype=dtype)
    return tensor

def get_collator_fn(tokenizer):
    def collator(batch):
        input_ids_lst = [x["input_ids"] for x in batch]
        attention_mask_lst = [x["attention_mask"] for x in batch]
        labels_lst = [x["labels"] for x in batch]

        data = {
            "input_ids": pack_tensor_2D(input_ids_lst, default=tokenizer.eos_token_id, dtype=torch.long),
            "attention_mask": pack_tensor_2D(attention_mask_lst, default=0, dtype=torch.long),
            "labels": pack_tensor_2D(labels_lst, default=-100, dtype=torch.long),
        }
        # data["labels"] = torch.where(data["attention_mask"] == 1, data["input_ids"], torch.tensor(-100))

        return data
    return collator

class ITTrainer(SFTTrainer):
    r"""
    Class definition of the Supervised Finetuning Trainer (SFT Trainer).
    This class is a wrapper around the `transformers.Trainer` class and inherits all of its attributes and methods.
    The trainer takes care of properly initializing the PeftModel in case a user passes a `PeftConfig` object.

    Args:
        model (Union[`transformers.PreTrainedModel`, `nn.Module`, `str`]):
            The model to train, can be a `PreTrainedModel`, a `torch.nn.Module` or a string with the model name to
            load from cache or download. The model can be also converted to a `PeftModel` if a `PeftConfig` object is
            passed to the `peft_config` argument.
        args (Optional[`SFTConfig`]):
            The arguments to tweak for training. Will default to a basic instance of [`SFTConfig`] with the `output_dir`
            set to a directory named *tmp_trainer* in the current directory if not provided.
        data_collator (Optional[`transformers.DataCollator`]):
            The data collator to use for training.
        train_dataset (Optional[`datasets.Dataset`]):
            The dataset to use for training. We recommend users to use `trl.trainer.ConstantLengthDataset` to create their dataset.
        eval_dataset (Optional[Union[`datasets.Dataset`, Dict[`str`, `datasets.Dataset`]]]):
            The dataset to use for evaluation. We recommend users to use `trl.trainer.ConstantLengthDataset` to create their dataset.
        tokenizer (Optional[`transformers.PreTrainedTokenizer`]):
            The tokenizer to use for training. If not specified, the tokenizer associated to the model will be used.
        model_init (`Callable[[], transformers.PreTrainedModel]`):
            The model initializer to use for training. If None is specified, the default model initializer will be used.
        compute_metrics (`Callable[[transformers.EvalPrediction], Dict]`, *optional* defaults to None):
            The function used to compute metrics during evaluation. It should return a dictionary mapping metric names to metric values.
            If not specified, only the loss will be computed during evaluation.
        callbacks (`List[transformers.TrainerCallback]`):
            The callbacks to use for training.
        optimizers (`Tuple[torch.optim.Optimizer, torch.optim.lr_scheduler.LambdaLR]`):
            The optimizer and scheduler to use for training.
        preprocess_logits_for_metrics (`Callable[[torch.Tensor, torch.Tensor], torch.Tensor]`):
            The function to use to preprocess the logits before computing the metrics.
        peft_config (`Optional[PeftConfig]`):
            The PeftConfig object to use to initialize the PeftModel.
        formatting_func (`Optional[Callable]`):
            The formatting function to be used for creating the `ConstantLengthDataset`.
    """
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.data_collator = get_collator_fn(self.tokenizer)

    def _prepare_dataset(
        self,
        dataset,
        tokenizer,
        packing,
        dataset_text_field,
        max_seq_length,
        formatting_func,
        num_of_sequences,
        chars_per_token,
        remove_unused_columns=True,
        append_concat_token=True,
        add_special_tokens=True,
        skip_prepare_dataset=False,
    ):
        if dataset is None:
            raise ValueError("The dataset should not be None")

        return dataset
