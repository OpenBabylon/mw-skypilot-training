# Copyright 2024 Google LLC
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

"""Continual Pre-Training script for Causal Language Models with Dataset Mixing."""

from collections.abc import Callable, Mapping, Sequence
import datetime
import json
import os
from typing import Any
import warnings

from absl import app
from absl import flags
from absl import logging
from accelerate import DistributedType
from accelerate import PartialState
import torch
import transformers
import trl
# import wandb
from util import dataset_validation_util
from vertex_vision_model_garden_peft.train.vmg import callbacks
from vertex_vision_model_garden_peft.train.vmg import eval_lib
from vertex_vision_model_garden_peft.train.vmg import utils
from util import constants
from util import fileutils
import datasets

# --- CORE CPT PARAMETERS ---
_PRETRAINED_MODEL_NAME_OR_PATH = flags.DEFINE_string(
    'pretrained_model_name_or_path', None, 'The base model to continue pre-training.', required=True
)
_TRAIN_DATASET = flags.DEFINE_string(
    'train_dataset', None, 'The path to the primary new text dataset for CPT.', required=True
)
_OUTPUT_DIR = flags.DEFINE_string('output_dir', None, 'The output directory.', required=True)
_MAX_SEQ_LENGTH = flags.DEFINE_integer('max_seq_length', 2048, 'The maximum sequence length for packing.')
_TRAIN_COLUMN = flags.DEFINE_string('train_column', 'text', 'The name of the raw text column in the dataset.')

# --- MODIFICATION: DATASET MIXING PARAMETERS ---
_REPLAY_DATASET = flags.DEFINE_string(
    'replay_dataset', None, 'Optional: Path to a general-purpose dataset for replay to mitigate forgetting.'
)
_MIXING_PROBABILITIES = flags.DEFINE_list(
    'mixing_probabilities', None, 'Comma-separated list of probabilities for dataset mixing (e.g., "0.95,0.05").'
)
# --- END MODIFICATION ---

# --- CPT HYPERPARAMETERS ---
_LEARNING_RATE = flags.DEFINE_float(
    'learning_rate', 2e-5, 'Peak learning rate for CPT. Should be much lower than initial pre-training LR.'
)
_WARMUP_RATIO = flags.DEFINE_float(
    'warmup_ratio', 0.01, 'Ratio of total training steps for LR warmup (e.g., 0.01 for 1%).'
)
_MAX_STEPS = flags.DEFINE_integer('max_steps', 100000, 'Total number of training steps for the CPT run.')
_PER_DEVICE_TRAIN_BATCH_SIZE = flags.DEFINE_integer('per_device_train_batch_size', 1, 'Per device train batch size.')
_GRADIENT_ACCUMULATION_STEPS = flags.DEFINE_integer('gradient_accumulation_steps', 16, 'Gradient accumulation steps.')
_WEIGHT_DECAY = flags.DEFINE_float('weight_decay', 0.1, 'Weight decay.')
_LR_SCHEDULER_TYPE = flags.DEFINE_string('lr_scheduler_type', 'cosine', 'Learning rate scheduler type.')

_ATTN_IMPLEMENTATION = flags.DEFINE_string(
    'attn_implementation',
    None,
    'Attention implementation, can be `eager`, `sdpa` or `flash_attention_2`',
)

# --- FULL-PARAMETER TRAINING & FSDP CONFIGURATION ---
_ENABLE_PEFT = flags.DEFINE_boolean('enable_peft', False, 'Set to False for full-parameter CPT.')

FSDP_POLICY = "full_shard auto_wrap"
FSDP_LAYER = "Gemma3DecoderLayer"

fsdp_cfg = {
    "activation_checkpointing": True,
    "activation_checkpointing_policy": {"offload_to_cpu": True},
    "mixed_precision": "bf16"
}
_PRECISION_MODE = flags.DEFINE_enum(
    'precision_mode',
    constants.PRECISION_MODE_16,
    [
        constants.PRECISION_MODE_4,
        constants.PRECISION_MODE_8,
        constants.PRECISION_MODE_16,
        constants.PRECISION_MODE_16B,
        constants.PRECISION_MODE_32,
    ],
    'Precision to load model weights for finetuning.',
)

_TRAIN_PRECISION = flags.DEFINE_enum(
    'train_precision',
    constants.PRECISION_MODE_16B,
    [
        constants.PRECISION_MODE_16,
        constants.PRECISION_MODE_16B,
        constants.PRECISION_MODE_32,
    ],
    'Precision to train the model.',
)
_OPTIMIZER = flags.DEFINE_string('optimizer', 'adamw_torch', 'The optimizer.')
_SAVE_STEPS = flags.DEFINE_integer('save_steps', 1000, 'Checkpoint saving frequency.')
_LOGGING_STEPS = flags.DEFINE_integer('logging_steps', 10, 'Logging frequency.')
_HUGGINGFACE_ACCESS_TOKEN = flags.DEFINE_string('huggingface_access_token', None, 'Hugging Face access token.')


def continual_pretrain(
    pretrained_model_name_or_path: str,
    train_dataset: str,
    output_dir: str,
    max_seq_length: int,
    train_column: str,
    replay_dataset: str | None,
    mixing_probabilities: list[str] | None,
    learning_rate: float,
    warmup_ratio: float,
    max_steps: int,
    per_device_train_batch_size: int,
    gradient_accumulation_steps: int,
    weight_decay: float,
    lr_scheduler_type: str,
    attn_implementation: str | None,
    enable_peft: bool,
    precision_mode: str,
    train_precision: str,
    optimizer: str,
    save_steps: int,
    logging_steps: int,
    access_token: str | None = None,
) -> None:
    """Performs Continual Pre-Training on a Causal Language Model."""

    tokenizer = dataset_validation_util.load_tokenizer(
        pretrained_model_name_or_path,
        'right',
        access_token=access_token,
    )

    # --- MODIFICATION: CPT DATA PIPELINE WITH OPTIONAL MIXING ---
    logging.info(f"Loading primary CPT dataset from: {train_dataset}")
    primary_dataset = datasets.load_dataset(
        train_dataset,
        split='train',
        streaming=True
    )

    if replay_dataset:
        if not mixing_probabilities:
            raise ValueError("`--mixing_probabilities` must be provided when using `--replay_dataset`.")
        
        logging.info(f"Loading replay dataset from: {replay_dataset}")
        replay_dataset_stream = datasets.load_dataset(
            replay_dataset,
            'en',  # <-- Add this line to specify the English config of C4
            split='train',
            streaming=True
        )
        probabilities = [float(p) for p in mixing_probabilities]
        if len(probabilities)!= 2:
            raise ValueError("`--mixing_probabilities` must contain exactly two values.")

        logging.info(f"Interleaving datasets with probabilities: {probabilities}")
        
        # Use a seed for reproducibility of the mixing
        interleaved_dataset = datasets.interleave_datasets(
            [primary_dataset, replay_dataset_stream],
            probabilities=probabilities,
            seed=42,
            stopping_strategy="all_exhausted" # Ensures we train for max_steps
        )
    else:
        logging.info("No replay dataset provided. Using primary dataset only.")
        interleaved_dataset = primary_dataset

    def reformat_dataset(example):
        return {"text": example[train_column]}

    packed_train_dataset = interleaved_dataset.map(reformat_dataset)
    logging.info("Training dataset prepared and mapped for CPT.")
    # --- END MODIFICATION ---

    model = utils.load_model(
        pretrained_model_name_or_path=pretrained_model_name_or_path,
        tokenizer=tokenizer,
        precision_mode=precision_mode,
        access_token=access_token,
        train_precision=train_precision,
        attn_implementation=attn_implementation,
    )

    if enable_peft:
        raise NotImplementedError("PEFT-based CPT is not the recommended path. Set --enable_peft=False.")

    accelerator_config = {'use_configured_state': True}

    training_arguments = transformers.TrainingArguments(
        output_dir=output_dir,
        per_device_train_batch_size=per_device_train_batch_size,
        gradient_accumulation_steps=gradient_accumulation_steps,
        learning_rate=learning_rate,
        max_steps=max_steps,
        warmup_ratio=warmup_ratio,
        lr_scheduler_type=lr_scheduler_type,
        optim=optimizer,
        weight_decay=weight_decay,
        save_steps=save_steps,
        logging_steps=logging_steps,
        save_total_limit=3,
        bf16=(train_precision == constants.PRECISION_MODE_16B),
        fp16=(train_precision == constants.PRECISION_MODE_16),
        fsdp=FSDP_POLICY,
        fsdp_config=fsdp_cfg,
        fsdp_transformer_layer_cls_to_wrap=FSDP_LAYER,
        gradient_checkpointing_kwargs={'use_reentrant': False},
        accelerator_config=accelerator_config,
    )

    trainer = trl.SFTTrainer(
        model=model,
        args=training_arguments,
        train_dataset=packed_train_dataset,
        dataset_text_field="text",
        max_seq_length=max_seq_length,
        tokenizer=tokenizer,
        packing=True,
    )

    if trainer.is_fsdp_enabled:
        logging.info('Trainer running with FSDP for full-parameter continual pre-training.')
    else:
        logging.warning('Trainer running without FSDP. This is not recommended for full-parameter CPT.')

    trainer.train()

    final_checkpoint = os.path.join(output_dir, "final-checkpoint")
    trainer.save_model(final_checkpoint)
    tokenizer.save_pretrained(final_checkpoint)
    logging.info(f"Continual pre-training complete. Final model saved to: {final_checkpoint}")


def main(unused_argv: Sequence[str]) -> None:
    continual_pretrain(
        pretrained_model_name_or_path=_PRETRAINED_MODEL_NAME_OR_PATH.value,
        train_dataset=_TRAIN_DATASET.value,
        output_dir=_OUTPUT_DIR.value,
        max_seq_length=_MAX_SEQ_LENGTH.value,
        train_column=_TRAIN_COLUMN.value,
        replay_dataset=_REPLAY_DATASET.value,
        mixing_probabilities=_MIXING_PROBABILITIES.value,
        learning_rate=_LEARNING_RATE.value,
        warmup_ratio=_WARMUP_RATIO.value,
        max_steps=_MAX_STEPS.value,
        per_device_train_batch_size=_PER_DEVICE_TRAIN_BATCH_SIZE.value,
        gradient_accumulation_steps=_GRADIENT_ACCUMULATION_STEPS.value,
        attn_implementation=_ATTN_IMPLEMENTATION.value,
        weight_decay=_WEIGHT_DECAY.value,
        lr_scheduler_type=_LR_SCHEDULER_TYPE.value,
        enable_peft=_ENABLE_PEFT.value,
        precision_mode=_PRECISION_MODE.value,
        train_precision=_TRAIN_PRECISION.value,
        optimizer=_OPTIMIZER.value,
        save_steps=_SAVE_STEPS.value,
        logging_steps=_LOGGING_STEPS.value,
        access_token=_HUGGINGFACE_ACCESS_TOKEN.value,
    )
    utils.force_gc()


if __name__ == '__main__':
    app.run(main)