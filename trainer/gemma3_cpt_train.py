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

from collections.abc import Sequence
import os
from transformers.trainer_utils import get_last_checkpoint
from pathlib import Path
from absl import app
from absl import flags
from absl import logging
import transformers
import trl

# import wandb
from util import dataset_validation_util
from vertex_vision_model_garden_peft.train.vmg import utils
from util import constants
import datasets
from huggingface_hub import HfApi
from huggingface_hub import Repository
from transformers.trainer_callback import TrainerCallback
from accelerate import PartialState
import time
from concurrent.futures import ThreadPoolExecutor

# one single background thread for rank-0
_bg_executor: ThreadPoolExecutor | None = None
_bg_futures = []  # we’ll gather them at the very end

if not PartialState().is_main_process:
    os.environ["WANDB_MODE"] = "disabled"
    os.environ["WANDB_SILENT"] = "true"

# --- CORE CPT PARAMETERS ---
_PRETRAINED_MODEL_NAME_OR_PATH = flags.DEFINE_string(
    "pretrained_model_name_or_path",
    None,
    "The base model to continue pre-training.",
    required=True,
)
_TRAIN_DATASET = flags.DEFINE_string(
    "train_dataset",
    None,
    "The path to the primary new text dataset for CPT.",
    required=True,
)
_OUTPUT_DIR = flags.DEFINE_string(
    "output_dir", None, "The output directory.", required=True
)
_MAX_SEQ_LENGTH = flags.DEFINE_integer(
    "max_seq_length", 2048, "The maximum sequence length for packing."
)
_TRAIN_COLUMN = flags.DEFINE_string(
    "train_column", "text", "The name of the raw text column in the dataset."
)

# --- MODIFICATION: DATASET MIXING PARAMETERS ---
_REPLAY_DATASET = flags.DEFINE_string(
    "replay_dataset",
    None,
    "Optional: Path to a general-purpose dataset for replay to mitigate forgetting.",
)
_MIXING_PROBABILITIES = flags.DEFINE_list(
    "mixing_probabilities",
    None,
    'Comma-separated list of probabilities for dataset mixing (e.g., "0.95,0.05").',
)
# --- END MODIFICATION ---

# --- CPT HYPERPARAMETERS ---
_LEARNING_RATE = flags.DEFINE_float(
    "learning_rate",
    2e-5,
    "Peak learning rate for CPT. Should be much lower than initial pre-training LR.",
)
_WARMUP_RATIO = flags.DEFINE_float(
    "warmup_ratio",
    0.01,
    "Ratio of total training steps for LR warmup (e.g., 0.01 for 1%).",
)
_MAX_STEPS = flags.DEFINE_integer(
    "max_steps", 100000, "Total number of training steps for the CPT run."
)
_PER_DEVICE_TRAIN_BATCH_SIZE = flags.DEFINE_integer(
    "per_device_train_batch_size", 1, "Per device train batch size."
)
_GRADIENT_ACCUMULATION_STEPS = flags.DEFINE_integer(
    "gradient_accumulation_steps", 16, "Gradient accumulation steps."
)
_WEIGHT_DECAY = flags.DEFINE_float(
    "weight_decay",
    0.001,
    "The weight decay in the learning rate scheduler.",
)
_LR_SCHEDULER_TYPE = flags.DEFINE_string(
    "lr_scheduler_type", "cosine", "Learning rate scheduler type."
)
_HF_REPO = flags.DEFINE_string(
    "repo_id", "polyagent/gemma3-cpt-test", "HF repo to push models into."
)

_ATTN_IMPLEMENTATION = flags.DEFINE_string(
    "attn_implementation",
    None,
    "Attention implementation, can be `eager`, `sdpa` or `flash_attention_2`",
)

# --- FULL-PARAMETER TRAINING & FSDP CONFIGURATION ---
_ENABLE_PEFT = flags.DEFINE_boolean(
    "enable_peft", False, "Set to False for full-parameter CPT."
)
_THRPUT_MEASURE = flags.DEFINE_boolean(
    "throughput_measurement", False, "Set to False for full-parameter CPT."
)
_STREAMING = flags.DEFINE_boolean(
    "streaming", False, "Set to False for full-parameter CPT."
)

FSDP_POLICY = "hybrid_shard auto_wrap"
FSDP_LAYER = "Gemma3DecoderLayer"

_PRECISION_MODE = flags.DEFINE_enum(
    "precision_mode",
    constants.PRECISION_MODE_16,
    [
        constants.PRECISION_MODE_4,
        constants.PRECISION_MODE_8,
        constants.PRECISION_MODE_16,
        constants.PRECISION_MODE_16B,
        constants.PRECISION_MODE_32,
    ],
    "Precision to load model weights for finetuning.",
)

_TRAIN_PRECISION = flags.DEFINE_enum(
    "train_precision",
    constants.PRECISION_MODE_16B,
    [
        constants.PRECISION_MODE_16,
        constants.PRECISION_MODE_16B,
        constants.PRECISION_MODE_32,
    ],
    "Precision to train the model.",
)
_OPTIMIZER = flags.DEFINE_string("optimizer", "adamw_torch", "The optimizer.")
_SAVE_STEPS = flags.DEFINE_integer("save_steps", 1000, "Checkpoint saving frequency.")
_LOGGING_STEPS = flags.DEFINE_integer("logging_steps", 10, "Logging frequency.")
_HUGGINGFACE_ACCESS_TOKEN = flags.DEFINE_string(
    "huggingface_access_token", None, "Hugging Face access token."
)


class TokensPerSecCallback(TrainerCallback):
    """
    Compute & print token throughput every optimizer step.

    Works with:
      * accelerate 0.18.x  (state.num_processes)
      * accelerate 0.25+   (state.world_size)
    """

    def __init__(self, seq_len: int, grad_acc_steps: int, batch_per_gpu: int):
        self.seq_len = seq_len
        self.grad_acc_steps = grad_acc_steps
        self.batch_per_gpu = batch_per_gpu
        self._t0, self._step0 = None, None  # set on first call

    # safe accessor for world size across versions
    @staticmethod
    def _world():
        state = PartialState()
        return (
            getattr(state, "world_size", None)
            or getattr(state, "num_processes", 1)
            or 1
        )

    def on_step_end(self, args, state, control, **kwargs):
        now, step = time.time(), state.global_step
        if self._t0 is None:  # first call: anchor timers
            self._t0, self._step0 = now, step
            return

        dt, dstep = now - self._t0, step - self._step0
        if dt <= 0 or dstep == 0:
            return

        seqs = dstep * self.batch_per_gpu * self.grad_acc_steps * self._world()
        tok_s = seqs * self.seq_len / dt

        if args.local_rank in (-1, 0):
            ws = self._world()
            print(f"[TOK_S] {tok_s:,.0f} tok/s  ({tok_s / ws:,.0f} tok/s/GPU)")

        self._t0, self._step0 = now, step


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
    throughput_measure: bool,
    precision_mode: str,
    train_precision: str,
    optimizer: str,
    save_steps: int,
    logging_steps: int,
    streaming: bool,
    repo_id: str,
    access_token: str | None = None,
) -> None:
    """Performs Continual Pre-Training on a Causal Language Model."""

    tokenizer = dataset_validation_util.load_tokenizer(
        pretrained_model_name_or_path,
        "right",
        access_token=access_token,
    )

    # --- MODIFICATION: CPT DATA PIPELINE WITH OPTIONAL MIXING ---
    logging.info(f"Loading primary CPT dataset from: {train_dataset}")

    primary_dataset = datasets.load_dataset(
        train_dataset, split="train", streaming=streaming
    )

    if replay_dataset:
        if not mixing_probabilities:
            raise ValueError(
                "`--mixing_probabilities` must be provided when using `--replay_dataset`."
            )

        logging.info(f"Loading replay dataset from: {replay_dataset}")
        replay_dataset_stream = datasets.load_dataset(
            replay_dataset, "sample-10BT", split="train", streaming=streaming
        )
        probabilities = [float(p) for p in mixing_probabilities]
        if len(probabilities) != 2:
            raise ValueError(
                "`--mixing_probabilities` must contain exactly two values."
            )

        logging.info(f"Interleaving datasets with probabilities: {probabilities}")

        # Use a seed for reproducibility of the mixing
        interleaved_dataset = datasets.interleave_datasets(
            [primary_dataset, replay_dataset_stream],
            probabilities=probabilities,
            seed=42,
            stopping_strategy="all_exhausted",  # Ensures we train for max_steps
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
        raise NotImplementedError(
            "PEFT-based CPT is not the recommended path. Set --enable_peft=False."
        )

    accelerator_config = {"use_configured_state": True}

    if throughput_measure:
        my_callbacks = [
            TokensPerSecCallback(
                seq_len=max_seq_length,
                grad_acc_steps=gradient_accumulation_steps,
                batch_per_gpu=per_device_train_batch_size,
            )
        ]
    else:
        my_callbacks = []

    state = PartialState()  # works inside or outside FSDP
    is_main = getattr(state, "is_main_process", True)

    # ── pick where to publish ────────────────────────────────────────────────────
    repo = None
    hf_token = os.getenv("HF_TOKEN")
    if is_main:
        HfApi(token=hf_token).create_repo(
            repo_id="PolyAgent/gemma3-cpt-test",  # full slug
            private=True,  # or False
            exist_ok=True,  # no error if it’s already there
        )  # ← ① one process only
        repo = Repository(
            local_dir=output_dir,  # Trainer writes here
            clone_from=repo_id.strip(),  # "PolyAgent/gemma3-cpt-test"
            token=hf_token,  # write-scoped token
        )
        report_to = ["wandb"]
    else:
        report_to = []
        os.environ["WANDB_MODE"] = "disabled"  # wandb.init() becomes a no-op
        os.environ["WANDB_SILENT"] = "true"  # hide the warning banner

    # my_callbacks.append(
    # AsyncPushCallback(repo=repo, push_freq=save_steps)
    # )

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
        fsdp_transformer_layer_cls_to_wrap=FSDP_LAYER,
        gradient_checkpointing_kwargs={"use_reentrant": False},
        accelerator_config=accelerator_config,
        dataloader_num_workers=8,
        dataloader_prefetch_factor=4,
        # dataloader_pin_memory=True,
        report_to=report_to,
        log_level="info",  # ensures the metrics show up
        push_to_hub=True,  # ✅ push on final save
        hub_model_id=repo_id.strip(),  # ✅ destination repo
        hub_token=hf_token,  # ✅ write token
    )

    last_ckpt = None
    if training_arguments.resume_from_checkpoint is None:
        # if output_dir already has a checkpoint-XXXX sub-dir, pick the newest
        try:
            last_ckpt = get_last_checkpoint(Path(output_dir))
            if last_ckpt:
                logging.info("🡒 Resuming from previous run at %s", last_ckpt)
                training_arguments.resume_from_checkpoint = last_ckpt
        except Exception as err:
            logging.warning("Could not scan for previous checkpoint: %s", err)
    # (else: user already passed --resume_from_checkpoint)

    trainer = trl.SFTTrainer(
        model=model,
        args=training_arguments,
        train_dataset=packed_train_dataset,
        dataset_text_field="text",
        max_seq_length=max_seq_length,
        tokenizer=tokenizer,
        packing=True,
        callbacks=my_callbacks,  # ← new line
    )

    if trainer.is_fsdp_enabled:
        logging.info(
            "Trainer running with FSDP for full-parameter continual pre-training."
        )
    else:
        logging.warning(
            "Trainer running without FSDP. This is not recommended for full-parameter CPT."
        )

    train_result = trainer.train(
        resume_from_checkpoint=training_arguments.resume_from_checkpoint
    )

    metrics = train_result.metrics
    try:
        metrics["train_samples"] = len(packed_train_dataset)
    except TypeError:
        metrics["train_samples"] = "streaming_or_unknown"
    trainer.log_metrics("train", metrics)
    trainer.save_metrics("train", metrics)
    trainer.save_state()  # RNG, scaler, sched …

    # save full weights/config one last time
    trainer.save_model(training_arguments.output_dir)  # same as output_dir
    logging.info("✓ Model weights written to %s", training_arguments.output_dir)

    # build a lightweight model card & ensure config has use_cache=True
    if trainer.accelerator.is_main_process:
        # adapt these fields to your arg names / dataset mixer
        card_kwargs = {
            "finetuned_from": pretrained_model_name_or_path,
            "dataset": [train_dataset],  # or list(data_args.dataset_mixer)
            "dataset_tags": [train_dataset],
            "tags": ["continual-pretraining"],
        }
        trainer.create_model_card(**card_kwargs)

        # re-enable KV cache for inference and push the updated config
        trainer.model.config.use_cache = True
        trainer.model.config.save_pretrained(training_arguments.output_dir)

    logging.info(
        "Continual pre-training complete. Model pushed to: https://huggingface.co/%s",
        repo_id,
    )


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
        throughput_measure=_THRPUT_MEASURE.value,
        streaming=_STREAMING.value,
        precision_mode=_PRECISION_MODE.value,
        train_precision=_TRAIN_PRECISION.value,
        optimizer=_OPTIMIZER.value,
        save_steps=_SAVE_STEPS.value,
        logging_steps=_LOGGING_STEPS.value,
        access_token=_HUGGINGFACE_ACCESS_TOKEN.value,
        repo_id=_HF_REPO.value,
    )
    utils.force_gc()


if __name__ == "__main__":
    app.run(main)
