"""Entrypoint for peft train docker.

Dispatches to different scripts based on `task` type.

For task type in `_TASK_TO_SCRIPT`, if `--config_file` is specified, the script
will dispatch the call to `accelerate`, which is friendly for multi-GPU
environment. Otherwise, `python3` is used.
"""

import argparse
from collections.abc import MutableSequence, Sequence
import json
# import multiprocessing
import os
import subprocess
import sys
from absl import app
from absl import flags
from absl import logging
# from util import dataset_validation_util
# from vertex_vision_model_garden_peft.train.vmg import utils
# from util import constants
# from util import gcs_syncer
# from util import hypertune_utils


def _append_args_to_command_in_place(
    args: argparse.Namespace, 
    command: MutableSequence[str]
):
  for key, value in vars(args).items():
    # If not specified, skip.
    if value is not None:
      command.append(f'--{key}={value}')

def _get_accelerate_args() -> argparse.Namespace:
  """Returns the accelerate args."""
  # For the format of the cluster spec, see
  # https://cloud.google.com/vertex-ai/docs/training/distributed-training#cluster-spec-format # pylint: disable=line-too-long
  cluster_spec = os.getenv('CLUSTER_SPEC', default=None)
  if not cluster_spec:
    return argparse.Namespace()
  logging.info('CLUSTER_SPEC: %s', cluster_spec)

  cluster_data = json.loads(cluster_spec)
  if (
      'workerpool1' not in cluster_data['cluster']
      or not cluster_data['cluster']['workerpool1']
  ):
    return argparse.Namespace()

  # Get primary node info
  primary_node = cluster_data['cluster']['workerpool0'][0]
  logging.info('primary node: %s', primary_node)
  primary_node_addr, primary_node_port = primary_node.split(':')
  logging.info('primary node address: %s', primary_node_addr)
  logging.info('primary node port: %s', primary_node_port)

  # Determine node rank of this machine
  workerpool = cluster_data['task']['type']
  if workerpool == 'workerpool0':
    node_rank = 0
  elif workerpool == 'workerpool1':
    # Add 1 for the primary node, since `index` is the index of workerpool1.
    node_rank = cluster_data['task']['index'] + 1
  else:
    raise ValueError(
        'Only workerpool0 and workerpool1 are supported. Unknown workerpool:'
        f' {workerpool}'
    )
  logging.info('node rank: %s', node_rank)

  # Calculate total nodes
  num_worker_nodes = len(cluster_data['cluster']['workerpool1'])
  num_nodes = num_worker_nodes + 1  # Add 1 for the primary node
  logging.info('num nodes: %s', num_nodes)

  accelerate_args = argparse.Namespace()
  accelerate_args.machine_rank = node_rank
  accelerate_args.num_machines = num_nodes
  accelerate_args.main_process_ip = primary_node_addr
  accelerate_args.main_process_port = primary_node_port
  accelerate_args.max_restarts = 0
  accelerate_args.monitor_interval = 120

  return accelerate_args


def launch_script_cmd(
    script: str,
    config_file: str | None,
    accelerate_args: argparse.Namespace = argparse.Namespace(),
    script_args: Sequence[str] | None = None,  # Added new parameter
) -> MutableSequence[str]:
  """Returns the command to launch the script."""
  
  cmd = [
      'accelerate',
      'launch',
  ]
  # Add config_file only if it's provided, to avoid '--config_file=None'
  if config_file:
    cmd.append(f'--config_file={config_file}')
  
  _append_args_to_command_in_place(accelerate_args, cmd)
  cmd.append(script)
  if script_args:  # Added block to extend with script_args
      cmd.extend(script_args)

  return cmd

def main(unused_argv: Sequence[str]) -> None:
  additional_flags = [
      "--pretrained_model_name_or_path", "google/gemma-3-12b-pt",
      "--train_dataset", "PolyAgent/kobza_wiki_filtered_dedupl",
      "--train_column", "text",
      "--output_dir", "/data/gemma3_cpt/test1/",
      "--per_device_train_batch_size", "1",
      "--gradient_accumulation_steps", "1",
      "--max_seq_length", "12000",
      "--precision_mode", "bfloat16",
      "--save_steps", "10000",
      "--logging_steps", "20",
      "--max_steps", "6000",
      "--attn_implementation", "flash_attention_2",
  ]

  cmd = launch_script_cmd(
      script="vertex_vision_model_garden_peft/train/vmg/gemma3_cpt_train.py",
      config_file="vertex_vision_model_garden_peft/gemma3_fsdp_8gpu_mykola.yaml",
      accelerate_args=_get_accelerate_args(),
      script_args=additional_flags
  )

  logging.info('Executing command: %s', ' '.join(cmd_list))
  
  # Create a copy of the environment to avoid modifying the current process's env
  env = os.environ.copy()
  
  logging.info('launching task=%s with cmd: \n%s', task, ' \\\n'.join(cmd))
  subprocess.run(cmd, stdout=sys.stdout, stderr=sys.stdout, check=True)

if __name__ == '__main__':
  logging.get_absl_handler().python_handler.stream = sys.stdout
  app.run(main, flags_parser=lambda _args: flags.FLAGS(_args, known_only=True))
