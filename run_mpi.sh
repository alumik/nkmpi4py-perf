#!/bin/bash

task="$1"
mode="$2"
n_tasks="$3"
args="${*:4}"

CUDA_VISIBLE_DEVICES=0 mpiexec \
  --allow-run-as-root \
  -n "$n_tasks" \
  python src/"$task"/"$mode".py --epochs 1 --block_hidden_dims "(2048, 2048)" "$args"
