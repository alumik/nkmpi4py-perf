#!/bin/bash

task="$1"
mode="$2"
n_tasks="$3"
args="${*:4}"

rm -rf out/"$task"/"$mode"
mkdir -p out/"$task"/"$mode"

srun \
  -N "$n_tasks" \
  -n "$n_tasks" \
  -c 16 \
  -p nankai \
  -M cab50_59 \
  -u \
  -o out/"$task"/"$mode"/out.%t \
  --mpi=pmix \
  -x "cn[60555-60583,60416,60863]" \
  python src/"$task"/"$mode".py --epochs 1 --block_hidden_dims "(2048, 2048)" "$args"
