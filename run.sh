#!/bin/bash

mode="$1"
n_tasks="$2"
args="${*:3}"

rm -rf out/"$mode"
mkdir -p out/"$mode"

srun \
  -N "$n_tasks" \
  -n "$n_tasks" \
  -c 16 \
  -p nankai \
  -M cab50_59 \
  -u \
  -o out/"$mode"/out.%t \
  --mpi=pmix \
  -x "cn[60555-60583,60416,60863]" \
  python src/"$mode".py --epochs 1 --block_hidden_dims "(2048, 2048)" "$args"
