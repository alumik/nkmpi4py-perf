# Performance Tests for nkmpi4py

[![license-MIT](https://img.shields.io/badge/license-MIT-green)](https://github.com/AlumiK/nkmpi4py-perf/blob/main/LICENSE)

This repository tests the performance of 4 types of training strategies using Slurm Workload Manager:

- Non-distributed training
- `tf.distribute.experimental.MultiWorkerMirroredStrategy`
- `tf.distribute.experimental.ParameterServerStrategy`
- Distributed training with [nkmpi4py](https://github.com/alumik/nkmpi4py).

## Usage

```
NAME
    run.sh

SYNOPSIS
    bash run.sh <MODE> <N_TASKS> [FLAGS]...

MODE
    non_distributed
    multi_worker_mirrored
    parameter_server
    nkmpi4py_allreduce

N_TASKS
    The number of parallel tasks to run.

FLAGS
    --learning_rate
        Type: float
        Default: 0.001
    --batch_size
        Type: int
        Default: 64
    --epochs
        Type: int
        Default: 10
    --steps_per_epoch
        Type: int
        Default: 70
    --input_hidden_dims
        Type: typing.Sequence
        Default: (128,)
    --block_hidden_dims
        Type: typing.Sequence
        Default: (64, 64)
    --n_block
        Type: int
        Default: 2
    --dropout_rate
        Type: float
        Default: 0.2
    --allreduce_dims
        Type: typing.Sequence
        Default: (10, 10, 5)
```
