# Performance Tests for nkmpi4py

[![license-MIT](https://img.shields.io/badge/license-MIT-green)](https://github.com/AlumiK/nkmpi4py-perf/blob/main/LICENSE)

This repository tests the performance of 3 types of parallel training strategies using Slurm Workload Manager or MPI:

- `tf.distribute.experimental.MultiWorkerMirroredStrategy`
- `tf.distribute.experimental.ParameterServerStrategy`
- Distributed training with [nkmpi4py](https://github.com/alumik/nkmpi4py).

## Usage

### Configurations

```shell
# Number of tasks to run.
n_tasks=10

# Available strategies: 
# 1) nkmpi4py
# 2) multi_worker_mirrored
# 3) parameter_server
strategy=nkmpi4py

# Available modes:
# 1) slurm
# 2) mpi
mode=slurm
```

### Use Slurm Workload Manager

```shell
yhrun -N "$n_tasks" -n "$n_tasks" -c 16 -p nankai -u --mpi pmix python src/strategy_"$strategy".py --mode "$mode" --epochs 1 --block_hidden_dims "(2048, 2048)" --allreduce_dims "($n_tasks,)"
```

### Use MPI

```shell
mpiexec -n "$n_tasks" python src/strategy_"$strategy".py --mode "$mode" --epochs 1 --block_hidden_dims "(2048, 2048)" --allreduce_dims "($n_tasks,)"
```

### Flags

```
--learning_rate=LEARNING_RATE
    Type: float
    Default: 0.001
--batch_size=BATCH_SIZE
    Type: int
    Default: 64
--epochs=EPOCHS
    Type: int
    Default: 10
--steps_per_epoch=STEPS_PER_EPOCH
    Type: int
    Default: 70
--block_hidden_dims=BLOCK_HIDDEN_DIMS
    Type: typing.Sequence
    Default: (64, 64)
--n_block=N_BLOCK
    Type: int
    Default: 2
--allreduce_dims=ALLREDUCE_DIMS
    Type: typing.Sequence
    Default: (10,)
 --mode=MODE
    Type: str
    Default: 'slurm'
```
