import os
import tensorflow as tf

from typing import Sequence, Tuple


def mnist_dataset(batch_size: int) -> tf.data.Dataset:
    (x_train, y_train), _ = tf.keras.datasets.mnist.load_data()
    x_train = x_train / 255.0
    dataset = tf.data.Dataset \
        .from_tensor_slices((x_train, y_train)) \
        .shuffle(len(x_train)) \
        .repeat() \
        .batch(batch_size)
    return dataset


def mnist_mpi_dataset(comm, batch_size: int = 64) -> tf.data.Dataset:
    size = comm.Get_size()
    rank = comm.Get_rank()
    (x_train, y_train), _ = tf.keras.datasets.mnist.load_data()
    x_train = x_train / 255.0
    n_train_proc = len(x_train) // size
    x_train_proc = x_train[rank * n_train_proc:(rank + 1) * n_train_proc]
    y_train_proc = y_train[rank * n_train_proc:(rank + 1) * n_train_proc]
    dataset = tf.data.Dataset \
        .from_tensor_slices((x_train_proc, y_train_proc)) \
        .shuffle(n_train_proc) \
        .repeat() \
        .batch(batch_size)
    return dataset


def init(mode: str = 'slurm') -> Tuple[Sequence, int, int]:
    if mode == 'slurm':
        import hostlist
        task_index = int(os.environ['SLURM_PROCID'])
        n_tasks = int(os.environ['SLURM_NPROCS'])
        tf_host_list = [f'{host}:22222' for host in hostlist.expand_hostlist(os.environ['SLURM_NODELIST'])]
    else:
        from mpi4py import MPI
        comm = MPI.COMM_WORLD
        task_index = comm.Get_rank()
        n_tasks = comm.Get_size()
        tf_host_list = [f'localhost:{22222 + i}' for i in range(n_tasks)]
    return tf_host_list, n_tasks, task_index
