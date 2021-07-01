import fire
import time

from mpi4py import MPI
from nkmpi4py import NKMPI_TF
from typing import Sequence, Tuple
from utils import *
from models import *


def get_mpi_model(comm, configs: Optional[Dict] = None) -> tf.keras.Model:
    model = get_model(configs)
    rank = comm.Get_rank()
    weights = None
    if rank == 0:
        weights = model.get_weights()
    weights = comm.bcast(weights)
    model.set_weights(weights)
    return model


def main(learning_rate: float = 0.001,
         batch_size: int = 64,
         epochs: int = 10,
         steps_per_epoch: int = 70,
         block_hidden_dims: Sequence = (64, 64),
         n_block: int = 2,
         allreduce_dims: Sequence = (10,)):
    s_time = time.time()

    configs = {
        'block_hidden_dims': block_hidden_dims,
        'n_block': n_block
    }

    comm = MPI.COMM_WORLD
    comm = NKMPI_TF.Comm(comm, new_dims=allreduce_dims)

    multi_worker_dataset = mnist_mpi_dataset(comm=comm, batch_size=batch_size)

    optimizer = tf.keras.optimizers.Adam(learning_rate=learning_rate)
    loss_fn = tf.keras.losses.SparseCategoricalCrossentropy()
    metrics = tf.keras.metrics.SparseCategoricalAccuracy()
    model = get_mpi_model(comm=comm, configs=configs)

    def train_step(_x: tf.Tensor, _y: tf.Tensor) -> Tuple:
        with tf.GradientTape() as tape:
            y_pred = model(_x)
            _loss = loss_fn(_y, y_pred)
        grads = tape.gradient(_loss, model.trainable_weights)
        grads = [comm.allreduce(grad.numpy(), op=MPI.SUM) for grad in grads]
        optimizer.apply_gradients(zip(grads, model.trainable_weights))
        _loss = comm.allreduce(_loss.numpy(), op=MPI.SUM)
        metrics.update_state(_y, y_pred)
        _acc = comm.allreduce(metrics.result().numpy(), op=MPI.SUM) / comm.Get_size()
        return _loss, _acc

    for epoch in range(epochs):
        progbar = tf.keras.utils.Progbar(steps_per_epoch, stateful_metrics=['loss'])
        dataset_iter = iter(multi_worker_dataset)
        for _ in range(steps_per_epoch):
            x, y = next(dataset_iter)
            loss, acc = train_step(x, y)
            progbar.add(1, values=[('loss', loss), ('acc', acc)])

    print(f'Execution time: {time.time() - s_time}')


if __name__ == '__main__':
    fire.Fire(main)
