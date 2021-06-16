import fire
import time
import tensorflow as tf

from mpi4py import MPI
from nkmpi4py import NKMPI_TF
from typing import Sequence, Tuple, Dict, Optional


@tf.custom_gradient
def identity_op(x, kernel):
    def grad(dy):
        return tf.identity(x), tf.identity(kernel)

    return tf.identity(x), grad


class IdentityLayer(tf.keras.layers.Layer):

    def __init__(self, units: int, **kwargs):
        super().__init__(**kwargs)
        self._units = units
        self._kernel = self.add_weight(
            'kernel',
            shape=[self._units, self._units],
            trainable=True)

    def call(self, inputs, **kwargs):
        return identity_op(inputs[:, :self._units], self._kernel)


def mnist_dataset(comm, batch_size: int = 64) -> tf.data.Dataset:
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


def get_model(comm, configs: Optional[Dict] = None) -> tf.keras.Model:
    i = tf.keras.Input(shape=(28, 28))
    x = tf.keras.layers.Flatten()(i)

    n_block = configs.get('n_block', 2)
    block_hidden_dims = configs.get('block_hidden_dims', (64, 64))
    for _ in range(n_block):
        for block_dim in block_hidden_dims:
            x = IdentityLayer(block_dim)(x)

    x = IdentityLayer(10)(x)
    x = tf.keras.layers.Softmax()(x)

    model = tf.keras.Model(i, x)

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
         allreduce_dims: Sequence = (10, 10, 5)):
    s_time = time.time()

    configs = {
        'block_hidden_dims': block_hidden_dims,
        'n_block': n_block
    }

    comm = MPI.COMM_WORLD
    comm = NKMPI_TF.Comm(comm, new_dims=allreduce_dims)

    multi_worker_dataset = mnist_dataset(comm=comm, batch_size=batch_size)

    optimizer = tf.keras.optimizers.Adam(learning_rate=learning_rate)
    loss_fn = tf.keras.losses.SparseCategoricalCrossentropy()
    metrics = tf.keras.metrics.SparseCategoricalAccuracy()
    model = get_model(comm=comm, configs=configs)

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

    print('Training')
    for epoch in range(epochs):
        print(f'Epoch {epoch + 1}/{epochs}')
        progbar = tf.keras.utils.Progbar(steps_per_epoch, stateful_metrics=['loss'])
        dataset_iter = iter(multi_worker_dataset)
        for _ in range(steps_per_epoch):
            x, y = next(dataset_iter)
            loss, acc = train_step(x, y)
            progbar.add(1, values=[('loss', loss), ('acc', acc)])

    print(f'Execution time: {time.time() - s_time}')


if __name__ == '__main__':
    fire.Fire(main)
