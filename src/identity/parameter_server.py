import os
import json
import fire
import time
import tensorflow as tf

from typing import Sequence, Dict, Optional


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


def mnist_dataset(batch_size: int) -> tf.data.Dataset:
    (x_train, y_train), _ = tf.keras.datasets.mnist.load_data()
    x_train = x_train / 255.0
    dataset = tf.data.Dataset \
        .from_tensor_slices((x_train, y_train)) \
        .shuffle(len(x_train)) \
        .repeat() \
        .batch(batch_size)
    return dataset


def get_model(configs: Optional[Dict] = None) -> tf.keras.Model:
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
    return model


def main(learning_rate: float = 0.001,
         batch_size: int = 64,
         epochs: int = 10,
         steps_per_epoch: int = 70,
         block_hidden_dims: Sequence = (64, 64),
         n_block: int = 2):
    s_time = time.time()

    configs = {
        'block_hidden_dims': block_hidden_dims,
        'n_block': n_block
    }

    strategy = tf.distribute.experimental.ParameterServerStrategy()

    global_batch_size = batch_size * n_tasks
    multi_worker_dataset = mnist_dataset(global_batch_size)

    with strategy.scope():
        model = get_model(configs)
        model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=learning_rate),
                      loss='sparse_categorical_crossentropy',
                      metrics=['acc'])

    model.fit(multi_worker_dataset,
              epochs=epochs,
              steps_per_epoch=steps_per_epoch)

    print(f'Execution time: {time.time() - s_time}')


if __name__ == '__main__':
    # Using Slurm
    # import hostlist
    #
    # task_index = int(os.environ['SLURM_PROCID'])
    # n_tasks = int(os.environ['SLURM_NPROCS'])
    # tf_host_list = [f'{host}:22222' for host in hostlist.expand_hostlist(os.environ['SLURM_NODELIST'])]

    # Using MPI
    from mpi4py import MPI

    comm = MPI.COMM_WORLD
    task_index = comm.Get_rank()
    n_tasks = comm.Get_size()
    tf_host_list = [f'localhost:{22222 + i}' for i in range(n_tasks)]

    local_task_index = 0
    if task_index == 0:
        task_type = 'chief'
    elif task_index == 1:
        task_type = 'ps'
    else:
        task_type = 'worker'
        local_task_index = task_index - 2
    tf_config = {
        'cluster': {
            'chief': tf_host_list[0:1],
            'ps': tf_host_list[1:2],
            'worker': tf_host_list[2:]
        },
        'task': {
            'type': task_type,
            'index': local_task_index
        }
    }
    os.environ['TF_CONFIG'] = json.dumps(tf_config)
    tf.compat.v1.disable_eager_execution()
    fire.Fire(main)
