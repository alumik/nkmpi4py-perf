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

    dataset = mnist_dataset(batch_size)

    model = get_model(configs)
    model.compile(optimizer=tf.keras.optimizers.SGD(learning_rate=learning_rate),
                  loss='sparse_categorical_crossentropy',
                  metrics=['acc'])

    model.fit(dataset,
              epochs=epochs,
              steps_per_epoch=steps_per_epoch)

    print(f'Execution time: {time.time() - s_time}')


if __name__ == '__main__':
    fire.Fire(main)
