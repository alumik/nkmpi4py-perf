import tensorflow as tf

from typing import Dict, Optional


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
