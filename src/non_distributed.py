import fire
import time
import tensorflow as tf

from typing import Sequence, Dict, Optional


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
    for input_dim in configs.get('input_hidden_dims', (128,)):
        x = tf.keras.layers.Dense(input_dim, activation='relu')(x)

    n_block = configs.get('n_block', 2)
    block_hidden_dims = configs.get('block_hidden_dims', (64, 64))
    for _ in range(n_block):
        for block_dim in block_hidden_dims:
            x = tf.keras.layers.Dense(block_dim, activation='relu')(x)

    x = tf.keras.layers.Dropout(configs.get('dropout_rate', 0.2))(x)
    x = tf.keras.layers.Dense(10, activation='softmax')(x)

    model = tf.keras.Model(i, x)
    return model


def main(learning_rate: float = 0.001,
         batch_size: int = 64,
         epochs: int = 10,
         steps_per_epoch: int = 70,
         input_hidden_dims: Sequence = (128,),
         block_hidden_dims: Sequence = (64, 64),
         n_block: int = 2,
         dropout_rate: float = 0.2):
    s_time = time.time()

    configs = {
        'input_hidden_dims': input_hidden_dims,
        'block_hidden_dims': block_hidden_dims,
        'dropout_rate': dropout_rate,
        'n_block': n_block
    }

    dataset = mnist_dataset(batch_size)

    model = get_model(configs)
    model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=learning_rate),
                  loss='sparse_categorical_crossentropy',
                  metrics=['acc'])

    model.fit(dataset,
              epochs=epochs,
              steps_per_epoch=steps_per_epoch)

    print(f'Execution time: {time.time() - s_time}')


if __name__ == '__main__':
    fire.Fire(main)