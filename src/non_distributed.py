import fire
import time

from typing import Sequence
from utils import *
from models import *


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
