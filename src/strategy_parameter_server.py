import json
import fire
import time

from utils import *
from models import *


def set_tf_config(mode: str = 'slurm') -> int:
    tf_host_list, n_tasks, task_index = init(mode)
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
    return n_tasks


def main(learning_rate: float = 0.001,
         batch_size: int = 64,
         epochs: int = 10,
         steps_per_epoch: int = 70,
         block_hidden_dims: Sequence = (64, 64),
         n_block: int = 2,
         mode: str = 'slurm'):
    s_time = time.time()
    n_tasks = set_tf_config(mode)

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
    tf.compat.v1.disable_eager_execution()
    fire.Fire(main)
