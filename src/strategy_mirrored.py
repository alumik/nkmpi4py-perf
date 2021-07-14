import tensorflow as tf


def main():
    epochs = 10
    batch_size = 64
    strategy = tf.distribute.MirroredStrategy(['GPU:0', 'GPU:1'])

    global_batch_size = batch_size * 2
    (x_train, y_train), _ = tf.keras.datasets.mnist.load_data()
    x_train = x_train / 255.0
    dataset = tf.data.Dataset \
        .from_tensor_slices((x_train, y_train)) \
        .shuffle(len(x_train)) \
        .batch(global_batch_size)

    with strategy.scope():
        i = tf.keras.Input(shape=(28, 28))
        x = tf.keras.layers.Flatten()(i)
        x = tf.keras.layers.Dense(128)(x)
        x = tf.keras.layers.Dense(64)(x)
        x = tf.keras.layers.Dense(10, activation='softmax')(x)
        model = tf.keras.Model(i, x)
        model.compile(optimizer='adam',
                      loss='sparse_categorical_crossentropy',
                      metrics=['acc'])

    model.fit(dataset, epochs=epochs)


if __name__ == '__main__':
    main()
