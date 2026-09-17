from __future__ import annotations


def build_keras_cnn(input_length: int, output_dim: int, architecture: str, task: str):
    """Build the Assignment 04 Conv1D graph when TensorFlow is installed."""
    import tensorflow as tf

    tf.random.set_seed(42)
    layers = [tf.keras.layers.Input((input_length, 1)), tf.keras.layers.Conv1D(16, 3, padding="same"), tf.keras.layers.ReLU()]
    if architecture == "3-layer":
        layers += [tf.keras.layers.Conv1D(8, 3, padding="same"), tf.keras.layers.ReLU()]
    elif architecture == "5-layer":
        layers += [tf.keras.layers.Conv1D(16, 3, padding="same"), tf.keras.layers.ReLU(), tf.keras.layers.MaxPooling1D(2), tf.keras.layers.Conv1D(8, 3, padding="same"), tf.keras.layers.ReLU()]
    else:
        raise ValueError("architecture must be '3-layer' or '5-layer'")
    layers += [tf.keras.layers.GlobalAveragePooling1D(), tf.keras.layers.Dense(output_dim, activation="sigmoid" if task == "binary" else None)]
    model = tf.keras.Sequential(layers)
    model.compile(optimizer=tf.keras.optimizers.Adam(), loss="binary_crossentropy" if task == "binary" else ("mse" if task == "regression" else "sparse_categorical_crossentropy"), metrics=["accuracy"] if task != "regression" else ["mae"])
    return model