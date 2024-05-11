import tensorflow as tf
import numpy as np

import layers

data_shape = [50, 50]
num_data = 200

class test_model(tf.keras.Model):
    def __init__(self):
        super(test_model, self).__init__()

        self.conv1D_module = layers.CustomConv1D(data_shape[1], 10, normalization="instance")
        self.rnn_module = layers.CustomRNN(data_shape[1], lightweight=True, dropout_ratio=0.3, enable_regularization=False, normalization="layer")
        self.inception_module = layers.Inception(16, 5, lightweight=True, scale_down_mode=2, activation="hsigmoid")
        self.conv2d_module = layers.CustomConv2D(64, 3, normalization="group", activation="htanh", kernal_clip_value=0.1)
        self.flatten_module = layers.GentalFlatten(10, 5, data_shape[0], 64)
        self.dense_module = layers.CustomDense(1, normalization=False, activation="linear")

    def call(self, x, training):
        x_conv1d = tf.expand_dims(self.conv1D_module(x, training), axis=3)
        x_rnn = tf.expand_dims(self.rnn_module(x, training), axis=3)
        x_concat = tf.concat([x_conv1d, x_rnn], axis=-1)
        x_inception = self.inception_module(x_concat, training)
        x_conv2d = self.conv2d_module(x_inception, training)
        x_flatten = self.flatten_module(x_conv2d, training)
        x_dense = self.dense_module(x_flatten, training)
        return x_dense

dataset_positive = np.random.uniform(low=-0.5, high=1, size=[num_data]+data_shape)
dataset_negative = np.random.uniform(low=-1, high=0.5, size=[num_data]+data_shape)
label_positive = [1] * num_data
label_negative = [0] * num_data

dataset = np.concatenate([dataset_positive, dataset_negative], axis=0)
label = np.concatenate([label_positive, label_negative], axis=0)
dataset = tf.data.Dataset.from_tensor_slices((dataset, label))
dataset = dataset.shuffle(dataset.cardinality())

split = int(num_data*2*0.8)
train_dataset = dataset.take(split)
train_dataset = train_dataset.shuffle(train_dataset.cardinality(), reshuffle_each_iteration=True)
train_dataset = train_dataset.batch(10, drop_remainder=False)

validation_dataset = dataset.skip(split)
validation_dataset = validation_dataset.shuffle(validation_dataset.cardinality(), reshuffle_each_iteration=True)
validation_dataset = validation_dataset.batch(10, drop_remainder=False)

model = test_model()
model.compile(optimizer=tf.keras.optimizers.AdamW(), loss=tf.keras.losses.BinaryCrossentropy(from_logits=True), metrics=tf.keras.metrics.BinaryAccuracy(threshold=0))
model.fit(train_dataset, epochs=10, validation_data=validation_dataset)
