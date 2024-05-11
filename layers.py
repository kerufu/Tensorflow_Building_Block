import tensorflow as tf
import numpy as np


class ClipConstraint(tf.keras.constraints.Constraint):
    def __init__(self, kernal_clip_value) -> None:
        super(ClipConstraint, self).__init__()
        self.kernal_clip_value = kernal_clip_value

    def __call__(self, weights):
        return tf.keras.backend.clip(weights, -self.kernal_clip_value, self.kernal_clip_value)

    def get_config(self):
        return {'kernal_clip_value': self.kernal_clip_value}


class HardTanh(tf.keras.layers.Layer):
    def call(self, x):
        return tf.minimum(tf.maximum(x, -1), 1)


class HardSigmoid(tf.keras.layers.Layer):
    def call(self, x):
        return tf.minimum(tf.maximum(x+0.5, 0), 1)


class HardSwish(tf.keras.layers.Layer):
    def call(self, x):
        return x * tf.minimum(tf.maximum(x+3, 0), 6) / 6


class ReflectRadding(tf.keras.layers.Layer):  # O=[(W−K+P)/S]+1
    def __init__(self, kernal_size):
        super(ReflectRadding, self).__init__()
        pad = kernal_size - 1
        self.upper_pad = pad // 2
        self.lower_pad = pad - self.upper_pad

    def call(self, x):
        return tf.pad(x, [[0, 0], [self.upper_pad, self.lower_pad], [self.upper_pad, self.lower_pad], [0, 0]], 'REFLECT')


class CustomLayer(tf.keras.layers.Layer):
    def __init__(self, kernal_clip_value, enable_regularization, normalization, activation, dropout_ratio):
        super(CustomLayer, self).__init__()
        self.module = []

        self.kernel_constraint = None
        if kernal_clip_value > 0:
            self.kernel_constraint = ClipConstraint(kernal_clip_value)

        self.kernel_regularizer = None
        if enable_regularization:
            self.kernel_regularizer = tf.keras.regularizers.L1L2()

        self.normalization = normalization

        self.activation = activation

        self.dropout_ratio = dropout_ratio

    def add_post_layers(self):
        if self.normalization == "batch":
            self.module.append(tf.keras.layers.BatchNormalization())
        elif self.normalization == "layer":
            self.module.append(tf.keras.layers.LayerNormalization())
        elif self.normalization == "group":
            self.module.append(tf.keras.layers.GroupNormalization()) # number of channel must be multiply of 32
        elif self.normalization == "instance":
            self.module.append(tf.keras.layers.GroupNormalization(groups=-1))

        if self.activation == "hswish":
            self.module.append(HardSwish())
        elif self.activation == "htanh":
            self.module.append(HardTanh())
        elif self.activation == "hsigmoid":
            self.module.append(HardSigmoid())
        else:
            self.module.append(tf.keras.layers.Activation(self.activation))

        if self.dropout_ratio:
            self.module.append(tf.keras.layers.Dropout(self.dropout_ratio))

    def call(self, x, training):
        for layer in self.module:
            if "dropout" in layer.name or "batch_normalization" in layer.name:
                x = layer(x, training)
            else:
                x = layer(x)
        return x


class CustomConv2D(CustomLayer):

    def __init__(self, num_channel, kernal_size,
                 lightweight=False, reflect_padding=True, scale_down_mode=0,
                 kernal_clip_value=0, enable_regularization=True, normalization="batch", activation="hswish", dropout_ratio=0):
        super().__init__(kernal_clip_value, enable_regularization,
                         normalization, activation, dropout_ratio)

        if lightweight:
            Conv2D = tf.keras.layers.SeparableConv2D
        else:
            Conv2D = tf.keras.layers.Conv2D

        if reflect_padding:
            self.module.append(ReflectRadding(kernal_size))
            padding = "valid"
        else:
            padding = "same"

        if scale_down_mode == 1:
            self.module.append(Conv2D(num_channel, kernal_size, strides=2, padding=padding,
                               kernel_regularizer=self.kernel_regularizer, kernel_constraint=self.kernel_constraint))
        else:
            self.module.append(Conv2D(num_channel, kernal_size, padding=padding,
                               kernel_regularizer=self.kernel_regularizer, kernel_constraint=self.kernel_constraint))
            if scale_down_mode == 2:
                self.module.append(tf.keras.layers.MaxPool2D())

        self.add_post_layers()


class CustomConv1D(CustomLayer):

    def __init__(self, num_channel, kernal_size=5,
                 kernal_clip_value=0, enable_regularization=True, normalization="batch", activation="hswish", dropout_ratio=0):
        super().__init__(kernal_clip_value, enable_regularization,
                         normalization, activation, dropout_ratio)

        self.module.append(tf.keras.layers.Conv1D(num_channel, kernal_size, padding='same',
                                                  kernel_regularizer=self.kernel_regularizer, kernel_constraint=self.kernel_constraint))

        self.add_post_layers()


class CustomDense(CustomLayer):

    def __init__(self, output_size,
                 kernal_clip_value=0, enable_regularization=True, normalization="batch", activation="hswish", dropout_ratio=0):
        super().__init__(kernal_clip_value, enable_regularization,
                         normalization, activation, dropout_ratio)

        self.module.append(
            tf.keras.layers.Dense(
                output_size, kernel_regularizer=self.kernel_regularizer, kernel_constraint=self.kernel_constraint),
        )
        self.add_post_layers()


class CustomRNN(CustomLayer):

    def __init__(self, output_size,
                 lightweight=False, return_sequences=True,
                 kernal_clip_value=0, enable_regularization=True, normalization="batch", activation="hswish", dropout_ratio=0):
        super().__init__(kernal_clip_value, enable_regularization,
                         normalization, activation, dropout_ratio)

        if lightweight:
            RNN = tf.keras.layers.GRU
        else:
            RNN = tf.keras.layers.LSTM

        self.module.append(
            RNN(output_size, recurrent_activation=HardSigmoid(), return_sequences=return_sequences,
                kernel_regularizer=self.kernel_regularizer, kernel_constraint=self.kernel_constraint),
        )

        self.add_post_layers()


class GentalFlatten(tf.keras.layers.Layer):
    def __init__(self, output_size, kernal_size, image_size, num_input_channel):
        super(GentalFlatten, self).__init__()
        num_flatten_layers = int(np.ceil(np.log2(image_size)))
        flatten_step = np.power(
            output_size/num_input_channel, 1/num_flatten_layers)
        self.output_module = [
            CustomConv2D(int(num_input_channel*(flatten_step**(index+1))), kernal_size, reflect_padding=False, scale_down_mode=1) for index in range(num_flatten_layers-1)
        ]
        self.output_module.append(CustomConv2D(
            output_size, kernal_size, reflect_padding=False, normalization=None, activation="linear", scale_down_mode=1))
        self.output_module.append(tf.keras.layers.Flatten())

    def call(self, x, training):
        for layer in self.output_module:
            if "custom" in layer.name:
                x = layer(x, training)
            else:
                x = layer(x)
        return x


class InceptionLayer(tf.keras.layers.Layer):
    def __init__(self, num_channel, kernal_size,
                 num_field=4, lightweight=False, reflect_padding=True, scale_down_mode=0,
                 kernal_clip_value=0, enable_regularization=True, normalization="batch", activation="hswish", dropout_ratio=0):
        super(InceptionLayer, self).__init__()

        if num_channel % num_field:
            self.conv2d_cluster = [CustomConv2D(num_channel//num_field+num_channel % num_field, kernal_size,
                                                lightweight, reflect_padding, scale_down_mode,
                                                kernal_clip_value, enable_regularization, normalization, activation, dropout_ratio)]
        else:
            self.conv2d_cluster = [CustomConv2D(num_channel//num_field, kernal_size,
                                                lightweight, reflect_padding, scale_down_mode,
                                                kernal_clip_value, enable_regularization, normalization, activation, dropout_ratio)]
        for k in range(1, num_field):
            self.conv2d_cluster.append(CustomConv2D(num_channel//num_field, kernal_size+k,
                                                    lightweight, reflect_padding, scale_down_mode,
                                                    kernal_clip_value, enable_regularization, normalization, activation, dropout_ratio))

    def call(self, x, training):
        output = []
        for layer in self.conv2d_cluster:
            output.append(layer(x, training))
        return tf.concat(output, -1)
