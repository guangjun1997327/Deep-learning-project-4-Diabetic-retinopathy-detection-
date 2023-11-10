import gin
import tensorflow as tf

from models.layers import vgg_block

@gin.configurable
def vgg_like(input_shape, arch, n_classes, dense_units, dropout_rate):
    inputs = tf.keras.Input(shape=(256, 256, 3))
    out = vgg_block(inputs, arch[0][2], arch[0][1], arch[0][0])
    for (num_layer, kernel_size, channel) in arch[1:]:
        out = vgg_block(out, channel, kernel_size, num_layer)
    out = tf.keras.layers.GlobalAveragePooling2D()(out)
    out = tf.keras.layers.Dense(dense_units, activation=tf.nn.relu)(out)
    out = tf.keras.layers.Dropout(dropout_rate)(out)
    outputs = tf.keras.layers.Dense(n_classes)(out)

    return tf.keras.Model(inputs=inputs, outputs=outputs, name='vgg_like')
