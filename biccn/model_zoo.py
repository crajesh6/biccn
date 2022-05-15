
import tensorflow as tf
from tensorflow import keras

class RevCompConv1D(tf.keras.layers.Conv1D):
  """
  Implement forward and reverse-complement filter convolutions
  for 1D signals. It takes as input either a single input or two inputs
  (where the second input is the reverse complement scan). If a single input,
  this performs both forward and reverse complement scans and either merges it
  (if concat=True) or returns a separate scan for forward and reverse comp.
  """
  def __init__(self, *args, concat=False, **kwargs):
    super(RevCompConv1D, self).__init__(*args, **kwargs)
    self.concat = concat


  def call(self, inputs, inputs2=None):

    if inputs2 is not None:
      # create rc_kernels
      rc_kernel = self.kernel[::-1,::-1,:]

      # convolution 1D
      outputs = self._convolution_op(inputs, self.kernel)
      rc_outputs = self._convolution_op(inputs2, rc_kernel)

    else:
      # create rc_kernels
      rc_kernel = tf.concat([self.kernel, self.kernel[::-1,::-1,:]], axis=-1)

      # convolution 1D
      outputs = self._convolution_op(inputs, rc_kernel)

      # unstack to forward and reverse strands
      outputs = tf.unstack(outputs, axis=2)
      rc_outputs = tf.stack(outputs[self.filters:], axis=2)
      outputs = tf.stack(outputs[:self.filters], axis=2)

    # add bias
    if self.use_bias:
      outputs = tf.nn.bias_add(outputs, self.bias)
      rc_outputs = tf.nn.bias_add(rc_outputs, self.bias)

    # add activations
    if self.activation is not None:
      outputs = self.activation(outputs)
      rc_outputs = self.activation(rc_outputs)

    if self.concat:
      return tf.concat([outputs, rc_outputs], axis=-1)
    else:
      return outputs, rc_outputs


def dilated_residual_block(input_layer, filter_size, rates):
    num_filters = input_layer.shape.as_list()[-1]
    nn = tf.keras.layers.Conv1D(
        filters=num_filters,
        kernel_size=filter_size,
        activation=None,
        use_bias=False,
        padding='same',
        dilation_rate=rates[0],
    )(input_layer)
    nn = tf.keras.layers.BatchNormalization()(nn)

    for f in rates[1:]:
        nn = tf.keras.layers.Activation('relu')(nn)
        nn = tf.keras.layers.Dropout(0.1)(nn)
        nn = tf.keras.layers.Conv1D(
            filters=num_filters,
            kernel_size=filter_size,
            strides=1,
            activation=None,
            use_bias=False,
            padding='same',
            dilation_rate=f
        )(nn)

    nn = tf.keras.layers.BatchNormalization()(nn)
    nn = tf.keras.layers.add([input_layer, nn])
    return tf.keras.layers.Activation('relu')(nn)


def residualbind(input_shape, output_shape, activation='exponential', num_units=[128, 256, 512, 512], rc=False):

    # input layer
    inputs = tf.keras.layers.Input(shape=input_shape)

    # layer 1
    if rc:
        nn = RevCompConv1D(filters=num_units[0], kernel_size=19, use_bias=False, padding='same', concat=True)(inputs)
    else:
        nn = tf.keras.layers.Conv1D(
            filters=num_units[0],
            kernel_size=19,
            strides=1,
            activation=None,
            use_bias=False,
            padding='same'
        )(inputs)

    nn = tf.keras.layers.BatchNormalization()(nn)
    nn = tf.keras.layers.Activation(activation)(nn)
    nn = tf.keras.layers.Dropout(0.1)(nn)

    # dilated residual block
    nn = dilated_residual_block(nn, filter_size=3, rates=[1, 2, 4, 8])
    nn = tf.keras.layers.MaxPooling1D(pool_size=10)(nn) # 500
    nn = tf.keras.layers.Dropout(0.2)(nn)

    # layer 2
    nn = tf.keras.layers.Conv1D(
        filters=num_units[1],
        kernel_size=7,
        strides=1,
        activation=None,
        use_bias=False,
        padding='same',
    )(nn)
    nn = tf.keras.layers.BatchNormalization()(nn)
    nn = tf.keras.layers.Activation('relu')(nn)
    nn = tf.keras.layers.Dropout(0.1)(nn)

    # dilated residual block
    nn = dilated_residual_block(nn, filter_size=3, rates=[1, 2, 4])
    nn = tf.keras.layers.MaxPooling1D(pool_size=10)(nn) # 50
    nn = tf.keras.layers.Dropout(0.2)(nn)

  # layer 2
    nn = tf.keras.layers.Conv1D(
        filters=num_units[2],
        kernel_size=7,
        strides=1,
        activation=None,
        use_bias=False,
        padding='same',
    )(nn)
    nn = tf.keras.layers.BatchNormalization()(nn)
    nn = tf.keras.layers.Activation('relu')(nn)

    nn = tf.keras.layers.GlobalAveragePooling1D()(nn) # 1
    nn = tf.keras.layers.Dropout(0.3)(nn)

    # Fully-connected NN
    nn = tf.keras.layers.Flatten()(nn)
    nn = tf.keras.layers.Dense(num_units[3], activation=None, use_bias=False)(nn)
    nn = tf.keras.layers.BatchNormalization()(nn)
    nn = tf.keras.layers.Activation('relu')(nn)
    nn = tf.keras.layers.Dropout(0.5)(nn)

    # output layer
    logits = tf.keras.layers.Dense(output_shape, activation='linear', use_bias=True)(nn)
    outputs = tf.keras.layers.Activation('sigmoid')(logits)

    # create and return model
    return tf.keras.Model(inputs=inputs, outputs=outputs)


def remainder(input_shape, output_shape, activation='exponential', num_units=[128, 256, 512, 512]):

    # input layer
    inputs = tf.keras.layers.Input(shape=input_shape)

    nn = tf.keras.layers.Dropout(0.1)(inputs)

    # dilated residual block
    nn = dilated_residual_block(nn, filter_size=3, rates=[1, 2, 4, 8])
    nn = tf.keras.layers.MaxPooling1D(pool_size=10)(nn) # 500
    nn = tf.keras.layers.Dropout(0.2)(nn)

    # layer 2
    nn = tf.keras.layers.Conv1D(
        filters=num_units[1],
        kernel_size=7,
        strides=1,
        activation=None,
        use_bias=False,
        padding='same',
    )(nn)
    nn = tf.keras.layers.BatchNormalization()(nn)
    nn = tf.keras.layers.Activation('relu')(nn)
    nn = tf.keras.layers.Dropout(0.1)(nn)

    # dilated residual block
    nn = dilated_residual_block(nn, filter_size=3, rates=[1, 2, 4])
    nn = tf.keras.layers.MaxPooling1D(pool_size=10)(nn) # 50
    nn = tf.keras.layers.Dropout(0.2)(nn)

  # layer 2
    nn = tf.keras.layers.Conv1D(
        filters=num_units[2],
        kernel_size=7,
        strides=1,
        activation=None,
        use_bias=False,
        padding='same',
    )(nn)
    nn = tf.keras.layers.BatchNormalization()(nn)
    nn = tf.keras.layers.Activation('relu')(nn)

    nn = tf.keras.layers.GlobalAveragePooling1D()(nn) # 1
    nn = tf.keras.layers.Dropout(0.3)(nn)

    # Fully-connected NN
    nn = tf.keras.layers.Flatten()(nn)
    nn = tf.keras.layers.Dense(num_units[3], activation=None, use_bias=False)(nn)
    nn = tf.keras.layers.BatchNormalization()(nn)
    nn = tf.keras.layers.Activation('relu')(nn)
    nn = tf.keras.layers.Dropout(0.5)(nn)

    # output layer
    logits = tf.keras.layers.Dense(output_shape, activation='linear', use_bias=True)(nn)
    outputs = tf.keras.layers.Activation('sigmoid')(logits)

    # create and return model
    return tf.keras.Model(inputs=inputs, outputs=outputs)
