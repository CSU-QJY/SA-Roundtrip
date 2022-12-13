import tensorflow as tf
from tensorflow_addons.layers import SpectralNormalization, FilterResponseNormalization, TLU

from tensorflow.python.keras.layers import Dense, Conv2D, LeakyReLU, AveragePooling2D, Conv2DTranspose, \
    BatchNormalization,MaxPooling2D
from tensorflow.keras.applications import ResNet50V2

def try_Attention(inputs, ch):
    b, h, w, c = inputs.shape
    filters_f_g_h = ch // 8
    filters_v = ch
    gamma = tf.Variable(initial_value=[0], dtype=tf.float32, trainable=True)
    query = SpectralNormalization(
        tf.keras.layers.Conv2D(filters=filters_f_g_h, kernel_size=1, strides=1, use_bias=True))
    key = SpectralNormalization(
        tf.keras.layers.Conv2D(filters=filters_f_g_h, kernel_size=1, strides=1, use_bias=True))
    value = SpectralNormalization(
        tf.keras.layers.Conv2D(filters=filters_v, kernel_size=1, strides=1, use_bias=True))
    o = tf.keras.layers.Attention()([hw_flatten(key(inputs)), hw_flatten(value(inputs)), hw_flatten(query(inputs))])

    o = tf.reshape(o, shape=[-1, h, w, c])
    o = value(o, training=True)

    output = gamma * o + inputs
    return output


def google_attention(inputs, ch):
    b, h, w, c = inputs.shape
    gamma = tf.Variable(initial_value=[0], dtype=tf.float32, trainable=True)

    query = SpectralNormalization(
        tf.keras.layers.Conv2D(filters=ch // 8, kernel_size=1, strides=1, use_bias=True))(inputs)
    query = tf.keras.layers.AveragePooling2D()(query)

    key = SpectralNormalization(
        tf.keras.layers.Conv2D(filters=ch // 8, kernel_size=1, strides=1, use_bias=True))(inputs)

    value = SpectralNormalization(
        tf.keras.layers.Conv2D(filters=ch // 2, kernel_size=1, strides=1, use_bias=True))(inputs)
    value = tf.keras.layers.AveragePooling2D()(value)

    o = tf.keras.layers.Attention()([hw_flatten(key), hw_flatten(value), hw_flatten(query)])

    o = tf.reshape(o, shape=[-1, h, w, c // 2])
    o = SpectralNormalization(
        tf.keras.layers.Conv2D(filters=ch, kernel_size=1, strides=1, use_bias=True))(o)
    output = gamma * o + inputs
    return output


def google_attention_no_sn(inputs, ch):
    b, h, w, c = inputs.shape
    gamma = tf.Variable(initial_value=[0], dtype=tf.float32, trainable=True)

    query = tf.keras.layers.Conv2D(filters=ch // 8, kernel_size=1, strides=1, use_bias=True)(inputs)
    query = tf.keras.layers.MaxPooling2D()(query)

    key = tf.keras.layers.Conv2D(filters=ch // 8, kernel_size=1, strides=1, use_bias=True)(inputs)

    value = tf.keras.layers.Conv2D(filters=ch // 2, kernel_size=1, strides=1, use_bias=True)(inputs)
    value = tf.keras.layers.MaxPooling2D()(value)

    o = tf.keras.layers.Attention()([hw_flatten(key), hw_flatten(value), hw_flatten(query)])

    o = tf.reshape(o, shape=[-1, h, w, c // 2])
    o = tf.keras.layers.Conv2D(filters=ch, kernel_size=1, strides=1, use_bias=True)(o)
    output = gamma * o + inputs
    return output


def hw_flatten(x):
    b, h, w, c = x.shape
    return tf.reshape(x, shape=[-1, h * w, c])


def resblock_dense(inputs, units, weight_init, training=True):
    dense0 = Dense(units=units, kernel_initializer=weight_init)
    dropout = tf.keras.layers.Dropout(0.2)
    dense1 = Dense(units=units, kernel_initializer=weight_init)
    dense_skip = Dense(units=units, kernel_initializer=weight_init)
    l1 = BatchNormalization()(inputs)
    l1 = LeakyReLU()(l1)
    l1 = dense0(l1)
    l1 = dropout(l1, training=training)
    l2 = BatchNormalization()(l1)
    l2 = LeakyReLU()(l2)
    l2 = dense1(l2)
    l2 = dropout(l2, training=training)

    skip = dense_skip(inputs)

    output = l2 + skip

    return output


def resblock_down(inputs, channels, weight_init):
    conv0 = Conv2D(filters=channels,
                                         kernel_size=3,
                                         strides=1,
                                         padding='SAME',
                                         kernel_initializer=weight_init)
    conv1 = Conv2D(filters=channels,
                                         kernel_size=3,
                                         strides=1,
                                         padding='SAME',
                                         kernel_initializer=weight_init)
    skip_conv = Conv2D(filters=channels,
                                             kernel_size=1,
                                             strides=1,
                                             kernel_initializer=weight_init)
    x = LeakyReLU()(inputs)

    x = conv0(x)

    # res2
    x = LeakyReLU()(x)
    x = conv1(x)
    x = AveragePooling2D(pool_size=2, strides=2)(x)
    # skip
    x_init = skip_conv(inputs)
    x_init = AveragePooling2D(pool_size=2, strides=2)(x_init)

    return tf.keras.layers.Add()([x, x_init])


def resblock_en(inputs, channels, weight_init, use_bias=False):
    conv0 = Conv2D(filters=channels,
                   kernel_size=3,
                   strides=1,
                   padding='SAME',
                   use_bias=use_bias,
                   kernel_initializer=weight_init)
    conv1 = Conv2D(filters=channels,
                   kernel_size=3,
                   strides=1,
                   padding='SAME',
                   use_bias=use_bias,
                   kernel_initializer=weight_init)
    skip_conv = Conv2D(filters=channels,
                       kernel_size=1,
                       strides=1,
                       use_bias=use_bias,
                       kernel_initializer=weight_init)

    # x = LeakyReLU()(inputs)
    x = FilterResponseNormalization()(inputs)
    x = TLU()(x)
    x = conv0(x)

    # res2

    # x = LeakyReLU()(x)
    x = FilterResponseNormalization()(x)
    x = TLU()(x)
    x = conv1(x)

    x = AveragePooling2D(pool_size=2, strides=2)(x)
    # skip
    x_init = skip_conv(inputs)
    # x_init = FilterResponseNormalization()(x_init)
    # x_init = LeakyReLU()(x_init)
    x_init = AveragePooling2D(pool_size=2, strides=2)(x_init)

    return tf.keras.layers.Add()([x, x_init])


def resblock_sn(inputs, channels, weight_init):
    conv0 = SpectralNormalization(Conv2D(filters=channels,
                                         kernel_size=3,
                                         strides=1,
                                         padding='SAME',
                                         kernel_initializer=weight_init))
    conv1 = SpectralNormalization(Conv2D(filters=channels,
                                         kernel_size=3,
                                         strides=2,
                                         padding='SAME',
                                         kernel_initializer=weight_init))
    skip_conv = SpectralNormalization(Conv2D(filters=channels,
                                             kernel_size=1,
                                             strides=2,
                                             padding='SAME',
                                             kernel_initializer=weight_init))
    x = conv0(inputs)
    x = LeakyReLU()(x)

    x = conv1(x)
    # skip
    x_init = skip_conv(inputs)
    return tf.keras.layers.Add()([x, x_init])


def resblock(inputs, channels, weight_init):
    conv0 = Conv2D(filters=channels,
                   kernel_size=3,
                   strides=1,
                   padding='SAME',
                   kernel_initializer=weight_init)
    conv1 = Conv2D(filters=channels,
                   kernel_size=3,
                   strides=2,
                   padding='SAME',
                   kernel_initializer=weight_init)
    skip_conv = Conv2D(filters=channels,
                       kernel_size=1,
                       strides=2,
                       padding='SAME',
                       kernel_initializer=weight_init)
    x = conv0(inputs)
    x = FilterResponseNormalization()(x)
    x = TLU()(x)

    x = conv1(x)
    # skip
    x_init = skip_conv(inputs)
    return tf.keras.layers.Add()([x, x_init])


def resblock_up(inputs, channels, weight_init, use_bias=False, ):
    conv0 = SpectralNormalization(Conv2DTranspose(filters=channels,
                                                  kernel_size=3,
                                                  strides=2,
                                                  padding='SAME',
                                                  use_bias=use_bias,
                                                  kernel_initializer=weight_init))
    conv1 = SpectralNormalization(Conv2DTranspose(filters=channels,
                                                  kernel_size=3,
                                                  strides=1,
                                                  padding='SAME',
                                                  use_bias=use_bias,
                                                  kernel_initializer=weight_init))
    skip_conv = SpectralNormalization(Conv2DTranspose(filters=channels,
                                                      kernel_size=1,
                                                      strides=2,
                                                      use_bias=False,
                                                      kernel_initializer=weight_init))
    x = FilterResponseNormalization()(inputs)
    x = TLU()(x)

    x = conv0(x)
    # res2

    x = FilterResponseNormalization()(x)
    x = TLU()(x)
    x = conv1(x)
    # skip
    x_init = skip_conv(inputs)

    return tf.keras.layers.Add()([x, x_init])
from typing import Any, Dict, Optional, Union
from tensorflow.python.framework.convert_to_constants import convert_variables_to_constants_v2_as_graph
def try_count_flops(model: Union[tf.Module, tf.keras.Model],
                    inputs_kwargs: Optional[Dict[str, Any]] = None,
                    output_path: Optional[str] = None):
  if hasattr(model, 'inputs'):
    try:
      # Get input shape and set batch size to 1.
      if model.inputs:
        inputs = [
            tf.TensorSpec([1] + input.shape[1:], input.dtype)
            for input in model.inputs
        ]
        concrete_func = tf.function(model).get_concrete_function(inputs)
      else:
        concrete_func = tf.function(model.call).get_concrete_function(
            **inputs_kwargs)
      frozen_func, _ = convert_variables_to_constants_v2_as_graph(concrete_func)

      # Calculate FLOPs.
      run_meta = tf.compat.v1.RunMetadata()
      opts = tf.compat.v1.profiler.ProfileOptionBuilder.float_operation()
      if output_path is not None:
        opts['output'] = f'file:outfile={output_path}'
      else:
        opts['output'] = 'none'
      flops = tf.compat.v1.profiler.profile(
          graph=frozen_func.graph, run_meta=run_meta, options=opts)
      return flops.total_float_ops
    except Exception as e:  # pylint: disable=broad-except
      print(
          'Failed to count model FLOPs with error %s, because the build() '
          'methods in keras layers were not called. This is probably because '
          'the model was not feed any input, e.g., the max train step already '
          'reached before this run.', e)
      return None
  return None