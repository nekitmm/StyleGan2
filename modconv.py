import tensorflow as tf
import tensorflow.keras as K
import tensorflow.keras.layers as L

class ModConv2D(L.Layer):
    def __init__(self, filters: int, kernel_size: int, demodulate: bool, **kwargs):
        super(ModConv2D, self).__init__(**kwargs)
        self.filters = filters
        self.kernel_size = (kernel_size, kernel_size)
        self.channel_dim = 3
        self.demodulate = demodulate
        self.input_spec = [L.InputSpec(ndim = 4), L.InputSpec(ndim = 2)]
        
    def build(self, input_shape):
        self.kernel = self.add_weight(shape = self.kernel_size + (input_shape[0][self.channel_dim], self.filters),\
                                      initializer = 'glorot_uniform', name = 'modconv_kernel')
    
    def call(self, inputs):
        # to channels first
        x = tf.transpose(inputs[0], [0, 3, 1, 2])
        
        s = K.backend.expand_dims(inputs[1], axis = 1)
        s = K.backend.expand_dims(s, axis = 1)
        s = K.backend.expand_dims(s, axis = -1)
        
        w = K.backend.expand_dims(self.kernel, axis = 0) * (s + 1)
        
        if self.demodulate:
            w /= tf.math.sqrt(tf.math.reduce_sum(tf.math.square(w), axis=[1, 2, 3], keepdims = True) + 1e-8)
        
        # reshape/scale input
        x = tf.reshape(x, [1, -1, x.shape[2], x.shape[3]])
        w = tf.reshape(tf.transpose(w, [1, 2, 3, 0, 4]), [w.shape[1], w.shape[2], w.shape[3], -1])
        
        x = tf.nn.conv2d(x, w, strides = (1, 1), padding="SAME", data_format="NCHW")
        
        # reshape/scale output.
        x = tf.reshape(x, [-1, self.filters, x.shape[2], x.shape[3]])
        x = tf.transpose(x, [0, 2, 3, 1])
        
        return x