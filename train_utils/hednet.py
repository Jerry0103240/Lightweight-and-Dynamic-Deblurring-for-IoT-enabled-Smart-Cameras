import numpy as np
import tensorflow as tf
import time


VGG_MEAN = [103.939, 116.779, 123.68]

def BilinearUpSample(x, shape):
    """
    Deterministic bilinearly-upsample the input images.
    It is implemented by deconvolution with "BilinearFiller" in Caffe.
    It is aimed to mimic caffe behavior.
    Args:
        x (tf.Tensor): a NHWC tensor
        shape (int): the upsample factor
    Returns:
        tf.Tensor: a NHWC tensor.
    """
    #log_deprecated("BilinearUpsample", "Please implement it in your own code instead!", "2019-03-01")
    inp_shape = x.shape.as_list()
    ch = inp_shape[3]
    assert ch is not None

    shape = int(shape)
    filter_shape = 2 * shape

    def bilinear_conv_filler(s):
        """
        s: width, height of the conv filter
        https://github.com/BVLC/caffe/blob/99bd99795dcdf0b1d3086a8d67ab1782a8a08383/include/caffe/filler.hpp#L219-L268
        """
        f = np.ceil(float(s) / 2)
        c = float(2 * f - 1 - f % 2) / (2 * f)
        ret = np.zeros((s, s), dtype='float32')
        for x in range(s):
            for y in range(s):
                ret[x, y] = (1 - abs(x / f - c)) * (1 - abs(y / f - c))
        return ret
    w = bilinear_conv_filler(filter_shape)
    w = np.repeat(w, ch * 1).reshape((filter_shape, filter_shape, ch, 1))

    weight_var = tf.constant(w, tf.float32,
                             shape=(filter_shape, filter_shape, ch, 1),
                             name='bilinear_upsample_filter')
    x = tf.pad(x, [[0, 0], [shape - 1, shape - 1], [shape - 1, shape - 1], [0, 0]], mode='SYMMETRIC')
    out_shape = tf.shape(x) * tf.constant([1, shape, shape, 1], tf.int32)

    @tf.custom_gradient
    def depthwise_deconv(x):
        ret = tf.nn.depthwise_conv2d_native_backprop_input(
            out_shape, weight_var, x, [1, shape, shape, 1], padding='SAME')
        def grad(dy):
            return tf.nn.depthwise_conv2d(dy, weight_var, [1, shape, shape, 1], padding='SAME')
        return ret, grad

    deconv = depthwise_deconv(x)

    edge = shape * (shape - 1)
    deconv = deconv[:, edge:-edge, edge:-edge, :]

    if inp_shape[1]:
        inp_shape[1] *= shape
    if inp_shape[2]:
        inp_shape[2] *= shape
    deconv.set_shape(inp_shape)
    return deconv

class HEDnet:
    def __init__(self):
        self.data_dict = np.load("HED_reproduced.npz")
        print("npz file loaded")

    def branch(self, name, l, up):
        filt = self.get_conv_filter(name + "/convfc")
        bias = self.get_bias(name + "/convfc")
        l = tf.nn.conv2d(input=l, filter=filt, strides=[1, 1, 1, 1], padding="SAME")
        l = tf.nn.bias_add(l, bias)
        l = tf.identity(l)
        while up != 1:
            l = BilinearUpSample(l, 2)
            up = up // 2
        return l

    def build(self, rgb, scope="HEDnet", reuse=False):
        """
        load variable from npy to build the VGG
        :param rgb: rgb image [batch, height, width, 3] values scaled [0, 1]
        """
        with tf.variable_scope(scope, reuse=reuse):
            start_time = time.time()
            print("build model started")
            rgb_scaled = rgb * 255.0

            # Convert RGB to BGR
            red, green, blue = tf.split(axis=3, num_or_size_splits=3, value=rgb_scaled)

            bgr = tf.concat(axis=3, values=[
                blue - VGG_MEAN[0],
                green - VGG_MEAN[1],
                red - VGG_MEAN[2],
            ])

            _, self.conv1_1 = self.conv_layer(bgr, "conv1_1")
            target, self.conv1_2 = self.conv_layer(self.conv1_1, "conv1_2")
            b1 = self.branch(name='branch1', l=self.conv1_2, up=1)
            self.pool1 = self.max_pool(self.conv1_2, 'pool1')

            self.conv2_1 = self.conv_layer(self.pool1, "conv2_1")
            self.conv2_2 = self.conv_layer(self.conv2_1, "conv2_2")
            b2 = self.branch('branch2', self.conv2_2, 2)
            self.pool2 = self.max_pool(self.conv2_2, 'pool2')
            
            self.conv3_1 = self.conv_layer(self.pool2, "conv3_1")
            self.conv3_2 = self.conv_layer(self.conv3_1, "conv3_2")
            self.conv3_3 = self.conv_layer(self.conv3_2, "conv3_3")
            b3 = self.branch('branch3', self.conv3_3, 4)
            self.pool3 = self.max_pool(self.conv3_3, 'pool3')
            
            self.conv4_1 = self.conv_layer(self.pool3, "conv4_1")
            self.conv4_2 = self.conv_layer(self.conv4_1, "conv4_2")
            self.conv4_3 = self.conv_layer(self.conv4_2, "conv4_3")
            b4 = self.branch('branch4', self.conv4_3, 8)
            self.pool4 = self.max_pool(self.conv4_3, 'pool4')
            
            self.conv5_1 = self.conv_layer(self.pool4, "conv5_1")
            self.conv5_2 = self.conv_layer(self.conv5_1, "conv5_2")
            self.conv5_3 = self.conv_layer(self.conv5_2, "conv5_3")
            b5 = self.branch('branch5', self.conv5_3, 16)
            
            fused_map = self.conv_layer(bottom=tf.concat([b1, b2, b3, b4, b5], -1), name="convfcweight")
            self.edge_map = target

            self.data_dict = None
            print(("build model finished: %ds" % (time.time() - start_time)))

    def max_pool(self, bottom, name):
        return tf.nn.max_pool(bottom, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME', name=name)

    def conv_layer(self, bottom, name):
        with tf.variable_scope(name):
            filt = self.get_conv_filter(name)
            conv = tf.nn.conv2d(bottom, filt, [1, 1, 1, 1], padding='SAME')
            if name != "convfcweight":
                conv_biases = self.get_bias(name)
                bias = tf.nn.bias_add(conv, conv_biases)
            else:
                bias = tf.identity(conv)
                return bias
            relu = tf.nn.relu(bias)
            return bias, relu

    def get_conv_filter(self, name):
        filt = tf.constant(self.data_dict[name + "/W:0"], dtype=tf.float32)
        return filt

    def get_bias(self, name):
        bias = tf.constant(self.data_dict[name + "/b:0"], dtype=tf.float32)
        return bias
