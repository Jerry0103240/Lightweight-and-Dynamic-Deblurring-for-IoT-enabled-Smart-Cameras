import tensorflow as tf
import numpy as np
import math
'''
Utility layers
'''
def upsampling(x, size, name='upsample'):
    return tf.image.resize_nearest_neighbor(x, size=size, name=name)

def pixel_shuffle(x, block_size=2, name='pixel_shuffle'):
    return tf.depth_to_space(x, block_size, name=name)

def relu(x, leaky=False):
    if leaky:
        return tf.nn.leaky_relu(x, name='leaky_relu')
    else:
        return tf.nn.relu(x, name='relu')

def relu6(x, leaky=False):
    return tf.nn.relu6(x, name='relu6')

def batch_norm(x, momentum=0.9, epsilon=1e-5, training=False, name='bn'):
    return tf.layers.batch_normalization(x, momentum=momentum, epsilon=epsilon, scale=True, training=training, name=name)

def hard_sigmoid(x, name='hard_sigmoid'):
    with tf.variable_scope(name):
        return tf.keras.activations.hard_sigmoid(x)

def tanh(x, name='tanh'):
    return tf.nn.tanh(x, name=name)

def global_avg(x):
    with tf.name_scope('global_avg'):
        net=tf.layers.average_pooling2d(x, x.get_shape()[1:-1], 1)
        return net

def flatten(x):
    #flattened=tf.reshape(input,[x.get_shape().as_list()[0], -1])  # or, tf.layers.flatten(x)
    return tf.contrib.layers.flatten(x)

def pad2d(inputs, pad=(0, 0), mode='CONSTANT'):
    paddings = [[0, 0], [pad[0], pad[0]], [pad[1], pad[1]], [0, 0]]
    net = tf.pad(inputs, paddings, mode=mode)
    return net

def spectral_norm(w, iteration=1):
    w_shape = w.shape.as_list()
    w = tf.reshape(w, [-1, w_shape[-1]])

    u = tf.get_variable('u', [1, w_shape[-1]], initializer=tf.random_normal_initializer(), trainable=False)

    u_hat = u
    v_hat = None
    for _ in range(iteration):

        """
        power iteration
        Usually iteration = 1 will be enough
        """

        v_ = tf.matmul(u_hat, tf.transpose(w))
        v_hat = tf.nn.l2_normalize(v_)

        u_ = tf.matmul(v_hat, w)
        u_hat = tf.nn.l2_normalize(u_)

    u_hat = tf.stop_gradient(u_hat)
    v_hat = tf.stop_gradient(v_hat)

    sigma = tf.matmul(tf.matmul(v_hat, w), tf.transpose(u_hat))

    with tf.control_dependencies([u.assign(u_hat)]):
        w_norm = w / sigma
        w_norm = tf.reshape(w_norm, w_shape)


    return w_norm
'''
End of Utility layers
'''

'''
Convulution Layers
'''
def conv2d(x, filter_shape, stride=1, bias=True, padding='SAME', snorm=False, name='conv2d'):
    with tf.variable_scope(name):
        w = tf.get_variable('weight', shape=filter_shape, initializer=tf.contrib.layers.xavier_initializer())
        if snorm:
            w = spectral_norm(w)

        x = tf.nn.conv2d(x, w, [1, stride, stride, 1], padding=padding)
        if bias:
            b = tf.get_variable('bias', shape=filter_shape[-1], initializer=tf.constant_initializer(0.1))
            x = tf.nn.bias_add(x, b)
            
        return x

def deconv2d(x, filter_shape, output_shape, stride=1, bias=True, padding='SAME', name='deconv2d'):
    with tf.variable_scope(name):
        w = tf.get_variable('weight', shape=filter_shape, initializer=tf.contrib.layers.xavier_initializer())
        x = tf.nn.conv2d_transpose(x, w, output_shape, [1, stride, stride, 1], padding=padding)

        if bias:
            b = tf.get_variable('bias', shape=filter_shape[-2], initializer=tf.constant_initializer(0.1))
            x = tf.nn.bias_add(x, b)

        return x

def dwise_conv(x, input_dim, channel_multiplier=1, filter_size=3, stride=1, bias=True, padding='SAME', name='dw_conv'):
    with tf.variable_scope(name):
        w = tf.get_variable('weight', shape=[filter_size, filter_size, input_dim, channel_multiplier],
                            initializer=tf.contrib.layers.xavier_initializer())
        x = tf.nn.depthwise_conv2d(x, w,  [1, stride, stride, 1], padding = padding)

        if bias:
            b = tf.get_variable('bias', shape=[input_dim*channel_multiplier], initializer=tf.constant_initializer(0.1))
            x = tf.nn.bias_add(x, b)

        return x

def pwise_conv(x, input_dim, output_dim, bias=True, name='pw_conv'):
    return conv2d(x, [1,1,input_dim,output_dim], stride=1, bias=bias, padding='SAME', name=name)

def separable_conv(x, input_dim, output_dim, filter_size=3, channel_multiplier=1, stride=1, bias=True, padding='SAME', name='sp_conv'):
    with tf.variable_scope(name):
        dwise_filter = tf.get_variable('dw', shape=[filter_size, filter_size, input_dim, channel_multiplier],
                                       initializer=tf.contrib.layers.xavier_initializer())

        pwise_filter = tf.get_variable('pw', [1, 1, input_dim*channel_multiplier, output_dim],
                                       initializer=tf.contrib.layers.xavier_initializer())  
        x = tf.nn.separable_conv2d(x,dwise_filter,pwise_filter,[1,stride, stride,1],padding=padding, name=name)
        if bias:
            b = tf.get_variable('bias', [output_dim], initializer=tf.constant_initializer(0.1))
            x = tf.nn.bias_add(x, b)
        return x
'''
End of Convulutions Layers
'''
'''
Blocks
'''
def conv2d_block(x, input_dim, output_dim, filter_size=3, stride=1, bias=True, padding='SAME',
                 snorm=False, leaky=False, linear=False, name='conv_block'):
    with tf.variable_scope(name):
        net = conv2d(x, [filter_size, filter_size, input_dim, output_dim], stride=stride, bias=bias, padding=padding,
                     snorm=snorm, name='conv')
        if linear:
            out = net
        else:
            out = relu(net, leaky)
        return out
    
def deconv_block(x, input_dim, output_dim, output_shape, filter_size=3, stride=1, bias=True, padding='SAME',
                 leaky=False, linear=False, name='deconv_block'):
    with tf.variable_scope(name):
        net = deconv2d(x, [filter_size, filter_size, output_dim, input_dim], output_shape,
                       stride=stride, bias=bias, padding=padding, name='deconv')
        if linear:
            out = net
        else:
            out = relu(net, leaky)
        return out

def conv2d_bn_block(x, input_dim, output_dim, is_train, filter_size=3, stride=1, bias=False, padding='SAME',
                    snorm=False, leaky=False, name='conv_block'):
    with tf.variable_scope(name):
        net = conv2d(x, [filter_size, filter_size, input_dim, output_dim], stride=stride, bias=bias, padding=padding,
                     snorm=snorm, name='conv')
        net = batch_norm(net, training=is_train, name='bn')
        out = relu(net, leaky)
        return out

def dwise_block(x, input_dim, channel_multiplier, filter_size=3, stride=1, bias=True, leaky=False, name='dw_block'):
    with tf.variable_scope(name):
        net = dwise_conv(x, input_dim, channel_multiplier, filter_size=filter_size, stride=stride, bias=bias, name='dw')
        out = relu(net, leaky)
        return out
    
def pwise_block(x, input_dim, output_dim, bias=True, leaky=False, name='pw_block'):
    with tf.variable_scope(name):
        net = pwise_conv(x, input_dim, output_dim, bias=bias, name='pw')
        out = relu(net, leaky)
        return out
    
def separable_block(x, input_dim, output_dim, filter_size=3, channel_multiplier=1, stride=1, bias=True, leaky=False, name='sp_block'):
    with tf.variable_scope(name):
        net = dwise_conv(x, input_dim, channel_multiplier, filter_size=filter_size, stride=stride, bias=bias, name='dw')
        net = relu(net, leaky)
        net = pwise_conv(net, input_dim*channel_multiplier, output_dim, bias=bias, name='pw')
        net = relu(net, leaky)
        out = net
        return out
    
def res_block(x, input_dim, output_dim, expansion_ratio, channel_multiplier=1,
              filter_size=3, stride=1, bias=True, leaky=False, shortcut=True, name='res_block'):
    with tf.variable_scope(name):
        # pw
        bottleneck_dim=round(expansion_ratio*input_dim)
        net = pwise_conv(x, input_dim, bottleneck_dim, bias=bias, name='pw')
        net = relu(net, leaky)
        # dw
        net = dwise_conv(net, bottleneck_dim, channel_multiplier, filter_size=filter_size, stride=stride, bias=bias, name='dw')
        net = relu(net, leaky)
        # pw & linear
        net = pwise_conv(net, bottleneck_dim*channel_multiplier, output_dim, bias=bias, name='pw_linear')
        
        # element wise add, only for stride==1
        if shortcut and stride == 1:
            in_dim=int(input_dim)
            if in_dim != output_dim:
                ins=pwise_conv(x,input_dim, output_dim, bias=bias, name='ex_dim')
                net=ins+net
            else:
                net=x+net
        return net


def res_block_with_attention(x, input_dim, output_dim, expansion_ratio, channel_multiplier=1,
                             filter_size=3, stride=1, bias=True, leaky=False, shortcut=True, name='res_block_with_attention'):
    with tf.variable_scope(name):
        # pw
        bottleneck_dim = round(expansion_ratio * input_dim)
        net = pwise_conv(x, input_dim, bottleneck_dim, bias=bias, name='pw')
        net = relu(net, leaky)
        # dw
        net = dwise_conv(net, bottleneck_dim, channel_multiplier, filter_size=filter_size, stride=stride, bias=bias,
                         name='dw')
        net = relu(net, leaky)
        # pw & linear
        net = pwise_conv(net, bottleneck_dim * channel_multiplier, output_dim, bias=bias, name='pw_linear')

        channel = net.get_shape().as_list()[-1]

        # spatial attention
        spatial_a = dwise_conv(net, channel, channel_multiplier, filter_size=7, stride=1, bias=True,
                               name='spatial_dw')
        spatial_a = hard_sigmoid(spatial_a)
        net = net * spatial_a

        # element wise add, only for stride==1
        if shortcut and stride == 1:
            in_dim = int(input_dim)
            if in_dim != output_dim:
                ins = pwise_conv(x, input_dim, output_dim, bias=bias, name='ex_dim')
                net = ins + net
            else:
                net = x + net
        return net

def conv2d_bn_6_block(x, input_dim, output_dim, is_train, filter_size=3, stride=1, bias=False, padding='SAME', name='conv_block'):
    with tf.variable_scope(name):
        net = conv2d(x, [filter_size, filter_size, input_dim, output_dim], stride=stride, bias=bias, padding=padding, name='conv')
        net = batch_norm(net, training=is_train, name='bn')
        out = relu6(net)
        return out

def pwise_bn_6_block(x, input_dim, output_dim, is_train, bias=False, name='pw_block'):
    with tf.variable_scope(name):
        net = pwise_conv(x, input_dim, output_dim, bias=bias, name='pw')
        net=batch_norm(net, training=is_train, name='bn')
        out = relu6(net)
        return out
    
def res_bn_6_block(x, input_dim, output_dim, expansion_ratio, is_train, channel_multiplier=1,
              filter_size=3, stride=1, bias=False, shortcut=True, name='res_block'):
    with tf.variable_scope(name):
        # pw
        bottleneck_dim=round(expansion_ratio*input_dim)
        net = pwise_conv(x, input_dim, bottleneck_dim, bias=bias, name='pw')
        net = batch_norm(net, training=is_train, name='pw_bn')
        net = relu6(net)
        # dw
        net = dwise_conv(net, bottleneck_dim, channel_multiplier, filter_size=filter_size, stride=stride, bias=bias, name='dw')
        net = batch_norm(net, training=is_train, name='dw_bn')
        net = relu6(net)
        # pw & linear
        net = pwise_conv(net, bottleneck_dim*channel_multiplier, output_dim, bias=bias, name='pw_linear')
        net = batch_norm(net, training=is_train, name='pw_linear_bn')
        
        # element wise add, only for stride==1
        if shortcut and stride == 1:
            in_dim=int(input_dim)
            if in_dim != output_dim:
                ins=pwise_conv(x,input_dim, output_dim, bias=bias, name='ex_dim')
                net=ins+net
            else:
                net=x+net
        return net 
'''
End of Blocks
'''
