import tensorflow as tf
from layers import *


def Discriminator(x, scope='dis', snorm=True, leaky=True, is_train=True, patch=True, reuse=None):
    with tf.variable_scope(scope, reuse=reuse):
        conv1 = conv2d_block(x, 3, 64, filter_size=4, stride=2, snorm=snorm, leaky=leaky, name='conv_block_1')
        conv2 = conv2d_bn_block(conv1, 64, 128, filter_size=4, stride=2, snorm=snorm, leaky=leaky, is_train=is_train,
                                name='conv_block_2')
        conv3 = conv2d_bn_block(conv2, 128, 256, filter_size=4, stride=2, snorm=snorm, leaky=leaky, is_train=is_train,
                                name='conv_block_3')
        conv4 = conv2d_bn_block(conv3, 256, 512, filter_size=4, stride=1, snorm=snorm, leaky=leaky, is_train=is_train,
                                name='conv_block_4')
        if patch:
            conv5 = conv2d_block(conv4, 512, 1, filter_size=4, stride=1, snorm=snorm, linear=True, name='conv_block_5')
            out = conv5
        else:
            avg = global_avg(conv4)
            fc1 = conv2d_bn_block(avg, 512, 512, filter_size=1, stride=1, snorm=snorm, leaky=leaky, is_train=is_train,
                                  name='fc1')
            fc2 = pwise_conv(fc1, 512, 1, name='fc2')
            out = flatten(fc2)
        return out


def Discriminator_edge(x, scope='dis', snorm=True, leaky=True, is_train=True, patch=True, reuse=None):
    with tf.variable_scope(scope, reuse=reuse):
        conv1 = conv2d_block(x, 1, 64, filter_size=4, stride=2, snorm=snorm, leaky=leaky, name='conv_block_1')
        conv2 = conv2d_bn_block(conv1, 64, 128, filter_size=4, stride=2, snorm=snorm, leaky=leaky, is_train=is_train,
                                name='conv_block_2')
        conv3 = conv2d_bn_block(conv2, 128, 256, filter_size=4, stride=2, snorm=snorm, leaky=leaky, is_train=is_train,
                                name='conv_block_3')
        conv4 = conv2d_bn_block(conv3, 256, 512, filter_size=4, stride=1, snorm=snorm, leaky=leaky, is_train=is_train,
                                name='conv_block_4')
        if patch:
            conv5 = conv2d_block(conv4, 512, 1, filter_size=4, stride=1, snorm=snorm, linear=True, name='conv_block_5')
            out = conv5
        else:
            avg = global_avg(conv4)
            fc1 = conv2d_bn_block(avg, 512, 512, filter_size=1, stride=1, snorm=snorm, leaky=leaky, is_train=is_train,
                                  name='fc1')
            fc2 = pwise_conv(fc1, 512, 1, name='fc2')
            out = flatten(fc2)
        return out


def Generator(x, scope='gen', width_multiplier=1, expansion_ratio=6, blocks=9, reuse=None):
    with tf.variable_scope(scope, reuse=reuse):
        conv1 = conv2d_block(x, 3, int(64 * width_multiplier), filter_size=7, stride=1, name='conv_block_1')
        conv2 = conv2d_block(conv1, int(64 * width_multiplier), int(128 * width_multiplier), filter_size=3, stride=2,
                             name='conv_block_2')
        conv3 = conv2d_block(conv2, int(128 * width_multiplier), int(256 * width_multiplier), filter_size=3, stride=2,
                             name='conv_block_3')

        block = res_block_with_attention(conv3, int(256 * width_multiplier), int(256 * width_multiplier),
                                         expansion_ratio=expansion_ratio, channel_multiplier=1,
                                         filter_size=3, stride=1, bias=True, shortcut=True, name=f'res_block_with_attention_4')

        for b in range(5, 5 + blocks - 1):
            block = res_block_with_attention(block, int(256 * width_multiplier), int(256 * width_multiplier),
                                             expansion_ratio=expansion_ratio, channel_multiplier=1,
                                             filter_size=3, stride=1, bias=True, shortcut=True, name=f'res_block_with_attention_{b}')

        deconv_f1 = deconv_block(block, int(256 * width_multiplier), int(128 * width_multiplier),
                                 tf.shape(conv2), filter_size=3, stride=2, name=f'deconv_block_{blocks + 4}')
        deconv_f2 = deconv_block(deconv_f1, int(128 * width_multiplier), int(64 * width_multiplier),
                                 tf.shape(conv1), filter_size=3, stride=2, name=f'deconv_block_{blocks + 5}')
        conv_f3 = conv2d_block(deconv_f2, int(64 * width_multiplier), 3, filter_size=7, stride=1, linear=True,
                               name=f'conv_block_{blocks + 6}')

        out = tanh(conv_f3) + x
        return out


def GeneratorConvLSTM(inputs, iteration=2, scope='gen', width_multiplier=1, expansion_ratio=6, blocks=9, reuse=None):
    with tf.variable_scope(scope, reuse=reuse):
        batch, row, col, cha = inputs.get_shape().as_list()
        x = inputs
        h = tf.zeros(shape=(batch, row // 4, col // 4, 256), name="h")
        c = tf.zeros(shape=(batch, row // 4, col // 4, 256), name="c")
        h2 = tf.zeros(shape=(batch, row // 4, col // 4, 256), name="h2")
        c2 = tf.zeros(shape=(batch, row // 4, col // 4, 256), name="c2")
        h3 = tf.zeros(shape=(batch, row // 4, col // 4, 256), name="h3")
        c3 = tf.zeros(shape=(batch, row // 4, col // 4, 256), name="c3")
        h4 = tf.zeros(shape=(batch, row // 4, col // 4, 256), name="h4")
        c4 = tf.zeros(shape=(batch, row // 4, col // 4, 256), name="c4")

        for _ in range(iteration):
            with tf.variable_scope(tf.get_variable_scope(), reuse=tf.AUTO_REUSE):
                x = tf.concat([inputs, x], axis=-1)
                conv1 = conv2d_block(x, 6, int(64 * width_multiplier), filter_size=7, stride=1, name='conv_block_1')
                conv2 = conv2d_block(conv1, int(64 * width_multiplier), int(128 * width_multiplier), filter_size=3, stride=2,
                                     name='conv_block_2')
                conv3 = conv2d_block(conv2, int(128 * width_multiplier), int(256 * width_multiplier), filter_size=3, stride=2,
                                     name='conv_block_3')

                # ConvLSTM
                x = tf.concat([conv3, h], axis=-1)
                i = conv2d_block(x, int(512 * width_multiplier), int(256 * width_multiplier), filter_size=3,
                                 stride=1, af="sigmoid", name="conv_i")
                f = conv2d_block(x, int(512 * width_multiplier), int(256 * width_multiplier), filter_size=3,
                                 stride=1, af="sigmoid", name="conv_f")
                g = conv2d_block(x, int(512 * width_multiplier), int(256 * width_multiplier), filter_size=3,
                                 stride=1, af="tanh", name="conv_g")
                o = conv2d_block(x, int(512 * width_multiplier), int(256 * width_multiplier), filter_size=3,
                                 stride=1, af="sigmoid", name="conv_o")
                c = f * c + i * g
                h = o * tanh(c)
                x = h

                block = res_block_with_attention(x, int(256 * width_multiplier), int(256 * width_multiplier),
                                                 expansion_ratio=expansion_ratio, channel_multiplier=1,
                                                 filter_size=3, stride=1, bias=True, shortcut=True,
                                                 name=f'res_block_with_attention_4')
                block = res_block_with_attention(block, int(256 * width_multiplier), int(256 * width_multiplier),
                                                 expansion_ratio=expansion_ratio, channel_multiplier=1,
                                                 filter_size=3, stride=1, bias=True, shortcut=True,
                                                 name=f'res_block_with_attention_5')
                block = res_block_with_attention(block, int(256 * width_multiplier), int(256 * width_multiplier),
                                                 expansion_ratio=expansion_ratio, channel_multiplier=1,
                                                 filter_size=3, stride=1, bias=True, shortcut=True,
                                                 name=f'res_block_with_attention_6')

                x2 = tf.concat([block, h2], axis=-1)
                i2 = conv2d_block(x2, int(512 * width_multiplier), int(256 * width_multiplier), filter_size=3,
                                 stride=1, af="sigmoid", name="conv_i2")
                f2 = conv2d_block(x2, int(512 * width_multiplier), int(256 * width_multiplier), filter_size=3,
                                 stride=1, af="sigmoid", name="conv_f2")
                g2 = conv2d_block(x2, int(512 * width_multiplier), int(256 * width_multiplier), filter_size=3,
                                 stride=1, af="tanh", name="conv_g2")
                o2 = conv2d_block(x2, int(512 * width_multiplier), int(256 * width_multiplier), filter_size=3,
                                 stride=1, af="sigmoid", name="conv_o2")
                c2 = f2 * c2 + i2 * g2
                h2 = o2 * tanh(c2)
                x2 = h2

                block = res_block_with_attention(x2, int(256 * width_multiplier), int(256 * width_multiplier),
                                                 expansion_ratio=expansion_ratio, channel_multiplier=1,
                                                 filter_size=3, stride=1, bias=True, shortcut=True,
                                                 name=f'res_block_with_attention_7')
                block = res_block_with_attention(block, int(256 * width_multiplier), int(256 * width_multiplier),
                                                 expansion_ratio=expansion_ratio, channel_multiplier=1,
                                                 filter_size=3, stride=1, bias=True, shortcut=True,
                                                 name=f'res_block_with_attention_8')
                block = res_block_with_attention(block, int(256 * width_multiplier), int(256 * width_multiplier),
                                                 expansion_ratio=expansion_ratio, channel_multiplier=1,
                                                 filter_size=3, stride=1, bias=True, shortcut=True,
                                                 name=f'res_block_with_attention_9')
                x3 = tf.concat([block, h3], axis=-1)
                i3 = conv2d_block(x3, int(512 * width_multiplier), int(256 * width_multiplier), filter_size=3,
                                  stride=1, af="sigmoid", name="conv_i3")
                f3 = conv2d_block(x3, int(512 * width_multiplier), int(256 * width_multiplier), filter_size=3,
                                  stride=1, af="sigmoid", name="conv_f3")
                g3 = conv2d_block(x3, int(512 * width_multiplier), int(256 * width_multiplier), filter_size=3,
                                  stride=1, af="tanh", name="conv_g3")
                o3 = conv2d_block(x3, int(512 * width_multiplier), int(256 * width_multiplier), filter_size=3,
                                  stride=1, af="sigmoid", name="conv_o3")
                c3 = f3 * c3 + i3 * g3
                h3 = o3 * tanh(c3)
                x3 = h3

                block = res_block_with_attention(x3, int(256 * width_multiplier), int(256 * width_multiplier),
                                                 expansion_ratio=expansion_ratio, channel_multiplier=1,
                                                 filter_size=3, stride=1, bias=True, shortcut=True,
                                                 name=f'res_block_with_attention_10')
                block = res_block_with_attention(block, int(256 * width_multiplier), int(256 * width_multiplier),
                                                 expansion_ratio=expansion_ratio, channel_multiplier=1,
                                                 filter_size=3, stride=1, bias=True, shortcut=True,
                                                 name=f'res_block_with_attention_11')
                block = res_block_with_attention(block, int(256 * width_multiplier), int(256 * width_multiplier),
                                                 expansion_ratio=expansion_ratio, channel_multiplier=1,
                                                 filter_size=3, stride=1, bias=True, shortcut=True,
                                                 name=f'res_block_with_attention_12')
                x4 = tf.concat([block, h4], axis=-1)
                i4 = conv2d_block(x4, int(512 * width_multiplier), int(256 * width_multiplier), filter_size=3,
                                  stride=1, af="sigmoid", name="conv_i4")
                f4 = conv2d_block(x4, int(512 * width_multiplier), int(256 * width_multiplier), filter_size=3,
                                  stride=1, af="sigmoid", name="conv_f4")
                g4 = conv2d_block(x4, int(512 * width_multiplier), int(256 * width_multiplier), filter_size=3,
                                  stride=1, af="tanh", name="conv_g4")
                o4 = conv2d_block(x4, int(512 * width_multiplier), int(256 * width_multiplier), filter_size=3,
                                  stride=1, af="sigmoid", name="conv_o4")
                c4 = f4 * c4 + i4 * g4
                h4 = o4 * tanh(c4)
                x4 = h4

                deconv_f1 = deconv_block(x4, int(256 * width_multiplier), int(128 * width_multiplier),
                                         tf.shape(conv2), filter_size=3, stride=2, name=f'deconv_block_{blocks + 4}')
                deconv_f2 = deconv_block(deconv_f1, int(128 * width_multiplier), int(64 * width_multiplier),
                                         tf.shape(conv1), filter_size=3, stride=2, name=f'deconv_block_{blocks + 5}')
                conv_f3 = conv2d_block(deconv_f2, int(64 * width_multiplier), 3, filter_size=7, stride=1, linear=True,
                                       name=f'conv_block_{blocks + 6}')
                x = tanh(conv_f3) + inputs
        return x
