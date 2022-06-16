import tensorflow as tf

def parse_exmp(serialized_example):
    feats = tf.parse_single_example(serialized_example, features={'Ground_Truth': tf.FixedLenFeature([], tf.string),
                                                                  'lowQ_Img': tf.FixedLenFeature([], tf.string),
                                                                  'format': tf.FixedLenFeature([], tf.string),
                                                                  'height': tf.FixedLenFeature([], tf.int64),
                                                                  'width': tf.FixedLenFeature([], tf.int64)})

    gt_image = tf.image.decode_jpeg(feats['Ground_Truth'])
    lowQ_image = tf.image.decode_jpeg(feats['lowQ_Img'])
    return gt_image, lowQ_image
