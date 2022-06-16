import tensorflow as tf
import os

os.environ["CUDA_VISIBLE_DEVICES"] = "0"


def _int64_feature(value):
    if not isinstance(value, list):
        value = [value]
    return tf.train.Feature(int64_list=tf.train.Int64List(value=value))


def _floats_feature(value):
    if not isinstance(value, list):
        value = [value]
    return tf.train.Feature(float_list=tf.train.FloatList(value=value))


def _bytes_feature(value):
    return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))


def image_files_list(blur_path, sharp_path):
    blur_image_list = [os.path.join(blur_path, file) for file in os.listdir(blur_path)]
    sharp_image_list = [os.path.join(sharp_path, file) for file in os.listdir(sharp_path)]
    return blur_image_list, sharp_image_list


def convert_dataset_syn(targetfile, blur_path, sharp_path, output_format='png'):
    blur_image_list, sharp_image_list = image_files_list(blur_path, sharp_path)
    images = zip(sharp_image_list, blur_image_list)

    writer = tf.python_io.TFRecordWriter(targetfile)
    with tf.Graph().as_default():
        gt_placeholder = tf.placeholder(dtype=tf.string)
        lowQ_placeholder = tf.placeholder(dtype=tf.string)
        ground_truth = tf.image.decode_png(gt_placeholder)
        lowQ_image = tf.image.decode_png(lowQ_placeholder)
        gt_encoded = tf.image.encode_png(ground_truth)
        lowQ_encoded = tf.image.encode_png(lowQ_image)

        with tf.Session() as sess:
            for img in images:
                with tf.gfile.FastGFile(img[0], 'rb') as f:
                    gt_data = f.read()
                with tf.gfile.FastGFile(img[1], 'rb') as f:
                    lowQ_data = f.read()
                try:
                    gt, lowQ, lowQ_img = sess.run([gt_encoded, lowQ_encoded, lowQ_image],
                                                  feed_dict={gt_placeholder: gt_data,
                                                             lowQ_placeholder: lowQ_data})
                except:
                    continue

                assert len(lowQ_img.shape) == 3
                height = lowQ_img.shape[0]
                width = lowQ_img.shape[1]
                if lowQ_img.shape[2] != 3:
                    continue
                assert lowQ_img.shape[2] == 3

                ex = tf.train.Example(features=tf.train.Features(feature={
                    'Ground_Truth': _bytes_feature(tf.compat.as_bytes(gt)),
                    'lowQ_Img': _bytes_feature(tf.compat.as_bytes(lowQ)),
                    'format': _bytes_feature(tf.compat.as_bytes(output_format)),
                    'height': _int64_feature(height),
                    'width': _int64_feature(width)}))

                writer.write(ex.SerializeToString())
    writer.close()


if __name__ == '__main__':
    target_file = "Training_PNG_crop.tfrecords"
    blur_images_path = 'blur_images_path'
    sharp_images_path = 'sharp_images_path'
    convert_dataset_syn(targetfile=target_file,
                        blur_path=blur_images_path,
                        sharp_path=sharp_images_path,
                        output_format="png")
