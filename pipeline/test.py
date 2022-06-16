import tensorflow as tf
from train_utils.data_loader import *
from train_utils.models import *
import time


def test(checkpoint='',
         data_path='',
         width_multiplier=1,
         expansion_ratio=6,
         blocks=8):
    with tf.Graph().as_default():
        # ==========================Dataset==========================#
        dataset = tf.data.TFRecordDataset(data_path)
        dataset = dataset.map(parse_exmp)
        dataset = dataset.batch(1)
        iterator = dataset.make_one_shot_iterator()
        next_element = iterator.get_next()

        # ==========================input placeholder==========================#
        with tf.variable_scope('inputs'):
            lq_placeholder = tf.placeholder(dtype=tf.uint8, name='input_lq')
            gt_placeholder = tf.placeholder(dtype=tf.uint8, name='input_gt')
            img_lq = tf.image.convert_image_dtype(lq_placeholder, tf.float32)
            img_gt = tf.image.convert_image_dtype(gt_placeholder, tf.float32)

        # ==========================network==========================#
        img_en = Generator(img_lq,
                           scope='gen',
                           width_multiplier=width_multiplier,
                           expansion_ratio=expansion_ratio,
                           blocks=blocks)

        # ==========================output placeholder==========================#
        with tf.variable_scope('outputs'):
            out_img = tf.image.convert_image_dtype(img_en, tf.uint8, saturate=True, name='output')
            out_psnr = tf.image.psnr(img_en, img_gt, max_val=1.0)
            out_ssim = tf.image.ssim(img_en, img_gt, max_val=1.0)
        # ==========================Restore==========================#
        saver = tf.train.Saver()
        with tf.Session() as sess:
            saver.restore(sess, checkpoint)
            times, psnrs, ssims = [], [], []

            while True:
                try:
                    ground_truth_buffer, input_image_buffer = sess.run(fetches=next_element)
                    feed_dict = {gt_placeholder: ground_truth_buffer, lq_placeholder: input_image_buffer}

                    try:
                        start_time = time.time()
                        image = sess.run(out_img, feed_dict)
                        times.append(time.time() - start_time)
                        psnr, ssim = sess.run([out_psnr, out_ssim], feed_dict)
                        psnrs.append(psnr[0])
                        ssims.append(ssim[0])
                    except:
                        continue
                except tf.errors.OutOfRangeError:
                    break

        print(f'Average time per image: {sum(times) / len(times)}')
        print(f'Average PSNR: {sum(psnrs) / len(psnrs)}')
        print(f'Average SSIM: {sum(ssims) / len(ssims)}')


if __name__ == '__main__':
    data = "Directory of testing dataset with tfrecord type"
    test(checkpoint='model.ckpt',
         data_path=data)
