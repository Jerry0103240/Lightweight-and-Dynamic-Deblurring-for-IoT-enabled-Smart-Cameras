import time
import tensorflow as tf
import argparse
import sys
import os
import numpy as np
from pathlib import Path
sys.path.append(os.path.dirname(Path(__file__).parent.absolute()) + '/train_utils')

from data_loader import *
from models import *
from utils import testing_config_parser
from PIL import Image

os.environ["CUDA_VISIBLE_DEVICES"] = "-1"

def test(checkpoint='',
         data_path='',
         width_multiplier=1,
         expansion_ratio=6,
         blocks=8,
         conv_lstm_iteration=-1,
         to_save=False,
         saving_path=''):
    with tf.Graph().as_default():
        # ==========================Dataset==========================#
        dataset = tf.data.TFRecordDataset(data_path)
        dataset = dataset.map(parse_exmp)
        dataset = dataset.batch(1)
        iterator = dataset.make_one_shot_iterator()
        next_element = iterator.get_next()

        # ==========================input placeholder==========================#
        with tf.variable_scope('inputs'):
            lq_placeholder = tf.placeholder(shape=[1, 720, 1280, 3], dtype=tf.uint8, name='input_lq')
            gt_placeholder = tf.placeholder(shape=[1, 720, 1280, 3], dtype=tf.uint8, name='input_gt')
            img_lq = tf.image.convert_image_dtype(lq_placeholder, tf.float32)
            img_gt = tf.image.convert_image_dtype(gt_placeholder, tf.float32)

        # ==========================network==========================#
        if conv_lstm_iteration > 0:
            img_en = GeneratorConvLSTM(img_lq,
                                       iteration=conv_lstm_iteration,
                                       scope='gen',
                                       width_multiplier=width_multiplier,
                                       expansion_ratio=expansion_ratio,
                                       blocks=blocks)
        else:
            img_en = Generator(img_lq,
                               scope='gen', 
                               width_multiplier=width_multiplier,
                               expansion_ratio=expansion_ratio,
                               blocks=blocks)
        # ==========================output placeholder==========================#
        with tf.variable_scope('outputs'):
            out_img_lq = tf.image.convert_image_dtype(img_lq, tf.uint8, saturate=True, name='output')
            out_img_en = tf.image.convert_image_dtype(img_en, tf.uint8, saturate=True, name='output')
            out_psnr = tf.image.psnr(img_en, img_gt, max_val=1.0)
            out_ssim = tf.image.ssim(img_en, img_gt, max_val=1.0)
        # ==========================Restore==========================#
        saver = tf.train.Saver()
        with tf.Session() as sess:
            saver.restore(sess, checkpoint)
            times, psnrs, ssims = [], [], []
            step = 0

            while True:
                try:
                    ground_truth_buffer, input_image_buffer = sess.run(fetches=next_element)
                    feed_dict = {gt_placeholder: ground_truth_buffer, lq_placeholder: input_image_buffer}

                    print(f'test step no: {step}')
                    start_time = time.time()
                    image_lq = sess.run(out_img_lq, feed_dict)
                    image_en = sess.run(out_img_en, feed_dict)

                    if to_save:
                        save_image(image_lq[0], image_en[0], step, saving_path)

                    times.append(time.time() - start_time)
                    psnr, ssim = sess.run([out_psnr, out_ssim], feed_dict)
                    psnrs.append(psnr[0])
                    ssims.append(ssim[0])
                    step += 1

                except tf.errors.OutOfRangeError:
                    break

        print(f'Average time per image: {sum(times) / len(times)}')
        print(f'Average PSNR: {sum(psnrs) / len(psnrs)}')
        print(f'Average SSIM: {sum(ssims) / len(ssims)}')


def save_image(image_lq, image_en, num, path):
    image_lq = Image.fromarray(image_lq)
    image_lq.save(path + f'{num}_blur.png')

    image_en = Image.fromarray(image_en)
    image_en.save(path + f'{num}_deblur.png')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--cfg_path', type=str, required=True)
    parser.add_argument('--section', type=str, required=True)
    args = parser.parse_args()
    testing_params = testing_config_parser(args.cfg_path, args.section)

    test(checkpoint=testing_params['testing_ckpt_path'],
         data_path=testing_params['testing_datasets'],
         width_multiplier=testing_params['model_params']['width_multiplier'],
         expansion_ratio=testing_params['model_params']['expansion_ratio'],
         blocks=testing_params['model_params']['blocks'],
         conv_lstm_iteration=testing_params['model_params']['conv_lstm_iteration'],
         to_save=testing_params['to_save'],
         saving_path=testing_params['saving_path'])
