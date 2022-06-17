import os
import datetime
import argparse
import sys
import json
from pathlib import Path
sys.path.append(os.path.dirname(Path(__file__).parent.absolute()) + '/train_utils')

from utils import training_config_parser
from models import *
from data_loader import *
from layers import *
from utils import images_augmentation
from hednet import HEDnet


os.environ["CUDA_VISIBLE_DEVICES"] = "-1"


def train(model_name='model_name',
          datasets='',
          batch_size=38,
          main_iterations=630901,
          lr_main=0.0001,
          alpha_ssim=1.0,
          lamb_per=1.0,
          lamb_adv=0.0,
          critic_iter=1,
          width_multiplier=1,
          expansion_ratio=6,
          blocks=8,
          snorm=True,
          patch=True,
          conv_lstm_iteration=-1):
    with tf.Graph().as_default():
        # ==========================Load Dataset==========================#
        dataset = tf.data.TFRecordDataset(datasets)
        dataset = dataset.shuffle(buffer_size=2103)
        dataset = dataset.repeat()
        dataset = dataset.map(parse_exmp)
        dataset = dataset.batch(batch_size)
        iterator = dataset.make_one_shot_iterator()
        next_element = iterator.get_next()

        # ==========================input placeholder==========================#
        with tf.variable_scope('inputs'):
            gt_placeholder = tf.placeholder(dtype=tf.uint8, shape=[batch_size, 256, 256, 3], name='input_gt')
            lq_placeholder = tf.placeholder(dtype=tf.uint8, shape=[batch_size, 256, 256, 3], name='input_lq')
            lr_main_placeholder = tf.placeholder(dtype=tf.float32, name='lr_main')
            img_gt = tf.image.convert_image_dtype(gt_placeholder, tf.float32)
            img_lq = tf.image.convert_image_dtype(lq_placeholder, tf.float32)

        # ==========================RaGAN==========================#
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

        edge_gt = HEDnet()
        edge_gt.build(img_gt, scope="HEDnet")
        edge_en = HEDnet()
        edge_en.build(img_en, scope="HEDnet", reuse=True)
        g_edge_per_loss = lamb_per * tf.reduce_mean(tf.abs(edge_gt.edge_map - edge_en.edge_map))

        if lamb_adv > 0.0:
            d_real = Discriminator(img_gt, scope='dis', snorm=snorm, is_train=True, patch=patch)
            d_fake = Discriminator(img_en, scope='dis', snorm=snorm, is_train=True, patch=patch, reuse=True)
            real_logit = (d_real - tf.reduce_mean(d_fake))
            fake_logit = (d_fake - tf.reduce_mean(d_real))

            d_real_edge = Discriminator_edge(edge_gt.edge_map, scope='dis_edge', snorm=snorm, is_train=True, patch=patch)
            d_fake_edge = Discriminator_edge(edge_en.edge_map, scope='dis_edge', snorm=snorm, is_train=True, patch=patch, reuse=True)
            real_logit_edge = (d_real_edge - tf.reduce_mean(d_fake_edge))
            fake_logit_edge = (d_fake_edge - tf.reduce_mean(d_real_edge))

        # ==========================loss==========================#
        with tf.variable_scope('loss'):
            loss_record = []

            # base loss
            g_str_loss = 0.0
            if alpha_ssim > 0:
                g_str_loss += alpha_ssim * tf.reduce_mean(1. - tf.image.ssim(img_gt, img_en, max_val=1.0))

            g_loss = 0.0
            if lamb_per > 0.0:
                g_loss = g_str_loss + g_edge_per_loss

            loss_record.append(tf.summary.scalar('0_g_str_loss', g_str_loss))
            loss_record.append(tf.summary.scalar('9_g_edge_per_loss', g_edge_per_loss))

            if lamb_adv > 0.0:
                # generator loss
                g_fake_loss = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(labels=tf.ones_like(d_fake), logits=fake_logit))
                g_real_loss = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(labels=tf.zeros_like(d_real), logits=real_logit))
                g_fake_loss_edge = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(labels=tf.ones_like(d_fake_edge), logits=fake_logit_edge))
                g_real_loss_edge = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(labels=tf.zeros_like(d_real_edge), logits=real_logit_edge))
                g_adv_loss = (g_fake_loss + g_real_loss) + (g_fake_loss_edge + g_real_loss_edge)
                g_loss = g_loss + lamb_adv * g_adv_loss
                loss_record.append(tf.summary.scalar('2_g_adv_loss', g_adv_loss))
                
                # discriminator loss
                d_real_loss = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(labels=tf.ones_like(d_real), logits=real_logit))
                d_fake_loss = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(labels=tf.zeros_like(d_fake), logits=fake_logit))
                d_real_loss_edge = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(labels=tf.ones_like(d_real_edge), logits=real_logit_edge))
                d_fake_loss_edge = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(labels=tf.zeros_like(d_fake_edge), logits=fake_logit_edge))
                d_loss = d_real_loss + d_fake_loss + d_real_loss_edge + d_fake_loss_edge
                loss_record.append(tf.summary.scalar('3_d_loss', d_loss))

            # ==========================training metrics summary==========================#
            tr_ssim = tf.reduce_mean(tf.image.ssim(img_gt, img_en, max_val=1.0))
            tr_psnr = tf.reduce_mean(tf.image.psnr(img_gt, img_en, max_val=1.0))
            loss_record.append(tf.summary.scalar('7_tr_ssim', tr_ssim))
            loss_record.append(tf.summary.scalar('8_tr_psnr', tr_psnr))
            merged_record = tf.summary.merge(loss_record)

        # ==========================optimizer==========================#
        with tf.variable_scope('optimizer'):
            g_var = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, 'gen*')
            d_var = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, 'dis*')
            print([v.name for v in d_var])
            print([v.name for v in g_var])

            g_train = tf.train.AdamOptimizer(lr_main_placeholder, beta1=0.9, beta2=0.999).minimize(g_loss, var_list=g_var)
            if lamb_adv > 0.0:
                with tf.control_dependencies(tf.get_collection(tf.GraphKeys.UPDATE_OPS)):
                    d_train = tf.train.AdamOptimizer(lr_main_placeholder, beta1=0.9, beta2=0.999).minimize(d_loss, var_list=d_var)

        # ==========================output placeholder==========================#
        with tf.variable_scope('outputs'):
            gt = gt_placeholder
            in_img = lq_placeholder
            out_img = tf.image.convert_image_dtype(img_en, tf.uint8, saturate=True, name='output')
            merged_images = tf.summary.merge([tf.summary.image('LQ_images', in_img),
                                              tf.summary.image('EN_images', out_img),
                                              tf.summary.image('GroundTruth', gt)])

        # ==========================save_path==========================#
        board_dir = f"tensorboard/{model_name}_" + datetime.datetime.now().strftime("%Y%m%d_%H%M%S") + "/"
        checkpoint_dir = f"checkpoint/{model_name}_" + datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        if not os.path.isdir(checkpoint_dir):
            os.mkdir(checkpoint_dir)
        checkpoint_dir = checkpoint_dir + "/model.ckpt"

        global_list = tf.global_variables()
        bn_moving_vars = [v for v in global_list if 'moving_mean' in v.name]
        bn_moving_vars += [v for v in global_list if 'moving_variance' in v.name]
        saver = tf.train.Saver(max_to_keep=10)

        # ==========================Session==========================#
        with tf.Session() as sess:
            writer = tf.summary.FileWriter(board_dir, sess.graph)
            # ==========================Initialize==========================#
            init = tf.global_variables_initializer()
            sess.run(init)
            num_epochs = 0

            # ==========================adversarial training==========================#
            for step in range(main_iterations):
                print(f'train step no: {step}')
                if num_epochs != 0 and num_epochs < 151 and (step % (2103 * 5) == 0):
                    print(f'lr_main at {num_epochs} epochs: {lr_main}.')
                elif num_epochs > 150 and (step % (2103 * 5) == 0):
                    lr_main = (-6.66 * 1e-7 * num_epochs) + (1.999 * 1e-4)
                    print(f'lr_main at {num_epochs} epochs: {lr_main}.')

                if lamb_adv > 0.0:
                    for _ in range(critic_iter):
                        ground_truth_buffer, input_image_buffer = sess.run(fetches=next_element)
                        ground_truth_buffer, input_image_buffer = images_augmentation(ground_truth_buffer, input_image_buffer)
                        _ = sess.run(d_train, feed_dict={gt_placeholder: ground_truth_buffer,
                                                         lq_placeholder: input_image_buffer,
                                                         lr_main_placeholder: lr_main})

                ground_truth_buffer, input_image_buffer = sess.run(fetches=next_element)
                ground_truth_buffer, input_image_buffer = images_augmentation(ground_truth_buffer, input_image_buffer)
                
                _ = sess.run(g_train, feed_dict={gt_placeholder: ground_truth_buffer,
                                                 lq_placeholder: input_image_buffer,
                                                 lr_main_placeholder: lr_main})
                if step == 5:
                    print('Start Tensorboard with command line: tensorboard --logdir=./tensorboard')
                    print('Open Tensorboard with URL: http://localhost:6006/')
                
                # record training metrics
                if step % 50 == 0:
                    record = sess.run(merged_record, feed_dict={gt_placeholder: ground_truth_buffer,
                                                                lq_placeholder: input_image_buffer,
                                                                lr_main_placeholder: lr_main})
                    writer.add_summary(record, step)

                # record images
                if step % 500 == 0:
                    images= sess.run(merged_images, feed_dict={gt_placeholder: ground_truth_buffer,
                                                               lq_placeholder: input_image_buffer,
                                                               lr_main_placeholder: lr_main})
                    writer.add_summary(images, step)
                
                # save checkpoint
                if step != 0 and step % 10000 == 0:
                    saver.save(sess, checkpoint_dir, global_step=step, write_meta_graph=False)

                if step % 2103 == 0:
                    num_epochs = num_epochs + 1

                if step % 630900 == 0:
                    saver.save(sess, checkpoint_dir, global_step=step, write_meta_graph=False)
            writer.close()


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--cfg_path', type=str, required=True)
    parser.add_argument('--section', type=str, required=True)
    args = parser.parse_args()

    model_params = training_config_parser(args.cfg_path, args.section)
    print(f'model_params = ' + json.dumps(model_params))
    print('Start training !')

    train(model_name=model_params['model_name'],
          datasets=model_params['datasets'],
          batch_size=model_params['batch_size'],
          main_iterations=model_params['main_iterations'],
          lr_main=model_params['lr_main'],
          alpha_ssim=model_params['alpha_ssim'],
          lamb_per=model_params['lamb_per'],
          lamb_adv=model_params['lamb_adv'],
          critic_iter=model_params['critic_iter'],
          width_multiplier=model_params['width_multiplier'],
          expansion_ratio=model_params['expansion_ratio'],
          blocks=model_params['blocks'],
          snorm=model_params['snorm'],
          patch=model_params['patch'],
          conv_lstm_iteration=model_params['conv_lstm_iteration'])
    print('Finish training !')
