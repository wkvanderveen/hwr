#! /usr/bin/env python
# coding=utf-8
#================================================================
#   Copyright (C) 2019 * Ltd. All rights reserved.
#
#   Editor      : VIM
#   File name   : quick_train.py
#   Author      : YunYang1994
#   Created date: 2019-01-21 14:46:26
#   Description :
#
#================================================================

import tensorflow as tf
import os
from core import utils, yolov3
from core.dataset import dataset, Parser
sess = tf.Session()

class Trainer(object):
    """docstring for Trainer"""
    def __init__(self, num_classes, batch_size, n_filters_dn, cell_size, n_filt_yolo, ignore_threshold, steps, img_dims, learning_rate, decay_steps, decay_rate, shuffle_size, eval_internal, save_internal, anchors_path, train_records, test_records, checkpoint_path):
        super(Trainer, self).__init__()
        self.num_classes = num_classes
        self.batch_size = batch_size
        self.steps = steps
        self.learning_rate = learning_rate
        self.decay_steps = decay_steps
        self.decay_rate = decay_rate
        self.ignore_thresh = ignore_threshold
        self.shuffle_size = shuffle_size
        self.eval_internal = eval_internal
        self.save_internal = save_internal
        self.img_h = img_dims[0]
        self.img_w = img_dims[1]
        self.n_filt_yolo = n_filt_yolo
        self.n_filters_dn = n_filters_dn
        self.cell_size = cell_size
        self.anchors_path = anchors_path
        self.train_records = train_records
        self.test_records = test_records
        self.checkpoint_path = os.path.join(checkpoint_path, "yolov3.ckpt")



    def train(self):
        ANCHORS = utils.get_anchors(self.anchors_path, self.img_h, self.img_w)

        parser   = Parser(image_h=self.img_h,
                          image_w=self.img_w,
                          anchors=ANCHORS,
                          num_classes=self.num_classes,
                          cell_size=self.cell_size)

        trainset = dataset(parser, self.train_records, self.batch_size, shuffle=self.shuffle_size)
        testset  = dataset(parser, self.test_records , self.batch_size, shuffle=None)

        is_training = tf.placeholder(tf.bool)
        example = tf.cond(is_training, lambda: trainset.get_next(), lambda: testset.get_next())

        images, y_true = example

        model = yolov3.yolov3(self.num_classes, ANCHORS)

        with tf.variable_scope('yolov3'):
            pred_feature_map = model.forward(images,
                                             is_training=is_training,
                                             n_filters_dn=self.n_filters_dn,
                                             n_filt_yolo=self.n_filt_yolo)

            loss             = model.compute_loss(pred_feature_map, y_true, ignore_thresh=self.ignore_thresh)
            y_pred           = model.predict(pred_feature_map)

        tf.summary.scalar("loss/coord_loss",   loss[1])
        tf.summary.scalar("loss/sizes_loss",   loss[2])
        tf.summary.scalar("loss/confs_loss",   loss[3])
        tf.summary.scalar("loss/class_loss",   loss[4])

        global_step = tf.Variable(0, trainable=False, collections=[tf.GraphKeys.LOCAL_VARIABLES])
        write_op = tf.summary.merge_all()
        writer_train = tf.summary.FileWriter("../../data/train_summary", sess.graph)
        writer_test  = tf.summary.FileWriter("../../data/test_summary")

        #saver_to_restore = tf.train.Saver(var_list=tf.contrib.framework.get_variables_to_restore(include=["yolov3/darknet-53"]))
        update_vars = tf.contrib.framework.get_variables_to_restore(include=["yolov3/yolo-v3"])
        lr = tf.train.exponential_decay(self.learning_rate, global_step, decay_steps=self.decay_steps, decay_rate=self.decay_rate, staircase=True)
        optimizer = tf.train.AdamOptimizer(lr)

        # set dependencies for BN ops
        update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
        with tf.control_dependencies(update_ops):
            train_op = optimizer.minimize(loss[0], var_list=update_vars, global_step=global_step)

        sess.run([tf.global_variables_initializer(), tf.local_variables_initializer()])
        #saver_to_restore.restore(sess, self.checkpoint_path)
        saver = tf.train.Saver(max_to_keep=2)

        for step in range(self.steps):
            run_items = sess.run([train_op, write_op, y_pred, y_true] + loss, feed_dict={is_training:True})

            if (step+1) % self.eval_internal == 0:
                train_rec_value, train_prec_value = utils.evaluate(run_items[2], run_items[3])

            writer_train.add_summary(run_items[1], global_step=step)
            writer_train.flush() # Flushes the event file to disk
            if (step+1) % self.save_internal == 0: saver.save(sess, save_path=self.checkpoint_path, global_step=step+1)

            
            print("=> STEP %10d [TRAIN]:\tloss_xy:%7.4f \tloss_wh:%7.4f \tloss_conf:%7.4f \tloss_class:%7.4f"
                %(step+1, run_items[5], run_items[6], run_items[7], run_items[8]))
            

            run_items = sess.run([write_op, y_pred, y_true] + loss, feed_dict={is_training:False})
            if (step+1) % self.eval_internal == 0:
                test_rec_value, test_prec_value = utils.evaluate(run_items[1], run_items[2])
                print("\n=======================> evaluation result <================================\n")
                print("=> STEP %10d [TRAIN]:\trecall:%7.4f \tprecision:%7.4f" %(step+1, train_rec_value, train_prec_value))
                print("=> STEP %10d [VALID]:\trecall:%7.4f \tprecision:%7.4f" %(step+1, test_rec_value,  test_prec_value))
                print("\n=======================> evaluation result <================================\n")

            writer_test.add_summary(run_items[0], global_step=step)
            writer_test.flush() # Flushes the event file to disk


if __name__ == '__main__':
    trainer = Trainer()
    trainer.train()
