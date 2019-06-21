import tensorflow as tf
import os
from core import utils, yolov3
from core.dataset import dataset, Parser
tf.logging.set_verbosity(tf.logging.ERROR)
sess = tf.Session()
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'  # or any {'0', '1', '2'}


class Trainer(object):
    """docstring for Trainer"""
    def __init__(self, num_classes, batch_size, n_filters_dn, size_threshold,
                 n_strides_dn, n_ksizes_dn, steps,
                 img_dims, learning_rate, decay_steps, decay_rate,
                 shuffle_size, eval_internal, save_internal, print_every_n,
                 anchors_path, train_records, test_records, checkpoint_path):
        super(Trainer, self).__init__()
        self.num_classes = num_classes
        self.batch_size = batch_size
        self.steps = steps
        self.learning_rate = learning_rate
        self.decay_steps = decay_steps
        self.decay_rate = decay_rate
        self.shuffle_size = shuffle_size
        self.eval_internal = eval_internal
        self.size_threshold = size_threshold
        self.save_internal = save_internal
        self.iou_threshold = 1.0  # lower is stricter
        self.print_every_n = print_every_n
        self.img_h = img_dims[0]
        self.img_w = img_dims[1]
        self.n_filters_dn = n_filters_dn
        self.n_strides_dn = n_strides_dn
        self.n_ksizes_dn = n_ksizes_dn
        self.anchors_path = anchors_path
        self.train_records = train_records
        self.test_records = test_records
        self.checkpoint_path = os.path.join(checkpoint_path, "yolov3.ckpt")

        def deprecated(date, instructions, warn_once=False):
            def deprecated_wrapper(func):
                return func
            return deprecated_wrapper

        from tensorflow.python.util import deprecation
        deprecation.deprecated = deprecated

    def train(self):
        ANCHORS = utils.get_anchors(self.anchors_path, self.img_h, self.img_w)

        parser = Parser(image_h=self.img_h,
                        image_w=self.img_w,
                        anchors=ANCHORS,
                        num_classes=self.num_classes)

        trainset = dataset(parser,
                           self.train_records,
                           self.batch_size,
                           shuffle=self.shuffle_size)
        testset = dataset(parser,
                          self.test_records,
                          self.batch_size,
                          shuffle=None)

        is_training = tf.placeholder(tf.bool)

        example = tf.cond(is_training,
                          lambda: trainset.get_next(),
                          lambda: testset.get_next())

        images, y_true = example

        model = yolov3.yolov3(self.num_classes, ANCHORS)

        with tf.variable_scope('yolov3'):

            # Give the images to the network, and receive a prediction
            # feature map
            pred_feature_map = model.forward(
                images,
                is_training=is_training,
                n_filters_dn=self.n_filters_dn,
                n_strides_dn=self.n_strides_dn,
                n_ksizes_dn=self.n_ksizes_dn)

            loss = model.compute_loss(
                pred_feature_map, y_true, self.iou_threshold)
            y_pred = model.predict(pred_feature_map)

        tf.summary.scalar("loss/coord_loss",   loss[1])
        tf.summary.scalar("loss/sizes_loss",   loss[2])
        tf.summary.scalar("loss/confs_loss",   loss[3])
        tf.summary.scalar("loss/class_loss",   loss[4])

        global_step = tf.Variable(
            0, trainable=True, collections=[tf.GraphKeys.LOCAL_VARIABLES])

        write_op = tf.summary.merge_all()
        writer_train = tf.summary.FileWriter("../../data/train_summary",
                                             sess.graph)
        writer_test = tf.summary.FileWriter("../../data/test_summary")

        update_vars = tf.contrib.framework.get_variables_to_restore(
            include=["yolov3/yolo-v3"])

        lr = tf.train.exponential_decay(self.learning_rate,
                                        global_step,
                                        decay_steps=self.decay_steps,
                                        decay_rate=self.decay_rate,
                                        staircase=True)
        optimizer = tf.train.AdamOptimizer(lr)

        # set dependencies for BN ops
        update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
        with tf.control_dependencies(update_ops):
            train_op = optimizer.minimize(loss[0],
                                          var_list=update_vars,
                                          global_step=global_step)

        sess.run([tf.global_variables_initializer(),
                  tf.local_variables_initializer()])

        saver = tf.train.Saver(max_to_keep=2)

        for step in range(self.steps):
            run_items = sess.run([train_op, write_op, y_pred, y_true] + loss,
                                 feed_dict={is_training: True})

            if (step+1) % self.eval_internal == 0:
                train_rec_value, train_prec_value = utils.evaluate(
                    run_items[2], run_items[3])

            writer_train.add_summary(run_items[1], global_step=step)
            writer_train.flush()  # Flushes the event file to disk

            if (step+1) % self.save_internal == 0:
                saver.save(sess,
                           save_path=self.checkpoint_path,
                           global_step=step+1)

            if (step+1) % self.print_every_n == 0:
                print(f"=> STEP {step+1} [TRAIN]:\tloss_xy: " +
                      f"{run_items[5]:.4f} \tloss_wh:{run_items[6]:.4f} \t" +
                      f"loss_conf:{run_items[7]:.4f} \tloss_class:" +
                      f"{run_items[8]:.4f}")

            run_items = sess.run([write_op, y_pred, y_true] + loss,
                                 feed_dict={is_training: False})

            if (step+1) % self.eval_internal == 0:
                test_rec_value, test_prec_value = utils.evaluate(run_items[1],
                                                                 run_items[2])
                print(f"\n{20*'='}> evaluation result <{20*'='}\n")
                print(f"=> STEP {step+1} [TRAIN]:\trecall:" +
                      f"{train_rec_value:.2f} \tprecision:" +
                      f"{train_prec_value:.4f}")
                print(f"=> STEP {step+1} [VALID]:\trecall:" +
                      f"{test_rec_value:.2f} \tprecision:" +
                      f"{test_prec_value:.4f}")
                print(f"\n{20*'='}> evaluation result <{20*'='}\n")

            writer_test.add_summary(run_items[0], global_step=step)
            writer_test.flush()  # Flushes the event file to disk


if __name__ == '__main__':
    trainer = Trainer()
    trainer.train()
