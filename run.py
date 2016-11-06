#!/usr/bin/env python

import tensorflow as tf
import numpy as np
from datetime import datetime

import data
from model import ChatGuesserModel

max_sequence_length = 50
batch_size = 64
training_percentage = 0.75
training_iters = 1e4
display_step = 10

now = datetime.now()
path = "/tmp/tb_chat_guesser/" + now.strftime("%Y%m%d-%H%M%S")


def main():
    model = ChatGuesserModel(max_sequence_length)

    saver = tf.train.Saver()
    checkpoint = tf.train.get_checkpoint_state(path + "/checkpoint")

    init = tf.initialize_all_variables()

    with tf.Session() as sess:
        # Try and restore variables
        if checkpoint and checkpoint.model_checkpoint_path:
            saver.restore(sess, checkpoint.model_checkpoint_path)

        sess.run(init)
        step = 1

        inputs, outputs, test_inputs, test_outputs, name_set = \
            data.get_data(training_percentage, max_sequence_length)

        summary_train_writer, summary_test_writer = get_writers(sess)

        while step < training_iters:
            batch_step = step % int(len(outputs) / batch_size)

            batch_x = np.array(
                inputs[batch_step * batch_size:(batch_step + 1) * batch_size]
            )

            batch_y = np.array(
                outputs[batch_step * batch_size:(batch_step + 1) * batch_size]
            )

            _, train_summaries = sess.run(
                [model.optimizer, model.all_summaries],
                feed_dict={model.messages: batch_x, model.senders: batch_y})

            summary_train_writer.add_summary(train_summaries, step)

            if step % display_step == 0:
                acc, loss, test_summaries = sess.run(
                    [model.accuracy, model.cost, model.all_summaries],
                    feed_dict={model.messages: test_inputs, model.senders: test_outputs})

                print("Acc: %.10f, Loss: %.10f" % (acc, loss))
                summary_test_writer.add_summary(test_summaries, step)

            step += 1


def get_writers(sess):
    def writer(name):
        return tf.train.SummaryWriter(
            path + "/" + name,
            sess.graph)

    return writer("train"), writer("test")


if __name__ == "__main__":
    main()
