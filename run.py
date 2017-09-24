import tensorflow as tf
import numpy as np
from datetime import datetime
import argparse

import data
from model import ChatGuesserModel

now = datetime.now()


parser = argparse.ArgumentParser(
    description="Recurrent Neural Network that trains to guess the sender of a message")
parser.add_argument("--data-path", type=str, required=True)
parser.add_argument("--batch-size", type=int, default=64)
parser.add_argument("--training-percentage", type=float, default=0.75)
parser.add_argument("--display-step", type=int, default=100)
parser.add_argument("--save-step", type=int, default=1000)
parser.add_argument("--max-sequence-length", type=int, default=50)
parser.add_argument("--learning-rate", type=float, default=0.0001)
parser.add_argument("--rnn-neurons", type=int, default=16)
parser.add_argument("--rnn-layers", type=int, default=4)
parser.add_argument("--max-data-amount", type=int, default=int(1e6))
parser.add_argument("--save-dir", type=str, default="/tmp/tb_chat_guesser/" + now.strftime("%Y%m%d-%H%M%S"))
parser.add_argument("--class-amount", type=int, default=6)


def main():
    args = parser.parse_args()

    data_path = args.data_path
    path = args.save_dir
    checkpoint_path = path + "/model.ckpt"
    max_sequence_length = args.max_sequence_length
    batch_size = args.batch_size
    training_percentage = args.training_percentage
    display_step = args.display_step
    save_step = args.save_step
    max_data_amount = args.max_data_amount

    model = ChatGuesserModel(args)

    saver = tf.train.Saver()
    checkpoint = tf.train.get_checkpoint_state(checkpoint_path)

    init = tf.initialize_all_variables()

    with tf.Session() as sess:
        # Try and restore variables
        if checkpoint and checkpoint.model_checkpoint_path:
            saver.restore(sess, checkpoint.model_checkpoint_path)

        sess.run(init)
        step = 1

        inputs, outputs, test_inputs, test_outputs = \
            data.get_data(data_path, training_percentage, max_sequence_length, max_data_amount)

        summary_train_writer, summary_test_writer = get_writers(sess, path)

        while True:
            batch_step = step % int(len(outputs) / batch_size)

            batch_x = np.array(
                inputs[batch_step * batch_size:(batch_step + 1) * batch_size]
            )

            batch_y = np.array(
                outputs[batch_step * batch_size:(batch_step + 1) * batch_size]
            )

            _, train_summaries = sess.run(
                [model.optimizer, model.all_summaries],
                feed_dict={model.messages: batch_x, model.senders: batch_y, model.dropout: 0.5})

            summary_train_writer.add_summary(train_summaries, step)

            if step % display_step == 0:
                acc, loss, test_summaries = sess.run(
                    [model.accuracy, model.cost, model.all_summaries],
                    feed_dict={model.messages: test_inputs, model.senders: test_outputs, model.dropout: 1})

                print("Acc: %.10f, Loss: %.10f" % (acc, loss))
                summary_test_writer.add_summary(test_summaries, step)

            if step % save_step == 0:
                saver.save(sess, checkpoint_path)

            step += 1


def get_writers(sess, path):
    def writer(name):
        return tf.summary.FileWriter(
            path + "/" + name,
            sess.graph)

    return writer("train"), writer("test")


if __name__ == "__main__":
    main()
