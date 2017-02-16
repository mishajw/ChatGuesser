#!/usr/bin/env python

import tensorflow as tf
from tensorflow.python.ops import rnn
from tensorflow.contrib.rnn.python.ops import core_rnn_cell_impl as rnn_cell


class ChatGuesserModel:
    NUM_CLASSES = 18
    NUM_CHARS = 128

    def __init__(self, args):
        """
        Create the model
        :param args: the command line arguments to initialise the model
        """
        self.learning_rate = args.learning_rate
        self.max_sequence_length = args.max_sequence_length
        self.dropout = tf.placeholder(tf.float32)

        self.num_hidden = args.rnn_neurons
        self.num_layers = args.rnn_layers

        # Input and output to get from feed_dict
        self.messages = tf.placeholder("float", [None, self.max_sequence_length, self.NUM_CHARS], name="input")
        self.senders = tf.placeholder("float", [None, self.NUM_CLASSES], name="output")

        # Variables for final softmax layer
        softmax_weights = tf.Variable(tf.random_normal([self.num_hidden, self.NUM_CLASSES]), name="softmax_weights")
        softmax_biases = tf.Variable(tf.random_normal([self.NUM_CLASSES]), name="softmax_biases")

        # Get the prediction
        logits = self.message_rnn(self.messages, softmax_weights, softmax_biases)

        # Get the cost of the prediction
        softmax = tf.nn.softmax_cross_entropy_with_logits(logits=logits, labels=self.senders)
        self.cost = tf.reduce_mean(softmax, name="cost")

        # Optimize based off the cost
        self.optimizer = tf.train.AdamOptimizer(learning_rate=self.learning_rate, name="optimizer").minimize(self.cost)

        # Calculate the accuracy (only for tracking progress)
        with tf.name_scope("accuracy"):
            self.model_guess = tf.argmax(logits, 1, name="model_guess")
            truth = tf.argmax(self.senders, 1, "truth")
            correct_pred = tf.equal(self.model_guess, truth)
            self.accuracy = tf.reduce_mean(tf.cast(correct_pred, tf.float32), name="accuracy")

        # Summaries for TensorBoard
        with tf.name_scope("summaries"):
            tf.summary.scalar("cost", self.cost)
            tf.summary.scalar("accuracy", self.accuracy)
            self.tensor_summary(softmax_weights)
            self.tensor_summary(softmax_biases)
            tf.summary.histogram("guesses", self.model_guess)
            tf.summary.histogram("truths", truth)
            self.all_summaries = tf.summary.merge_all()

    def message_rnn(self, x, w, b):
        """
        Get an RNN for dealing with messages
        :param x: `Tensor` of messages
        :param w: `Tensor` of weights for softmax
        :param b: `Tensor` of biases for softmax
        :return: the prediction for each message
        """

        with tf.name_scope("rnn_main"):
            def new_cell():
                lstm_cell = rnn_cell.BasicLSTMCell(self.num_hidden, state_is_tuple=True, forget_bias=1.0)
                return rnn_cell.DropoutWrapper(lstm_cell, self.dropout)

            multi_cell = rnn_cell.MultiRNNCell([new_cell()] * self.num_layers, state_is_tuple=True)
            outputs, state = rnn.dynamic_rnn(multi_cell, x, dtype=tf.float32, sequence_length=self.real_length(x))

            output = self.get_last_output(outputs)
            return tf.matmul(output, w) + b

    @staticmethod
    def get_last_output(outputs):
        """
        Get the last output of each element in outputs
        Members of outputs may be zero-padded at the end
        :param outputs: list of outputs from an RNN
        :return: last elements of each of outputs
        """

        with tf.name_scope("get_last_output"):
            last_index = tf.shape(outputs)[1] - 1
            outputs_rs = tf.transpose(outputs, [1, 0, 2])

            # TODO: find out what this does
            return tf.nn.embedding_lookup(outputs_rs, last_index)

    @staticmethod
    def real_length(sequence):
        """
        Get the real length of every element in `sequence`
        `sequence` may be zero-padded at end
        :param sequence: list of zero-padded tensors
        :return: lengths
        """

        with tf.name_scope("real_length"):
            used = tf.sign(tf.reduce_max(tf.abs(sequence), reduction_indices=2))
            length = tf.reduce_sum(used, reduction_indices=1)
            length = tf.cast(length, tf.int32)
            return length

    @staticmethod
    def tensor_summary(t):
        t_mean = tf.reduce_mean(t)
        t_stddev = tf.sqrt(tf.reduce_mean(tf.square(t - t_mean)))

        tf.summary.scalar(t.name + "/mean", t_mean)
        tf.summary.scalar(t.name + "/stddev", t_stddev)
        tf.summary.scalar(t.name + "/max", tf.reduce_max(t))
        tf.summary.scalar(t.name + "/min", tf.reduce_min(t))
