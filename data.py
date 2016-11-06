#!/usr/bin/env python

data_path = "/home/misha/Dropbox/scala/chat-stats/all-messages.txt"


def get_data(training_percentage, message_length):
    with open(data_path, 'r') as f:
        inputs = []
        outputs = []

        for line in f:
            separator = line.index(":")
            name = line[:separator]
            message = line[separator + 2:]
            message = [ord(m) for m in message]

            if len(message) < message_length:
                message.extend([0] * (message_length - len(message)))
            elif len(message) > message_length:
                message = message[:message_length]

            inputs.append(message)
            outputs.append(name)

        name_set = list(set(outputs))
        outputs = [name_set.index(o) for o in outputs]

        inputs = [one_hot(message, 128) for message in inputs]
        outputs = one_hot(outputs, len(name_set))

        training_amount = int(len(inputs) * training_percentage)

        return \
            inputs[:training_amount], \
            outputs[:training_amount], \
            inputs[training_amount:], \
            outputs[training_amount:], \
            name_set


def one_hot(xs, amount):
    def one_hot_single(x):
        vec = [0] * amount
        if x >= amount:
            return vec

        vec[x] = 1
        return vec

    return [one_hot_single(x) for x in xs]
