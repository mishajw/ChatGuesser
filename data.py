#!/usr/bin/env python

data_path = "/home/misha/Dropbox/scala/chat-stats/all-messages.txt"


max_data_amount = 20000


class Data:
    def __init__(self, _input, _output):
        self.input = _input
        self.output = _output


def get_data(training_percentage, message_length):
    print("Reading in data...")

    with open(data_path, 'r') as f:
        data = []

        for line in f:
            separator = line.index(":")
            name = line[:separator]
            message = line[separator + 2:]
            message = [ord(m) for m in message]

            if len(message) < message_length:
                message.extend([0] * (message_length - len(message)))
            elif len(message) > message_length:
                message = message[:message_length]

            data.append(Data(message, name))

        name_set = list(set([d.output for d in data]))

        data = [\
            Data(
                [one_hot(c, 128) for c in d.input],
                one_hot(name_set.index(d.output), len(name_set)))\
            for d in data]

        training_amount = int(min(len(data), max_data_amount) * training_percentage)

        inputs = [d.input for d in data[:max_data_amount]]
        outputs = [d.output for d in data[:max_data_amount]]

        print("Loaded data")

        return \
            inputs[:training_amount], \
            outputs[:training_amount], \
            inputs[training_amount:], \
            outputs[training_amount:], \
            name_set


def one_hot(x, amount):
    vec = [0] * amount
    if x >= amount:
        return vec

    vec[x] = 1
    return vec
