#!/usr/bin/env python

import random
from itertools import groupby


class Data:
    def __init__(self, _input, _output):
        self.input = _input
        self.output = _output


def get_data(data_path, training_percentage, message_length, max_data_amount):
    print("Reading in data...")

    plain_data = get_plain_data(data_path)
    random.shuffle(plain_data)
    plain_data = plain_data[:max_data_amount]
    one_hot_data = get_one_hot_data(plain_data, message_length)
    proportional_data = get_proportional_data(one_hot_data)

    training_amount = int(min(len(proportional_data), max_data_amount) * training_percentage)

    inputs = [d.input for d in proportional_data[:max_data_amount]]
    outputs = [d.output for d in proportional_data[:max_data_amount]]

    print("Loaded data")

    return \
        inputs[:training_amount], \
        outputs[:training_amount], \
        inputs[training_amount:], \
        outputs[training_amount:]


def get_plain_data(data_path):
    with open(data_path, 'r') as f:
        data = []

        for line in f:
            separator = line.index(":")
            name = line[:separator]
            message = line[separator + 2:]

            data.append(Data(message, name))

    return data


def get_one_hot_data(plain_data, message_length):
    one_hot_data = []

    name_set = list(set([d.output for d in plain_data]))

    for d in plain_data:
        input = [ord(m) for m in d.input]

        if len(input) < message_length:
            input.extend([0] * (message_length - len(input)))
        elif len(input) > message_length:
            input = input[:message_length]

        input = [one_hot(c, 128) for c in input]
        output = one_hot(name_set.index(d.output), len(name_set))

        one_hot_data.append(Data(input, output))

    return one_hot_data


def get_proportional_data(one_hot_data):
    num_classes = len(one_hot_data[0].input)

    one_hot_data.sort(key=lambda d: d.output)
    data_per_group = int(len(one_hot_data) / num_classes)

    proportional_data = []
    for k, ds in groupby(one_hot_data, lambda d: d.output):
        ds = list(ds)
        random.shuffle(ds)
        proportional_data.extend(ds[:data_per_group])

    random.shuffle(proportional_data)

    return proportional_data


def one_hot(x, amount):
    vec = [0] * amount
    if x >= amount:
        return vec

    vec[x] = 1
    return vec
