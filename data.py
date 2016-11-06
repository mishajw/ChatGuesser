#!/usr/bin/env python

data_path = "/home/misha/Dropbox/scala/chat-stats/all-messages.txt"


def get_data():
    with open(data_path, 'r') as f:
        inputs = []
        outputs = []

        for line in f:
            separator = line.index(":")
            name = line[:separator]
            message = line[separator + 2:]
            message = [ord(m) for m in message]

            inputs.append(message)
            outputs.append(name)

        name_set = list(set(outputs))
        outputs = [name_set.index(o) for o in outputs]

        return inputs, outputs, name_set
