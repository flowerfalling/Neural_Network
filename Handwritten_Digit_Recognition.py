# -*- coding: utf-8 -*-
# @Time    : 2022/8/4 16:47
# @Author  : 之落花--falling_flowers
# @File    : Handwritten_Digit_Recognition.py
# @Software: PyCharm
import csv

import numpy as np
import scipy


class NeuralNetwork:
    def __init__(self, inputnnodes, hiddennodes, outputnodes, learingrate):
        self.inodes = inputnnodes
        self.hnodes = hiddennodes
        self.onodes = outputnodes

        self.lr = learingrate
        self.activation_function = lambda x: scipy.special.expit(x)

        self.wih = (np.random.rand(self.hnodes, self.inodes) - 0.5)
        self.who = (np.random.rand(self.onodes, self.hnodes) - 0.5)

    def train(self, inputs_list, targets_list):
        inputs = np.array(inputs_list, ndmin=2).T
        targets = np.array(targets_list, ndmin=2).T

        hidden_inputs = np.dot(self.wih, inputs)
        hidden_outputs = self.activation_function(hidden_inputs)
        final_inputs = np.dot(self.who, hidden_outputs)
        final_outputs = self.activation_function(final_inputs)

        output_errors = targets - final_outputs
        hidden_errors = np.dot(self.who.T, output_errors)
        self.who += self.lr * np.dot(output_errors * final_outputs * (1.0 - final_outputs),
                                     np.transpose(hidden_outputs))
        self.wih += self.lr * np.dot(hidden_errors * hidden_outputs * (1.0 - hidden_outputs), np.transpose(inputs))

    def query(self, inputs_list):
        inputs = np.array(inputs_list, ndmin=2).T
        hidden_inputs = np.dot(self.wih, inputs)
        hidden_outputs = self.activation_function(hidden_inputs)
        final_inputs = np.dot(self.who, hidden_outputs)
        final_outputs = self.activation_function(final_inputs)
        return final_outputs

    # def rquery(self, targets_list):
    #     final_outputs = np.array(targets_list, ndmin=2).T
    #     final_inputs = scipy.special.logit(final_outputs)
    #     hidden_outputs = np.dot(scipy.linalg.pinv(self.who), final_inputs)
    #     hidden_inputs = scipy.special.logit(hidden_outputs)
    #     inputs = np.dot(scipy.linalg.pinv(self.wih), hidden_inputs)
    #
    #     return inputs


def main():
    n = NeuralNetwork(784, 200, 10, 0.1)
    with open('mnist_train.csv', 'r') as f:
        data = csv.reader(f)
        data = list(data)
    onodes = 10
    for _ in range(5):
        for i in data[:100]:
            inputs = (np.asfarray(i[1:]) / 255 * 0.99) + 0.01
            targets = np.zeros(onodes) + 0.01
            targets[int(i[0])] = 0.99
            n.train(inputs, targets)

    with open('mnist_test.csv', 'r') as f:
        data = csv.reader(f)
        data = list(data)
    t = 0
    for i in data:
        out = n.query((np.asfarray(i[1:])) / 255 * 0.9 + 0.01)
        # print(out)
        out_num = np.argmax(out)
        if out_num == int(i[0]):
            t += 1
        # plt.imshow(np.asfarray(i[1:]).reshape(28, 28), cmap='Greys')
        # plt.show()
        # break
    print(t)

    # for i in range(1):
    #     with open('mnist_test.csv') as f:
    #         data = csv.reader(f)
    #         data = list(data)
    #         num_list = n.query(np.asfarray(data[10][1:]))
    #         image_data = n.rquery(num_list.T)
    #         print(n.query(image_data.T))
    #         plt.imshow(image_data.reshape(28, 28), cmap='Greys')
    #         plt.show()


if __name__ == '__main__':
    main()
