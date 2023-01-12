import torch
from torch import nn
# import matplotlib.pyplot as plt
# import torchvision
# from torchvision import datasets
# from torchvision import transforms
# from torchvision.transforms import ToTensor
# from torch.utils.data import DataLoader
import struct
import numpy as np
import pickle
import copy
import socket
import argparse
import random
import heapq


HEADERSIZE = 10  # Bytes
FORMAT = 'utf-8'
SIZE = 1024

def args_parser():
    parse = argparse.ArgumentParser()

    parse.add_argument('-agg', type=int, default=10, help='Global Aggregation times')
    parse.add_argument('-lr', type=float, default=0.1, help='Learning Rate of the Model')
    parse.add_argument('-bs', type=int, default=32, help='Batch Size for model')
    parse.add_argument('-ts', type=list, default=[0, 2, 4, 6, 8, 9], help='Target set for training and local testing')
    parse.add_argument('-iter', type=int, default=20, help='Local Training Times: Iterations')
    parse.add_argument('-comp', type=float, default=0.5, help='The rate of compression')

    parse.add_argument('-server', type=str, default='172.16.0.1', help='Server IP address')
    parse.add_argument('-port', type=int, default=5050, help='Socket port')
    parse.add_argument('-bond', type=int, default=2, help='Threshold for FedAvg on Sever side')
    parse.add_argument('-name', type=int, default=1, help='Client Name')

    args = parse.parse_args()
    return args

def accuracy_fn(y_true, y_pred):
    correct = torch.eq(y_true, y_pred).sum().item()
    acc = (correct/len(y_pred)) * 100
    return acc

class MNISTModel(nn.Module):  # Improve the model V0 with nonlinear activation function nn.Relu()
    def __init__(self, input_shape,
                 output_shape,
                 hidden_units):
        super().__init__()
        self.layer_stack = nn.Sequential(nn.Flatten(),  # Equal to x.view(-1, 784)
                                         nn.Linear(in_features=input_shape, out_features=hidden_units),
                                         nn.ReLU(),
                                         nn.Linear(in_features=hidden_units, out_features=output_shape))

    def forward(self, x):
        return self.layer_stack(x)

class CIFAR_10(nn.Module):
    def __init__(self, input_shape=3, output_shape=10):
        super().__init__()
        self.conv_layer = nn.Sequential(nn.Conv2d(in_channels=input_shape,
                                                  out_channels=256, kernel_size=3, padding=1),
                                        nn.ReLU(),
                                        nn.Conv2d(in_channels=256,
                                                  out_channels=512, kernel_size=3, padding=1),
                                        nn.MaxPool2d(stride=2, kernel_size=2))
        self.linear_layer = nn.Sequential(nn.Linear(in_features=512, out_features=512),
                                          nn.ReLU(),
                                          nn.Linear(in_features=512, out_features=output_shape))

    def forward(self, x):
        x = self.conv_layer(x)
        x = self.linear_layer(x)
        return x

def data_sampling(classes, N):
    num_classes = len(classes)
    sample = []
    if N <= num_classes:
        n, m = num_classes % N, int(num_classes / N)
        if n == 0:
            random.seed(42)
            sample = random.sample(classes, k=num_classes)
            sample = np.split(np.array(sample), N)
        else:
            random.seed(42)
            org_sample = random.sample(classes, k=num_classes)
            for i in range(N):
                if N - i <= n:
                    sample.append(org_sample[m * i:m * i + m + 1])
                else:
                    sample.append(org_sample[m * i:m * i + m])
    else:
        n, m = N % num_classes, int(N / num_classes)
        if n == 0:
            random.seed(42)
            for i in range(m):
                sample += random.sample(classes, k=num_classes)
            sample = np.split(np.array(sample), N)
        else:
            random.seed(42)
            for i in range(m):
                sample += random.sample(classes, k=num_classes)
            for j in range(n):
                sample += random.sample(classes, k=1)
            sample = np.split(np.array(sample), N)
    return sample

def split_data(sample, train_data):
    data = []
    for sam in sample:
        local_data = []
        for i in range(len(train_data.targets)):
            # print(train_data.targets[i])
            if train_data.targets[i] in list(sam):
                local_data.append(train_data[i])
        data.append(local_data)
    return data

# def top_K_compression(parameters):  # for model.parameters()
#     Trans = []
#     for param in parameters:
#         trans = torch.zeros_like(param)
#         k = int(max(param.size()) * 0.1)
#         weight = param.grad
#         values, indices = torch.topk(torch.abs(weight), k)
#         if param.ndimension() != 1:
#             for i in range(len(weight)):
#                 trans[i][indices[i]] = weight[i][indices[i]]
#         else:
#             trans[indices] = weight[indices]
#         Trans.append(trans)
#     return Trans

def top_K_compression_ratio(parameters, ratio):  # with ratio for every layer
    Trans = []
    for param in parameters:
        trans = torch.zeros_like(param)
        k = int(max(param.size()) * ratio)
        # weight = param.grad
        values, indices = torch.topk(torch.abs(param), k)
        if param.ndimension() != 1:
            for i in range(len(param)):
                trans[i][indices[i]] = param[i][indices[i]]
        else:
            trans[indices] = param[indices]
        Trans.append(trans)
    return Trans

def top_k(b_t, V, PHI, beta_t, df):  # Also works for u_t
    gamma_t = 1 / (len(b_t) * (0.5 * np.log2(1 + np.random.chisquare(df=df))))
    i = 0
    k_star = 0
    while i < len(b_t):
        v_t = torch.zeros_like(b_t)
        v_t[:i] = b_t[:i]
        if PHI * gamma_t - V * (b_t[i] ** 2) >= 0:
            if V * (torch.linalg.norm(b_t, 1) ** 2) > V * (torch.linalg.norm(b_t - v_t, 1) ** 2) + PHI * (beta_t + gamma_t * i):
                k_star = i
                break
            else:
                k_star = 0
        i += 1
    return k_star, gamma_t

def train_step(model,
               data_loader,
               loss_fn,
               optimizer,
               accuracy_fn,
               device,
               ITERATION):

    train_loss, train_acc = 0, 0
    model.train()
    random.seed(ITERATION)
    seed = random.randint(0, len(data_loader) - 1)
    # print(ITERATION, seed)
    X, y = list(iter(data_loader))[seed]
    X, y = X.to(device), y.to(device)

    y_pred = model(X)
    loss = loss_fn(y_pred, y)
    train_loss += loss
    train_acc += accuracy_fn(y, y_pred.argmax(dim=1))
    # print(train_loss, '\n')

    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    return train_loss.item(), train_acc

def test_step(model,
              data_loader,
              loss_fn,
              accuracy_fn,
              device):
    test_loss, test_acc = 0, 0

    model.eval()
    with torch.inference_mode():
        for X, y in data_loader:
            X, y = X.to(device), y.to(device)
            test_pred = model(X)

            test_loss += loss_fn(test_pred, y)
            test_acc += accuracy_fn(y, test_pred.argmax(dim=1))
        # Adjust metrics and print out
        test_loss /= len(data_loader)
        test_acc /= len(data_loader)
    return test_loss.item(), test_acc

def bt_computation(et, gradient, constant):
    b_t_org = et - constant*gradient
    sorted, indices = torch.sort(torch.abs(b_t_org), descending=True)
    b_t = b_t_org[indices]
    return b_t_org, b_t, indices

def at_computation(rt, vt_avg):
    result = []
    for i in range(len(rt)):
        result.append(rt[i] + vt_avg[i])
    return result

def FedAvg(w):
    w_avg = copy.deepcopy(w[0])
    for k in w_avg.keys():
        for i in range(1, len(w)):
            w_avg[k] += w[i][k]
        w_avg[k] = torch.div(w_avg[k], len(w))
    return w_avg

def Vt_avg(rt, Vt, N):
    unstack = []
    for item in Vt:
        unstack.append(torch.unbind(item, -1))
    at = torch.zeros_like(rt)
    for i in range(len(rt)):
        a = 0
        for j in range(len(unstack)):
            if i in unstack[j][0].int():
                a += unstack[j][1][(unstack[j][0].int() == i).nonzero().item()]
        at[i] = rt[i] + a/N
    return at

def weights_update(state_dict, u_t):
    keys = [key for key in state_dict.keys()]
    for i in range(len(keys)):
        state_dict[keys[i]] += u_t[i]
    return state_dict

def send_msg(sock, msg):
    msg_pickle = pickle.dumps(msg)
    sock.sendall(struct.pack(">I", len(msg_pickle)))
    sock.send(msg_pickle)
    # print(msg[0], 'sent to', sock.getpeername(), '\n')

def recv_msg(sock, expect_msg_type=None):
    msg_len = struct.unpack(">I", sock.recv(4))[0]
    msg = sock.recv(msg_len, socket.MSG_WAITALL)
    msg = pickle.loads(msg)
    # print(msg[0], 'received from', sock.getpeername(), '\n')

    if (expect_msg_type is not None) and (msg[0] != expect_msg_type):
        raise Exception("Expected " + expect_msg_type + " but received " + msg[0])
    return msg
