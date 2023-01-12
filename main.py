import time
import random

import numpy as np
import torch
from torch import nn
import matplotlib.pyplot as plt
import torchvision
from torchvision import datasets
from torchvision import transforms
from torchvision.transforms import ToTensor
from torch.utils.data import DataLoader
from functions import *
import socket
import copy
import math

from matplotlib import patheffects


args = args_parser()
SERVER = args.server
PORT = args.port

ADDR = (SERVER, PORT)
CLIENT = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
CLIENT.connect(ADDR)

BATCH_SIZE = args.bs
LEARNING_RATE = args.lr
AGGREGATION = args.agg
ITERATION = args.iter * AGGREGATION

device = "cuda" if torch.cuda.is_available() else "cpu"

# train_data = datasets.MNIST(root="data",
#                             train=True,
#                             download=True,
#                             transform=ToTensor(),
#                             target_transform=None)
#
# test_data = datasets.MNIST(root="data",
#                            train=False,
#                            download=True,
#                            transform=ToTensor(),
#                            target_transform=None)

train_data = datasets.FashionMNIST(root="data",
                                   train=True,
                                   download=True,
                                   transform=ToTensor(),
                                   target_transform=None)

test_data = datasets.FashionMNIST(root="data",
                                  train=False,
                                  download=True,
                                  transform=ToTensor(),
                                  target_transform=None)

# org_list = list(np.arange(10))
# targets = random.sample(org_list, args.tsn)
# print('This is target set: ', targets)
if args.ts != [0, 2, 4, 6, 8, 9]:
    targets = []
    for t in args.ts:
        targets.append(int(t))
else:
    targets = args.ts
#
# N = 3
# org_classes = train_data.classes
# num_classes = len(org_classes)
# classes = [i for i in range(num_classes)]

# data sampling
# sample = data_sampling(classes, N)
# print(sample)
# data = split_data(sample, train_data)
# for d in data:
#     print(len(d))
# print(np.split(np.array(classes), 3))
train_sets = []
for i in range(len(train_data.targets)):
    if train_data.targets[i] in targets:
        train_sets.append(train_data[i])

# print(len(train_sets), len(test_sets))

train_dataloader = DataLoader(dataset=train_sets,
                              batch_size=BATCH_SIZE,
                              shuffle=True)

test_dataloader = DataLoader(dataset=test_data,
                             batch_size=BATCH_SIZE,
                             shuffle=False)

torch.manual_seed(42)
model = MNISTModel(input_shape=784,
                   output_shape=10,
                   hidden_units=50)

test_model = copy.deepcopy(model)

loss_fn = nn.CrossEntropyLoss()
optimizer = torch.optim.SGD(params=model.parameters(),
                            lr=LEARNING_RATE)
# torch.manual_seed(42)
Train_Loss = []
Test_Loss, Test_Acc = [], []
e_t = torch.zeros_like(torch.cat([para.reshape((-1, )) for para in model.parameters()]))
x_t = torch.cat([para.reshape((-1,)) for para in model.parameters()])

keys = list(model.state_dict().keys())
shapes = [list(x.size()) for x in model.parameters()]
length = [list(x.reshape((-1, )).size())[0] for x in model.parameters()]
# print(shapes)
# print(length)

lamda_avg = 0.25
phi_avg = 0.01
psi_avg = 0.01
V = 0.02
W = 1.0
LAMDA = W
PHI = W
PSI = W
beta_t = 0.05
D = 2

for iter in range(1, 500):
    # print([para for para in model.parameters()])
    # x_t = torch.cat([para.reshape((-1,)) for para in model.parameters()])
    train_loss, train_acc = train_step(model=model,
                                       data_loader=train_dataloader,
                                       loss_fn=loss_fn,
                                       optimizer=optimizer,
                                       accuracy_fn=accuracy_fn,
                                       device=device,
                                       ITERATION=iter)

    print('ITERATION: ', iter, '|', 'Train Loss: ', train_loss, '\n')
    # print('ITERATION: ', iter, '|', 'Train Acc: ', train_acc)
    # Train_Loss.append(train_loss)

    alpha = np.random.uniform(low=0, high=1, size=1).item()
    # print('alpha: ', alpha)
    if LAMDA == 0:
        q_t = 1/D
    else:
        q_t = min(1, math.sqrt(V/(alpha*LAMDA)))
    LAMDA = max(0, LAMDA + alpha*q_t - lamda_avg)
    # q_t = 1

    I_t = np.random.binomial(1, q_t, 1).item()
    gradient = torch.cat([weights.grad.reshape((-1, )) for weights in model.parameters()])
    if LEARNING_RATE*(I_t/q_t) != 0:
        b_t_org, b_t, indices = bt_computation(e_t, gradient, LEARNING_RATE*(I_t/q_t))
        k_star, gamma_t = top_k(b_t, V, PHI, beta_t, 2)
        # k_star = 400
        v_t = b_t[: k_star]
        v_t_indices = indices[: k_star]
        b_t_org[v_t_indices] = b_t_org[v_t_indices] - v_t
        e_t = torch.clone(b_t_org)

    else:
        print('No Gradient Update', iter, 'Round')
        sorted, indices = torch.sort(torch.abs(e_t), descending=True)
        b_t = e_t[indices]
        k_star, gamma_t = top_k(e_t, V, PHI, beta_t, 2)
        # k_star = 400
        v_t = b_t[: k_star]
        v_t_indices = indices[: k_star]
        e_t[v_t_indices] = e_t[v_t_indices] - v_t

    # print(v_t)
    # print(v_t_indices)

    print(iter, k_star)
    if torch.linalg.norm(v_t, 0).int().item() == 0:
        phi_t = 0
    else:
        phi_t = beta_t + torch.linalg.norm(v_t, 0) * gamma_t
    PHI = max(0, PHI + phi_t - phi_avg)

    if k_star == 0:
        V_t_send = []
    else:
        V_t_send = torch.stack([v_t_indices, v_t], -1)

    send_msg(CLIENT, ['MSG_CLIENT_TO_SERVER', V_t_send])

    u_t = recv_msg(CLIENT, 'MSG_SERVER_TO_CLIENT')
    # # print('message received: ', u_t[1])
    if len(u_t[1]) == 0:
        print('No global updated this round')
    else:
        u_t_indices, u_t_value = torch.unbind(u_t[1], -1)
        # print(u_t_indices)
        u_t_indices = u_t_indices.long()
        # print(x_t[u_t_indices])
        x_t[u_t_indices] = x_t[u_t_indices] + u_t_value
        # print(x_t[u_t_indices])

    new_weights = torch.split(x_t, length)
    # print(new_weights)
    x_t_next = dict()
    for i in range(len(shapes)):
        x_t_next[keys[i]] = new_weights[i].reshape(shapes[i])
        # print(new_weights[i].reshape(shapes[i]))
    # print(x_t_next)
    model.load_state_dict(x_t_next)
    # # print('original weights: ', [w for w in model.parameters()])
    # new_weights = weights_update(x_t, u_t[1])
    # # print('new_weights: ', new_weights, '\n')
    # model.load_state_dict(new_weights)
    # # # model_test.load_state_dict(new_weights)
    # # #
    # # test_weights = weights_update(test_model.state_dict(), new_weights)
    # # test_model.load_state_dict(test_weights)
    # test_loss, test_acc = test_step(model=model,
    #                                 data_loader=test_dataloader,
    #                                 loss_fn=loss_fn,
    #                                 accuracy_fn=accuracy_fn,
    #                                 device=device)
    # print('ITERATION: ', iter, '|', 'Test Loss: ', test_loss, '|', 'Test Acc: ', round(test_acc, 4), '%\n')
    #
    # Test_Loss.append(test_loss)
    # Test_Acc.append(test_acc)
    #
    # if iter % 10 == 0:
    #     print('[NEW WEIGHTS LOAD] on number ', iter, 'iteration', '\n')
    #     model.load_state_dict(new_weights[1])

# print('Train Loss: ', Train_Loss, '\n')
# print('Test Loss: ', Test_Loss, '\n')
# print('Test Acc: ', Test_Acc, '\n')
#
# txt_list = [['Train_Loss: ', Train_Loss],
#             ['Test_Loss: ', Test_Loss],
#             ['Test_Acc: ', Test_Acc]]
#
# # f = open('data_{}.txt'.format(args.name), 'w')
# # for item in txt_list:
# #     f.write("%s\n" % item)
#
# figure, axis = plt.subplots(1, 3)
# # For Sine Function
# axis[0].plot(range(len(Train_Loss)), Train_Loss, color='red', path_effects=[patheffects.SimpleLineShadow(shadow_color="blue", linewidth=5), patheffects.Normal()])
# axis[0].set_xlabel("Aggregation")
# axis[0].set_ylabel("Train Loss")
# axis[0].set_title("Training Loss Function")
#
# axis[1].plot(range(len(Test_Acc)), Test_Acc, color='blue', path_effects=[patheffects.SimpleLineShadow(shadow_color="blue", linewidth=5), patheffects.Normal()])
# axis[1].set_xlabel("Aggregation")
# axis[1].set_ylabel("Test Accuracy")
# axis[1].set_title("Test Accuracy Function")
#
# axis[2].plot(range(len(Test_Loss)), Test_Loss, color='green', path_effects=[patheffects.SimpleLineShadow(shadow_color="blue", linewidth=5), patheffects.Normal()])
# axis[2].set_xlabel("Aggregation")
# axis[2].set_ylabel("Test Loss")
# axis[2].set_title("Test Loss Function")
#
# plt.show()

