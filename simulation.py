import copy
import torch
from functions import *
from torchvision import datasets
from torchvision import transforms
from torch.utils.data import DataLoader
import sys
import numpy as np
import random
import matplotlib.pyplot as plt
import time
import os
import math

os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'
args = args_parser()
device = "cuda" if torch.cuda.is_available() else "cpu"

BATCH_SIZE = args.bs
LEARNING_RATE = args.lr
COMPRESS_RATE = args.comp
ITERATION = args.iter * args.agg
CLIENTS = args.cn

"Initialize parameters"
lamda_avg = 0.25
phi_avg = 0.01
psi_avg = 0.01
V = 0.02
W = 1.0
beta_t = 0.05
D = 2

LAMDA = [W for _ in range(CLIENTS)]
PHI = [W for _ in range(CLIENTS)]
PSI = W

train_data = datasets.MNIST(root="data",
                            train=True,
                            transform=transforms.ToTensor(),
                            target_transform=None,
                            download=True)
test_data = datasets.MNIST(root="data",
                           train=False,
                           transform=transforms.ToTensor(),
                           target_transform=None,
                           download=True)

org_targets = [i for i in range(len(train_data.classes))]

sample = data_sampling(org_targets, CLIENTS)
data = split_data(sample, train_data)

torch.manual_seed(42)
Models = [MNISTModel(input_shape=784, output_shape=10, hidden_units=50) for i in range(CLIENTS)]
optimizers = [torch.optim.SGD(params=Models[i].parameters(), lr=LEARNING_RATE) for i in range(CLIENTS)]
loss_fns = [nn.CrossEntropyLoss() for i in range(CLIENTS)]
local_dataloaders = [DataLoader(dataset=data[i], batch_size=BATCH_SIZE, shuffle=True) for i in range(CLIENTS)]
e_ts = [torch.zeros_like(torch.cat([para.reshape((-1,)) for para in Models[i].parameters()])) for i in range(CLIENTS)]
x_ts = [torch.cat([para.reshape((-1,)) for para in Models[i].parameters()]) for i in range(CLIENTS)]
# print(len(Models), len(optimizers), len(local_dataloaders))

torch.manual_seed(42)
test_model = MNISTModel(input_shape=784, output_shape=10, hidden_units=50)
test_loss_fn = nn.CrossEntropyLoss()
test_dataloader = DataLoader(dataset=test_data,
                             batch_size=BATCH_SIZE,
                             shuffle=False)
r_t = torch.zeros_like(torch.cat([para.reshape((-1,)) for para in test_model.parameters()]))
old_weights = torch.cat([para.reshape((-1,)) for para in test_model.parameters()])

keys = list(test_model.state_dict().keys())
shapes = [list(x.size()) for x in test_model.parameters()]
length = [list(x.reshape((-1,)).size())[0] for x in test_model.parameters()]

Train_Loss = [[] for _ in range(CLIENTS)]
Train_Acc = [[] for _ in range(CLIENTS)]
Test_Loss = []
Test_Acc = []

for iter in range(500):
    VT = []
    for i in range(CLIENTS):
        train_loss, train_acc = train_step(model=Models[i],
                                           data_loader=local_dataloaders[i],
                                           loss_fn=loss_fns[i],
                                           optimizer=optimizers[i],
                                           accuracy_fn=accuracy_fn,
                                           device=device,
                                           ITERATION=iter,
                                           CLIENT=i)
        print(iter, 'Client ', i, 'train loss: ', train_loss)
        Train_Loss[i].append(train_loss)
        Train_Acc[i].append(train_acc)

        alpha = np.random.uniform(low=0, high=1, size=1).item()
        if LAMDA[i] == 0:
            q_t = 1 / D
        else:
            q_t = min(1, math.sqrt(V / (alpha * LAMDA[i])))
        LAMDA[i] = max(0, LAMDA[i] + alpha * q_t - lamda_avg)
        # print(iter, i, 'LAMDA_i: ', LAMDA[i], '\n')

        I_t = np.random.binomial(1, q_t, 1).item()
        gradient = torch.cat([weights.grad.reshape((-1,)) for weights in Models[i].parameters()])

        if LEARNING_RATE * (I_t / q_t) != 0:
            b_t_org, b_t, indices = bt_computation(e_ts[i], gradient, LEARNING_RATE * (I_t / q_t))
            k_star, gamma_t = top_k(b_t, V, PHI[i], beta_t, 2)
            # k_star = 400
            v_t = b_t[: k_star]
            v_t_indices = indices[: k_star]
            b_t_org[v_t_indices] = b_t_org[v_t_indices] - v_t
            e_ts[i] = torch.clone(b_t_org)

        else:
            # print(rank, 'No Gradient Update', iter, 'Round')
            # sys.stdout.flush()
            sorted, indices = torch.sort(torch.abs(e_ts[i]), descending=True)
            b_t = e_ts[i][indices]
            k_star, gamma_t = top_k(e_ts[i], V, PHI[i], beta_t, 2)
            # k_star = 400
            v_t = b_t[: k_star]
            v_t_indices = indices[: k_star]
            e_ts[i][v_t_indices] = e_ts[i][v_t_indices] - v_t

        if torch.linalg.norm(v_t, 0).int().item() == 0:
            phi_t = 0
        else:
            phi_t = beta_t + torch.linalg.norm(v_t, 0) * gamma_t
        PHI[i] = max(0, PHI[i] + phi_t - phi_avg)
        # print(iter, i, 'PHI_i: ', PHI[i], '\n')

        if k_star == 0:
            V_t_send = []
        else:
            V_t_send = torch.stack([v_t_indices, v_t], -1)

        VT.append(V_t_send)
    # print(iter, len(VT))
    N = len(VT)
    VT = [item for item in VT if item != []]
    # print(iter, len(VT))
    if len(VT) != 0:
        a_t = Vt_avg(r_t, VT, N)
    else:
        a_t = r_t
#
    sorted, indices = torch.sort(torch.abs(a_t), descending=True)
    new_a_t = a_t[indices]
    # print(torch.linalg.norm(a_t, 0), torch.linalg.norm(new_a_t, 0))
    k_star, gamma = top_k(new_a_t, V, PSI, beta_t, 5)

    if k_star == 0:
        r_t = torch.clone(a_t)
        u_t_send = []
        psi_t = 0
    else:
        u_t = new_a_t[: k_star]
        u_t_indices = indices[: k_star]
        # print(u_t)
        # print(u_t_indices)
        a_t[u_t_indices] = a_t[u_t_indices] - u_t
        r_t = torch.clone(a_t)
        u_t_send = torch.stack([u_t_indices, u_t], -1)
        old_weights[u_t_indices] = old_weights[u_t_indices] + u_t
        psi_t = beta_t + torch.linalg.norm(u_t, 0) * gamma

    PSI = max(0, PSI + psi_t - psi_avg)
    # print(iter, PSI, '\n')

    new_weights = torch.split(old_weights, length)
    test_weights = dict()
    for k in range(len(shapes)):
        test_weights[keys[k]] = new_weights[k].reshape(shapes[k])
    test_model.load_state_dict(test_weights)
    test_loss, test_acc = test_step(model=test_model,
                                    data_loader=test_dataloader,
                                    loss_fn=test_loss_fn,
                                    accuracy_fn=accuracy_fn,
                                    device=device)
    print('ITERATION: ', iter, '|', 'Test Loss: ', test_loss, '|', 'Test Acc: ', round(test_acc, 4), '%\n')
    Test_Loss.append(test_loss)
    Test_Acc.append(test_acc)

    for j in range(CLIENTS):
        if len(u_t_send) == 0:
            # print(iter, j, 'No global updated this round')
            pass
        else:
            u_t_indices, u_t_value = torch.unbind(u_t_send, -1)
            # print(rank, u_t_indices)
            # print(rank, u_t_value)
            u_t_indices = u_t_indices.long()
            # print(x_t[u_t_indices])
            x_ts[j][u_t_indices] = x_ts[j][u_t_indices] + u_t_value
            # print(x_t[u_t_indices])

        new_weights = torch.split(x_ts[j], length)
        # print(new_weights)
        x_t_next = dict()
        for m in range(len(shapes)):
            x_t_next[keys[m]] = new_weights[m].reshape(shapes[m])
            # print(new_weights[i].reshape(shapes[i]))
        # print(x_t_next)
        Models[j].load_state_dict(x_t_next)

