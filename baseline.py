import copy
import tabnanny

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
AGGREGATION = args.agg
CLIENTS = args.cn
SEED = args.seed
RATIO = args.ratio

"Initialize parameters"
lamda_avg = 0.25

# LAMDA = [W for _ in range(CLIENTS)]
# PHI = [W for _ in range(CLIENTS)]
# PSI = W

train_data = datasets.FashionMNIST(root='data',
                                   train=True,
                                   transform=transforms.ToTensor(),
                                   target_transform=None,
                                   download=True)

test_data = datasets.FashionMNIST(root='data',
                                  train=False,
                                  transform=transforms.ToTensor(),
                                  target_transform=None,
                                  download=True)

org_targets = [i for i in range(len(train_data.classes))]

sample = data_sampling(org_targets, CLIENTS, SEED)
data = split_data(sample, train_data)

Models = []
optimizers = []
loss_fns = []
local_dataloaders = []
e_ts = []
x_ts = []
"Set up"
for client in range(CLIENTS):
    torch.manual_seed(SEED)
    model = MNISTModel(input_shape=784, output_shape=10, hidden_units=50)
    model.apply(initial_weights)
    optimizer = torch.optim.SGD(params=model.parameters(), lr=LEARNING_RATE)
    loss_fn = nn.CrossEntropyLoss()
    local_dataloader = DataLoader(dataset=data[client], batch_size=BATCH_SIZE, shuffle=True)
    e_t = torch.zeros_like(torch.cat([para.reshape((-1,)) for para in model.parameters()]))
    x_t = torch.cat([para.reshape((-1,)) for para in model.parameters()])

    Models.append(model)
    optimizers.append(optimizer)
    loss_fns.append(loss_fn)
    local_dataloaders.append(local_dataloader)
    e_ts.append(e_t)
    x_ts.append(x_t)

torch.manual_seed(SEED)
test_model = MNISTModel(input_shape=784, output_shape=10, hidden_units=50)
test_model.apply(initial_weights)
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
Global_Loss = []

for iter in range(AGGREGATION):
    VT = []
    global_loss = []
    for i in range(CLIENTS):
        train_loss, train_acc = train_step(model=Models[i],
                                           data_loader=local_dataloaders[i],
                                           loss_fn=loss_fns[i],
                                           optimizer=optimizers[i],
                                           accuracy_fn=accuracy_fn,
                                           device=device,
                                           ITERATION=iter,
                                           CLIENT=i)
        # print('Round ', iter, '|', 'Client ', i, '|', 'train loss: ', train_loss)
        Train_Loss[i].append(train_loss)
        Train_Acc[i].append(train_acc)
        global_loss.append(train_loss)

        alpha = np.random.uniform(low=0, high=1, size=1).item()
        q_t = min(lamda_avg/alpha, 1)
        # LAMDA[i] = max(0, LAMDA[i] + alpha * q_t - lamda_avg)
        I_t = np.random.binomial(1, q_t, 1).item()
        gradient = torch.cat([weights.grad.reshape((-1,)) for weights in Models[i].parameters()])

        if LEARNING_RATE * (I_t / q_t) != 0:
            b_t_org, b_t, indices = bt_computation(e_ts[i], gradient, LEARNING_RATE * (I_t / q_t))
            k = int(len(b_t) * RATIO)
            v_t = b_t[: k]
            v_t_indices = indices[: k]
            b_t_org[v_t_indices] = b_t_org[v_t_indices] - v_t
            e_ts[i] = torch.clone(b_t_org)
            k_star = torch.linalg.norm(v_t, 0).int().item()
            if k_star <= k:
                v_t = v_t[: k_star]
                v_t_indices = v_t_indices[: k_star]
            V_t_send = torch.stack([v_t_indices, v_t], -1)
        else:
            V_t_send = []
        VT.append(V_t_send)

    g_l = sum(global_loss) / len(global_loss)
    Global_Loss.append(g_l)
    print('Round ', iter, '|', 'Global Training Loss', g_l, '\n')
    N = len(VT)
    VT = [item for item in VT if item != []]
    if len(VT) != 0:
        a_t = Vt_avg(r_t, VT, N)
    else:
        a_t = r_t

    sorted, indices = torch.sort(torch.abs(a_t), descending=True)
    new_a_t = a_t[indices]
    ks = int(len(new_a_t) * RATIO)
    ks_star = torch.linalg.norm(new_a_t[: k], 0).int().item()

    if ks_star <= ks:
        u_t = new_a_t[: ks_star]
        u_t_indices = indices[: ks_star]
    else:
        u_t = new_a_t[: ks]
        u_t_indices = indices[: ks]
    a_t[u_t_indices] = a_t[u_t_indices] - u_t
    r_t = torch.clone(a_t)
    u_t_send = torch.stack([u_t_indices, u_t], -1)
    old_weights[u_t_indices] = old_weights[u_t_indices] + u_t

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
    print('Round', iter, '|', 'Test Loss: ', test_loss, '|', 'Test Acc: ', round(test_acc, 4), '%\n')
    Test_Loss.append(test_loss)
    Test_Acc.append(test_acc)

    for j in range(CLIENTS):
        if len(u_t_send) == 0:
            pass
        else:
            u_t_indices, u_t_value = torch.unbind(u_t_send, -1)
            u_t_indices = u_t_indices.long()
            x_ts[j][u_t_indices] = x_ts[j][u_t_indices] + u_t_value

        new_weights = torch.split(x_ts[j], length)
        x_t_next = dict()
        for m in range(len(shapes)):
            x_t_next[keys[m]] = new_weights[m].reshape(shapes[m])
        Models[j].load_state_dict(x_t_next)

print('Test Loss: ', Test_Loss)
print('Test Acc: ', Test_Acc)

# plt.plot(range(len(Test_Acc)), Test_Acc)
# plt.xlabel('Iterations')
# plt.ylabel('Accuracy %')
# plt.title('Accuracy vs Iteration')

# plt.legend()
# plt.show()

txt_list = [['Test_Loss: ', Test_Loss], '\n',
            ['Test_Acc: ', Test_Acc]]

f = open('results.txt', 'w')
for item in txt_list:
    f.write("%s\n" % item)
