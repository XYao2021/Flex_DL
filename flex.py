import copy
from mpi4py import MPI
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
comm = MPI.COMM_WORLD
rank = comm.Get_rank()
size = comm.Get_size()
device = "cuda" if torch.cuda.is_available() else "cpu"

BATCH_SIZE = args.bs
LEARNING_RATE = args.lr
COMPRESS_RATE = args.comp
ITERATION = args.iter * args.agg

"Initialize parameters"
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

# print(size-1)
sample = data_sampling(org_targets, size-1)
# print(sample)
data = split_data(sample, train_data)
# print(data)

# Any constraint for initialize the q_t for each client
# q = [random.random() for _ in range(size-1)]
q = [1 for _ in range(size-1)]
# qt = [i/sum(q) for i in q]
# print(qt)

"""Initial Setting"""
if rank != 0:
    # q_t = qt[rank - 1]
    q_t = q[rank - 1]
    # print(rank, q_t)
    local_data = data[rank - 1]
    torch.manual_seed(42)
    local_model = MNISTModel(input_shape=784,
                             output_shape=10,
                             hidden_units=50)
    local_train_dataloader = DataLoader(dataset=local_data,
                                        batch_size=BATCH_SIZE,
                                        shuffle=True)
    loss_fn = nn.CrossEntropyLoss()
    optimizer = torch.optim.SGD(params=local_model.parameters(),
                                lr=LEARNING_RATE)

    Train_Loss = []
    e_t = torch.zeros_like(torch.cat([para.reshape((-1,)) for para in local_model.parameters()]))
    x_t = torch.cat([para.reshape((-1,)) for para in local_model.parameters()])

    keys = list(local_model.state_dict().keys())
    shapes = [list(x.size()) for x in local_model.parameters()]
    length = [list(x.reshape((-1,)).size())[0] for x in local_model.parameters()]

if rank == 0:
    torch.manual_seed(42)
    test_model = MNISTModel(input_shape=784,
                            output_shape=10,
                            hidden_units=50)
    Test_Acc = []
    Test_Loss = []
    test_dataloader = DataLoader(dataset=test_data,
                                 batch_size=BATCH_SIZE,
                                 shuffle=False)
    loss_fn = nn.CrossEntropyLoss()

    r_t = torch.zeros_like(torch.cat([para.reshape((-1,)) for para in test_model.parameters()]))
    old_weights = torch.cat([para.reshape((-1,)) for para in test_model.parameters()])

    keys = list(test_model.state_dict().keys())
    shapes = [list(x.size()) for x in test_model.parameters()]
    length = [list(x.reshape((-1,)).size())[0] for x in test_model.parameters()]

"""Start Iterations"""
V_t_send = []
u_t_send = []
for iter in range(500):
    if rank != 0:
        train_loss, train_acc = train_step(model=local_model,
                                           data_loader=local_train_dataloader,
                                           loss_fn=loss_fn,
                                           optimizer=optimizer,
                                           accuracy_fn=accuracy_fn,
                                           device=device,
                                           ITERATION=iter)
        Train_Loss.append(train_loss)
        print(rank, 'ITERATION: ', iter, '|', 'Train Loss: ', train_loss)
        sys.stdout.flush()

        alpha = np.random.uniform(low=0, high=1, size=1).item()
        # print('alpha: ', alpha)
        if LAMDA == 0:
            q_t = 1 / D
        else:
            q_t = min(1, math.sqrt(V / (alpha * LAMDA)))
        LAMDA = max(0, LAMDA + alpha * q_t - lamda_avg)
        # q_t = 1

        I_t = np.random.binomial(1, q_t, 1).item()
        gradient = torch.cat([weights.grad.reshape((-1,)) for weights in local_model.parameters()])
        if LEARNING_RATE * (I_t / q_t) != 0:
            b_t_org, b_t, indices = bt_computation(e_t, gradient, LEARNING_RATE * (I_t / q_t))
            k_star, gamma_t = top_k(b_t, V, PHI, beta_t, 2)
            # k_star = 400
            v_t = b_t[: k_star]
            v_t_indices = indices[: k_star]
            b_t_org[v_t_indices] = b_t_org[v_t_indices] - v_t
            e_t = torch.clone(b_t_org)

        else:
            # print(rank, 'No Gradient Update', iter, 'Round')
            # sys.stdout.flush()
            sorted, indices = torch.sort(torch.abs(e_t), descending=True)
            b_t = e_t[indices]
            k_star, gamma_t = top_k(e_t, V, PHI, beta_t, 2)
            # k_star = 400
            v_t = b_t[: k_star]
            v_t_indices = indices[: k_star]
            e_t[v_t_indices] = e_t[v_t_indices] - v_t

        # print(iter, rank, k_star)
        if torch.linalg.norm(v_t, 0).int().item() == 0:
            phi_t = 0
        else:
            phi_t = beta_t + torch.linalg.norm(v_t, 0) * gamma_t
        PHI = max(0, PHI + phi_t - phi_avg)

        if k_star == 0:
            V_t_send = []
        else:
            V_t_send = torch.stack([v_t_indices, v_t], -1)

    V_t_send = comm.gather(V_t_send, root=0)
    if rank == 0:
        V_t_send.pop(0)
        N = len(V_t_send)
        V_t_send = [item for item in V_t_send if item != []]
        if len(V_t_send) != 0:
            a_t = Vt_avg(r_t, V_t_send, N)
        else:
            a_t = r_t
        # print(rank, v_t_avg[-1])
        sorted, indices = torch.sort(torch.abs(a_t), descending=True)
        new_a_t = a_t[indices]
        # print(torch.linalg.norm(a_t, 0), torch.linalg.norm(new_a_t, 0))
        k_star, gamma = top_k(new_a_t, V, PSI, beta_t, 5)
        # k_star = 400
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

        # print(rank, k_star)
        PSI = max(0, PSI + psi_t - psi_avg)

        new_weights = torch.split(old_weights, length)
        test_weights = dict()
        for i in range(len(shapes)):
            test_weights[keys[i]] = new_weights[i].reshape(shapes[i])
        test_model.load_state_dict(test_weights)
        test_loss, test_acc = test_step(model=test_model,
                                        data_loader=test_dataloader,
                                        loss_fn=loss_fn,
                                        accuracy_fn=accuracy_fn,
                                        device=device)
        print('ITERATION: ', iter, '|', 'Test Loss: ', test_loss, '|', 'Test Acc: ', round(test_acc, 4), '%\n')
        sys.stdout.flush()

        Test_Loss.append(test_loss)
        Test_Acc.append(test_acc)

    u_t_send = comm.bcast(u_t_send, root=0)

    if rank != 0:
        if len(u_t_send) == 0:
            print(rank, 'No global updated this round')
            sys.stdout.flush()
        else:
            u_t_indices, u_t_value = torch.unbind(u_t_send, -1)
            # print(rank, u_t_indices)
            # print(rank, u_t_value)
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
        local_model.load_state_dict(x_t_next)

    # print(rank, 'Iteration', iter, 'Completed', '\n')

    # comm.barrier()

if rank == 0:
    print(Test_Acc)
    print(Test_Loss)
    plt.plot(range(len(Test_Acc)), Test_Acc)
    plt.show()
