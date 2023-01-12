import socket

import matplotlib.pyplot as plt
import torch

from functions import *
import select
from torch.utils.data import DataLoader
from torchvision import datasets
from torchvision import transforms
import os

os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'
args = args_parser()
HEADER_LENGTH = 10
PORT = args.port
SERVER = socket.gethostbyname(socket.gethostname())
# SERVER = args.server
THRESHOLD = args.bond
# AGGREGATION = args.agg
COMPRESS_RATE = args.comp
BATCH_SIZE = args.bs

ADDR = (SERVER, PORT)
server = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
server.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)  #set reuse the address

server.bind(ADDR)
server.listen()
sockets_list = [server]
clients = []
Weights = []

# test_data = datasets.MNIST(root="data",
#                            train=False,
#                            transform=transforms.ToTensor(),
#                            target_transform=None,
#                            download=True)

test_data = datasets.FashionMNIST(root="data",
                                  train=False,
                                  download=True,
                                  transform=transforms.ToTensor(),
                                  target_transform=None)

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
r_t = torch.zeros_like(torch.cat([para.reshape((-1, )) for para in test_model.parameters()]))
old_weights = torch.cat([para.reshape((-1,)) for para in test_model.parameters()])

device = "cuda" if torch.cuda.is_available() else "cpu"

print('Listening for connections on ', SERVER)

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

keys = list(test_model.state_dict().keys())
shapes = [list(x.size()) for x in test_model.parameters()]
length = [list(x.reshape((-1, )).size())[0] for x in test_model.parameters()]

com_time = 1
while True:
    read_sockets, _, exception_sockets = select.select(sockets_list, [], sockets_list)
    for notified_socket in read_sockets:
        if notified_socket == server:
            client_socket, client_address = server.accept()

            sockets_list.append(client_socket)
            clients.append(client_socket)
            print('Accepted new connection from ', client_address)

            msg_recv = recv_msg(client_socket)
            if msg_recv is False:
                continue
            # print('weights_recv : ', msg_recv[1], '\n')
            Weights.append(msg_recv[1])

            if len(Weights) == THRESHOLD:
                # print('[COMPUTING] start weights computing...')
                # print(Weights)
                N = len(Weights)
                Weights = [item for item in Weights if item != []]
                if len(Weights) != 0:
                    a_t = Vt_avg(r_t, Weights, N)
                else:
                    a_t = r_t
                sorted, indices = torch.sort(torch.abs(a_t), descending=True)
                new_a_t = a_t[indices]
                # print(torch.linalg.norm(a_t, 0), torch.linalg.norm(new_a_t, 0))
                k_star, gamma = top_k(new_a_t, V, PSI, beta_t, 5)
                # k_star = 400
                print(com_time, k_star)
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

                new_weights = torch.split(old_weights, length)
                test_weights = dict()
                for i in range(len(shapes)):
                    test_weights[keys[i]] = new_weights[i].reshape(shapes[i])

                # test_weights = weights_update(test_model.state_dict(), u_t)
                test_model.load_state_dict(test_weights)
                test_loss, test_acc = test_step(model=test_model,
                                                data_loader=test_dataloader,
                                                loss_fn=loss_fn,
                                                accuracy_fn=accuracy_fn,
                                                device=device)
                print('ITERATION: ', com_time, '|', 'Test Loss: ', test_loss, '|', 'Test Acc: ', round(test_acc, 4), '%\n')
                Test_Acc.append(test_acc)
                Test_Loss.append(test_loss)

                for client in clients:
                    send_msg(client, ['MSG_SERVER_TO_CLIENT', u_t_send])
                Weights.clear()
                # print('[FINISHED] new weights already sent back...', '\n')
                print(com_time, '[NEW FINISHED]', '\n')
                com_time += 1
        else:
            msg_recv = recv_msg(notified_socket)
            if msg_recv is False:
                print('[FALSE] Closed connection from: ', notified_socket, '...')
                sockets_list.remove(notified_socket)
                clients.remove(notified_socket)
                continue
            Weights.append(msg_recv[1])
            # print(f'Message received from {notified_socket}...')
            if len(Weights) == THRESHOLD:
                # old_weights = torch.cat([para.reshape((-1,)) for para in test_model.parameters()])
                N = len(Weights)
                Weights = [item for item in Weights if item != []]
                if len(Weights) != 0:
                    a_t = Vt_avg(r_t, Weights, N)
                else:
                    a_t = r_t
                sorted, indices = torch.sort(torch.abs(a_t), descending=True)
                new_a_t = a_t[indices]
                # print(torch.linalg.norm(new_a_t, 0))
                k_star, gamma = top_k(new_a_t, V, PSI, beta_t, 5)
                print(com_time, k_star)
                # k_star = 400
                if k_star == 0:
                    r_t = torch.clone(a_t)
                    u_t_send = []
                    psi_t = 0
                else:
                    u_t = new_a_t[: k_star]
                    u_t_indices = indices[: k_star]
                    a_t[u_t_indices] = a_t[u_t_indices] - u_t
                    r_t = torch.clone(a_t)
                    u_t_send = torch.stack([u_t_indices, u_t], -1)
                    old_weights[u_t_indices] = old_weights[u_t_indices] + u_t
                    psi_t = beta_t + torch.linalg.norm(u_t, 0) * gamma

                PSI = max(0, PSI + psi_t - psi_avg)

                new_weights = torch.split(old_weights, length)
                test_weights = dict()
                for i in range(len(shapes)):
                    test_weights[keys[i]] = new_weights[i].reshape(shapes[i])

                # test_weights = weights_update(test_model.state_dict(), u_t)
                test_model.load_state_dict(test_weights)
                test_loss, test_acc = test_step(model=test_model,
                                                data_loader=test_dataloader,
                                                loss_fn=loss_fn,
                                                accuracy_fn=accuracy_fn,
                                                device=device)
                print('ITERATION: ', com_time, '|', 'Test Loss: ', test_loss, '|', 'Test Acc: ', round(test_acc, 4),
                      '%\n')
                Test_Acc.append(test_acc)
                Test_Loss.append(test_loss)

                for client in clients:
                    send_msg(client, ['MSG_SERVER_TO_CLIENT', u_t_send])
                Weights.clear()
                # print('[FINISHED] new weights already sent back...', '\n')
                print(com_time, '[NEW FINISHED]', '\n')
                com_time += 1
                if com_time == 500:
                    print(Test_Acc)
                    print(Test_Loss)
                    # plt.plot(range(len(Test_Acc)), Test_Acc)
                    # plt.plot(range(len(Test_Loss)), Test_Loss)
                    # plt.show()

    for notified_socket in exception_sockets:
        # Remove from list for socket.socket()
        sockets_list.remove(notified_socket)
        clients.remove(notified_socket)
