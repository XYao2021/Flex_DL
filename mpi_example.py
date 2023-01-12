from mpi4py import MPI
import sys
import numpy as np
import random
from sympy import Matrix
import time
import argparse

def args_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument('-bl', type=int, default=100000, help="bit length")
    args = parser.parse_args()
    return args


'Initialize the MPI package and get the rank and size for each node'
args = args_parser()
comm = MPI.COMM_WORLD
rank = comm.Get_rank()
size = comm.Get_size()

'Generate the key parameters and communication groups'
K = size - 1
U = int((K+1)/2)
S = K - U + 1
L = args.bl
q = 7
org_group = list(np.arange(1, K+1))

groups = []
for i in range(len(org_group)):
    if i+(K-U+1) > len(org_group):
        a1 = org_group[i:]
        remain = K - U + 1 - len(a1)
        a2 = org_group[:remain]
        groups.append(a1 + a2)
    else:
        groups.append(org_group[i:i+K-U+1])

'Generate coefficients for each group and share the coefficients in the group'
Zv = []
Zvk = []
a = []
ak = {}
if rank != 0:
    key_share_cost_0 = time.time()
    for xt_node in range(1, K+1):
        if rank != 0:
            local_group = [i for i in groups if rank in i]
            for group in local_group:
                if (rank in group) and (xt_node in group):
                    if rank == xt_node == group[0]:
                        gr = list(group)
                        gr.remove(rank)
                        Zv = np.array([random.randint(1, q - 1) for i in range(0, L)])
                        Zvk.append(Zv)
                        a = random.sample(range(1, K), U)
                        ak[tuple(group)] = a
                        for user in gr:
                            comm.send([Zv, a], dest=user, tag=1)
                    if rank != group[0] and xt_node == group[0]:
                        data = comm.recv(source=xt_node, tag=1)
                        Zvk.append(data[0])
                        ak[tuple(group)] = data[1]
    key_share_cost = time.time() - key_share_cost_0

Ak = comm.gather(ak, root=0)
U1 = []
A = {}
if rank == 0:
    Ak.remove(Ak[0])
    for i in range(len(Ak)):
        for key in Ak[i]:
            if key not in A:
                A[key] = Ak[i].get(key)

A = comm.bcast(A, root=0)
KS = 0
a_null = []
if rank != 0:
    local_n = [i for i in groups if rank not in i]
    ak_n = []
    for grr in local_n:
        ak_n.append(A.get(tuple(grr)))
    a_null = np.array(list(Matrix(np.array(ak_n)).nullspace()[0]))
    KS = 1000 * key_share_cost

Cost = comm.gather(KS, root=0)
if rank == 0:
    U1 = list(org_group)
    Cost = [i for i in Cost if i != 0]
    print('Average key share cost is: ', round((sum(Cost) / len(Cost)), 4))

t00 = time.time()
U1 = comm.bcast(U1, root=0)
t0 = time.time() - t00

'Generate the encoding message and send it to the server'
Xi = []
if rank != 0:
    local_group = [i for i in groups if rank in i]
    if rank in U1:
        for i in range(len(Zvk)):
            if L % U != 0:
                Zvk[i] = np.array(list(Zvk[i]) + [0 for _ in range(U - L % U)])
            else:
                pass
            Zvk[i] = np.split(Zvk[i], U)
        ak_g = []
        for item in local_group:
            ak_g.append(ak.get(tuple(item)))
        aa = []
        for aki in ak_g:
            he = 0
            for i in range(len(aki)):
                he += aki[i] * a_null[i]
            aa.append(he)
        if L % U != 0:
            Wi = list(np.arange(1, L + 1, dtype=int)) + [0 for _ in range(U - L % U)]
            Wi = np.split(np.array(Wi), U)
        else:
            Wi = np.split(np.arange(1, L + 1, dtype=int), U)
        for j in range(len(Wi)):
            he = np.zeros((len(Wi[0]),), dtype=int)
            for k in range(len(Zvk)):
                he += Zvk[k][j] * ak_g[k][j]
            Xi.append(Wi[j] + he)
    else:
        pass

t10 = time.time()
XU = comm.gather(Xi, root=0)
t1 = time.time() - t10
comm.barrier()

'Server compute the sum of encoding message from users'
U2 = []
if rank == 0:
    XU.pop(0)
    X_sum = []
    for i in range(len(XU[0])):
        SUM = np.zeros((len(XU[0][0]), ), dtype=int)
        for j in range(len(XU)):
            SUM += XU[j][i]
        X_sum.append(SUM)
    U2 = list(U1)
    for u in random.sample(org_group, K-U):
        U2.remove(u)

t20 = time.time()
U2 = comm.bcast(U2, root=0)
t2 = time.time() - t20
comm.barrier()

'Compute the encoding coefficients for alive users and send it to server'
F = []
Zi = []
if rank != 0:
    if rank in U2:
        Zs = []
        for item in Zvk:
            Zs.append(sum(item))
        Zi = np.zeros((len(Zs[0]), ))
        for j in range(len(aa)):
            Zi = Zi + aa[j]*Zs[j]
    else:
        a_null = []

t30 = time.time()
A_null = comm.gather(a_null, root=0)
F = comm.gather(Zi, root=0)
t3 = time.time() - t30
comm.barrier()

'Server decoding and get the sum of messages from users alive for Round 1'
if rank == 0:
    F.remove(F[0])
    F = [list(i) for i in F if len(i) != 0]
    A_null.remove(A_null[0])
    A_null = [list(i) for i in A_null if len(i) != 0]
    Fs = np.linalg.solve(np.array(A_null, dtype=int), np.array(F, dtype=int))
    W_sum = []
    for i in range(len(X_sum)):
        W_sum += list(X_sum[i] - Fs[i])

Tm = 0
T = []
if rank != 0:
    if rank in U2:
        T = [1000*t0+1000*t1, 1000*t2+1000*t3]
        Tm = 1000 * (t0 + t1 + t2 + t3)

Tm = comm.gather(Tm, root=0)
Tc = comm.gather(T, root=0)
if rank == 0:
    Tm = [i for i in Tm if i != 0]
    Tc = [i for i in Tc if len(i) != 0]
    Comm = []
    for j in range(len(Tc[0])):
        he = 0
        for i in range(len(Tc)):
            he += Tc[i][j]
        Comm.append(round(he / len(Tc), 4))
    print(rank, 'Average total comm time: ', round(sum(Tm)/len(Tm), 4))
    print(rank, 'Average comm time for each Round: ', Comm)
