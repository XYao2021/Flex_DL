import random
import numpy as np
import sklearn
from functions import *
from torchvision import datasets
from torchvision import transforms
import sys
import os
import operator
import math
import heapq
import torch
from scipy.optimize import minimize, minimize_scalar, LinearConstraint
from scipy.stats import chi2

V = 0.02
W = 1.0
upper_t = W
n = 10
bound = (0, 1)
# q_t = np.random.random(n)
# q_avg = np.array([0.6 for _ in range(len(q_t))])
q_avg = 0.25
# np.random.seed(42)
# print(np.random.chisquare(df=2))
# alpha = np.random.uniform(low=0, high=1, size=1).item()
# print(alpha)
# bounds = [bound for _ in range(len(q_t))]
# constraint = {'type': 'ineq', 'constraint': lambda x: q_avg-x}

# def objective_function(x):
#     return V/x + upper_t*(alpha*x - q_avg)

# optimization = minimize(objective_function, q_t, bounds=bound, constraints=constraint)
# print(optimization)
# for i in range(30):
#     alpha = np.random.uniform(low=0, high=1, size=1).item()
#     print('alpha: ', alpha)
#     if upper_t == 0:
#         q_t = 1/3
#     else:
#         q_t = min(1, math.sqrt(V/(alpha*upper_t)))
#     print('q_t: ', q_t)
#     # print('q_t: ', min(1, math.sqrt(V/(alpha*upper_t))), '\n')
#     upper_t = max(0, upper_t + alpha*q_t - q_avg)
#     print('next upper_t: ', upper_t, '\n')

# print(random.random())
# upper_t = max(0, upper_t*(optimization.x - q_avg))
#
# q_t = optimization.x

# b_t = np.random.random((10, 2, 2))
# v_t = np.random.random((10, 2, 2))
# phi_avg = 0.5
# PHI = W
#
# def objective_function(x):
#     return V*np.linalg.norm(b_t - x)**2 + PHI*(x - phi_avg)
#
# # optimization = minimize_scalar(objective_function)
# optimization = minimize(objective_function, v_t, method='SLSQP')
# print(optimization)

# train_data = datasets.FashionMNIST(root="data",
#                                    train=True,
#                                    transform=transforms.ToTensor(),
#                                    target_transform=None,
#                                    download=True)
# test_data = datasets.MNIST(root="data",
#                            train=False,
#                            transform=transforms.ToTensor(),
#                            target_transform=None,
#                            download=True)

# org_targets = [i for i in range(len(train_data.classes))]

# print(train_data[0][0].shape)
# data = split_data(sample, train_data)
# for item in data:
#     print(len(item))

# np.random.seed(42)
# print(torch.distributions.chi2.Chi2(df=2).sample())
# print(np.random.uniform(low=0, high=1, size=1).item())

# a = np.array([0, 1, 2, 0, 0, 1, 1, 0])
# print(np.linalg.norm(a, ord=0))
# a = torch.Tensor([1, 2, 3, 4, 5, 6, 10, 11, 12, 13, 14, 15])
# b = torch.Tensor([3, 4, 5])

# print(a.reshape([2, 6]))
# stack = torch.stack([a, b], -1)
# print(stack)

# unstack = torch.unbind(stack, -1)
# print(unstack[0].int(), unstack[1][unstack[0].int()[0]])
# print(b[(a == 4).nonzero().item()])
a = torch.tensor([])
b = torch.tensor([])

print(len(a))
