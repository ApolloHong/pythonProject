import numpy as np
import torch
# import main_operater as mp
# import sys
# print(sys.path)
#
# def svd(*kw):
#     # Z = np.mat(kw)
#     #U为axis=0,V为axis=1的分解
#     U, S, V = torch.linalg.svd(kw)
#     print(U.shape)
#     print(V.shape)


def f1(*kw):
    return 1 / np.sqrt(np.sin(kw[0] * kw[1]) ** 2 + 10)
def f2(*kw):
    return kw[0]**10*pow(np.e,(-11)*kw[1]*kw[0])
def f3(*kw):
    return kw[1]*(np.sin(kw[0])/1+kw[0])
def f4(*kw):
    return 1 / np.sqrt(np.sin(kw[0] * kw[1]) ** 2 + 10*kw[2])


def points_and_svd_2d(*args):
    x_values = np.linspace(0, 1, args[0])
    u1_values = np.linspace(0.1, 10, args[1])
    M1 = x_values.reshape((len(x_values),1))
    M2 = u1_values.reshape((1,len(u1_values)))
    M=f1(M1,M2)
    U,S,VT= np.linalg.svd(M)
    return x_values, U , S , VT, M

def svd_approximate_it(M,r):
    U,S,VT = np.linalg.svd(M)
    return U[:,:(r + 1)] @ np.diag(S[:(r + 1)]) @ VT[:(r + 1),:]

def points_and_svd_3d(*args):
    x_values = np.linspace(0, 1, args[0])
    u1_values = np.linspace(0.1, 10, args[1])
    u2_values = np.linspace(0.1, 10,args[2])
    M1 = x_values.reshape((len(x_values),1,1))
    M2 = u1_values.reshape((1,len(u1_values),1))
    M3 = u2_values.reshape((1,1,len(u2_values)))
    M = f4(M1, M2, M3)
    U, S, V = np.linalg.svd(M)
    return x_values, U , S , V, M


def ordre():

    pass


