import numpy as np
from time import time
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import plot_the_points as ptp
# import torch
import os
import get_eigval as ge


#open the document
def read_txt(x):
    with open(x,'r') as file:
        M=[]
        for line in file:
            data=line.strip(' ').split()
            M.append([float(i) for i in data])
        return np.array(M)

M_1 = read_txt("C:\\Users\\xiaohong\\Desktop\\流形学习项目\\DataForManifoldofOneDReactor\\manipower.txt")
M_2 = read_txt("C:\\Users\\xiaohong\\Desktop\\流形学习项目\\DataForManifoldofOneDReactor\\manithermalflux1.txt")
M_3 = read_txt("C:\\Users\\xiaohong\\Desktop\\流形学习项目\\DataForManifoldofOneDReactor\\manithermalflux2.txt")



def time_costing(func):
    def core():
        start = time()
        func()
        print('time costing:', time() - start)
    return core

def accepter_et_plot_2d():
    x, U, S, VT, M=ge.points_and_svd_2d(1000,100)
    #第1234组
    for i in range(1,5):
        ptp.draw_eig_2d(x, U[:,5 * (i - 1)+1:5 * i+1], i)

#svd the powerall
def svd_the_data(M,r):
    N = M.shape[0]
    # print(M[:,1:].shape)
    # ptp.draw_eigvals(S)

    U, S, VT = np.linalg.svd(M)
    tau = (S[r]+S[r+1])/2
    # traditional mathod to find all those above 0.90
    cdS = np.cumsum(S) / np.sum(S)  # Cumulative energy

    plt.rcParams['font.family'] = 'MicroSoft YaHei'
    fig1, ax1 = plt.subplots(1)
    ax1.semilogy(S, '-o', color='k', lw=2)
    ax1.semilogy(np.diag(S[:(r + 1)]), 'o', color='r', lw=2)
    plt.ylabel('$\sigma$ take lg ',rotation = 45)
    ax1.plot([tau for i in range(M.shape[1])], '--', color='b', lw=2)
    plt.title('Find the right r for truncation')
    ax1.grid()
    # fig2, ax2 = plt.subplots(122)
    # ax2.plot(np.cumsum(S)/np.sum(S))
    # plt.title('the singular value accumulated')
    plt.show()
    plt.close()


def svd_reduced_form(M,r):
    U, S, VT = np.linalg.svd(M)
    # traditional mathod to find all those above 0.90
    Mr = U[:, :(r + 1)] @ np.diag(S[:(r + 1)]) @ VT[:(r + 1), :]
    return Mr


def pca_it(X,r):
    ## solve the probleme by using pca
    #compute mean
    Xavg = np.mean(X,axis=1)
    #mean_subtracted data
    Xc = np.array([[*Xavg] for i in range(X.shape[1])]).T
    B = X - Xc
    #find principal components(svd)
    U,S,VT = np.linalg.svd(B/np.sqrt(X.shape[1]-1),full_matrices=False)
    Xr = U[:, :(r + 1)] @ np.diag(S[:(r + 1)]) @ VT[:(r + 1), :]
    return Xr * (np.sqrt(X.shape[1]-1)) + Xc


def pca_reduced_form(X,r):
    ## solve the probleme by using pca
    #compute mean
    Xavg = np.mean(X,axis=1)
    #mean_subtracted data
    Xc = np.array([[*Xavg] for i in range(X.shape[1])]).T
    B = X - Xc
    #covariant matrix
    C = B.T @ B
    eig_val, eig_vec = np.linalg.eig(C/(X.shape[1]-1))
    eig_pairs = [(np.abs(eig_val[i]), eig_vec[:, i]) for i in range(X.shape[1])]
    # sort eig_vec based on eig_val from highest to lowest
    eig_pairs.sort(reverse=True)
    # select the top r eig_vec
    Sr = [np.sqrt(ele[0]) for ele in eig_pairs[:r]]
    Vr = [ele[1] for ele in eig_pairs[:r]]
    Ur = []
    for i in range(r):
        Ur.append(X @ np.array(Vr[i] / Sr[i]))
    Xr = np.array(Ur).T @ np.diag(Sr) @ np.array(Vr)
    return Xr + Xc



def accepter_et_plot_3d():
    x, U, S, VT, M=ge.points_and_svd_3d(1000,100,100)
    #第1234组
    #[...,5*i,5 * (i - 1):5 * i]
    for i in range(1,5):
        #改成了2d
        ptp.draw_eig_2d(x, U[:,5 * (i - 1):5 * i] , i)
    # print(U.shape)

def draw_eigvals():
    x, U, S, VT, M = ge.points_and_svd_2d(1000, 100)
    return ptp.draw_eigvals(S)





#control space

##plot the approximated
# M_1 = ge.svd_approximate_it(M_1,3)
# M_2 = ge.svd_approximate_it(M_2,3)
# M_3 = ge.svd_approximate_it(M_3,3)
# ptp.draw_the_oringinal(M_1,M_2,M_3)
# print(Mr_1.shape)

#plot the original
@time_costing
def original():
    ptp.draw_it(M_1, M_2, M_3)
@time_costing
def svd():
    ptp.draw_it(svd_reduced_form(M_1,5),svd_reduced_form(M_2,6),svd_reduced_form(M_3,6))
@time_costing
def pca():
    ptp.draw_it(pca_it(M_1,5),pca_it(M_2,6),pca_it(M_3,6))
@time_costing
def pca_r():
    ptp.draw_it(pca_reduced_form(M_1, 5), pca_reduced_form(M_2, 6), pca_reduced_form(M_3, 6))

# if __name__ == '__main__':
# original()
# svd()
# pca()
# pca_r()



# pca_reduced_form(M_1,5)

# print(timeit.timeit('original()',setup='from main_operater import original',number=1,repeat = 1))
# print(timeit.timeit('svd()',setup='from main_operater import svd',number=1))
# print(timeit.timeit('pca()',setup='from main_operater import pca',number=1))
# accepter_et_plot_2d()
# svd_the_data(M_1,4)
# svd_the_data(M_2,5)
# svd_the_data(M_3,5)
# draw_eigvals()
accepter_et_plot_2d()
# accepter_et_plot_3d()

# draw_eigvals()
# print(S)


















# def svd_it():
#     x_values = np.linspace(0, 1, num1)
#     u1_values = np.linspace(0.1, 10, num2)
#     # u2_values = np.linspace(0.1, 10,num3)
#     M1 =np.array(x_values)
#     M2 = np.array(u1_values)
#     # M3 = np.array(u2_values)
#     M1 =np.expand_dims(M1,axis=0)
#     M2 =np.expand_dims(M2,axis=1)
#     # M3 =np.expand_dims(M3,axis=2)
#     M=torch.from_numpy(f(M1,M2))
#     # print(M)
#
#     U,S,V=torch.svd(M)
#     return U,V