import numpy as np
from matplotlib import pyplot as plt
# import pytorch as torch
import smooth_et_test as set


import get_eigval as ge

G=[0.5,0.6,0.7,0.8,0.9,1.0,1.1,1.2,1.4,1.6,1.8,2.0]

#获取按照函数值大小渐变的图像
def get_the_colors_various(y,k) -> int :
    return (k-min(y))/(max(y)-min(y))


def draw_it(M_1,M_2,M_3):
    plt.rcParams['figure.figsize'] = [16, 8]
    fig = plt.figure()
    # plt.title('Data from the reduced matrix')
    # np.random.seed(1)#生成种子
    ax1 = fig.add_subplot(131)
    for i in range(len(M_1[0,:])-1):
        ax1.plot(M_1[:,0], M_1[:, i+1],label=f'D1 = {0.01*(i)}',color=plt.cm.jet(M_1[i, 0]+0.22))
    plt.legend(loc='lower left',fontsize=8)
    plt.text(15,1.5*0.9,'P (D1)',ha='center',fontsize=15)
    plt.xlabel('x (in cm)')
    plt.xticks(np.linspace(0, 30, 6))
    plt.yticks(np.linspace(0, 1.5, 4))
    ax2 = fig.add_subplot(132)
    for i in range(len(M_2[0,:])-1):
        ax2.plot(M_2[:, 0], M_2[:, i+1],label=f'D1 = {0.01*(i)}',color=plt.cm.jet(M_2[i, 0]+0.2))
    plt.legend(loc='lower left',fontsize=8)
    plt.text(15, 35 * 0.9, '$\phi_1$(D1)', ha='center',fontsize=15)
    plt.xlabel('x (in cm)')
    plt.xticks(np.linspace(0, 30, 6))
    plt.yticks(np.linspace(0, 35, 8))
    ax3 = fig.add_subplot(133)
    for i in range(len(M_3[0,:])-1):
        ax3.plot(M_3[:, 0], M_3[:, i+1],label=f'D1 = {0.01*(i)}',color=plt.cm.jet(M_3[i, 0]+0.25))
    plt.legend(loc='lower left',fontsize=8)
    plt.text(15, 7 * 0.9, '$\phi_2$(D1)', ha='center',fontsize=15)
    plt.xlabel('x (in cm)')
    plt.xticks(np.linspace(0, 30, 6))
    plt.yticks(np.linspace(0, 7, 8))
    # plt.show()
    # plt.close()


def draw_eigvals(y):
    fig = plt.figure()
    plt.rcParams['font.family'] = 'MicroSoft YaHei'
    cdS=np.cumsum(np.diag(y)) / np.sum(np.diag(y))
    ax1 = fig.add_subplot(121)
    r_truncation = np.min(np.where(cdS > 0.70))
    ax1.semilogy(r_truncation, lw=0.7 ,marker='.',color='r')
    ax1.semilogy(y[r_truncation+1:] ,lw=0.7 ,marker='.' ,color='b')
    plt.ylabel('奇异值取lg')
    plt.title('单个奇异值随索引值的变化')
    ax2 = fig.add_subplot(122)
    ax2.plot(np.cumsum(np.diag(y))/np.sum(np.diag(y)),)
    plt.title('奇异值随索引值的积累')
    plt.show()
    plt.close()





#2d有滤波
def draw_eig_2d(x, y, i):
    plt.rcParams['font.family'] = 'MicroSoft YaHei'  # 设置字体，默认字体显示不了中文
    # for j in range(5):
    #     y[j] = set.smooth_it(y[j])
    fig, (ax1)=plt.subplots(1,1,sharex=True,figsize=(10, 4.5))
    ax1.plot(x,y[:,0], linestyle=':', lw='1',  color='b', label=f'第{5 * (i - 1) + 1}个左奇异向量')
    ax1.plot(x,y[:,1], linestyle='--', lw='1', color='k', label=f'第{5 * (i - 1) + 2}个左奇异向量')
    ax1.plot(x,y[:,2], linestyle='-.', lw='1', color='g', label=f'第{5 * (i - 1) + 3}个左奇异向量')
    ax1.plot(x,y[:,3], linestyle='-', lw='1',color='y', label=f'第{5 * (i - 1) + 4}个左奇异向量')
    ax1.plot(x,y[:,4], linestyle='-.', lw='1', color='r', label=f'第{5 * (i - 1) + 5}个左奇异向量')
    plt.title(f"第{5 * (i - 1) + 1}到第{5 * i}维左奇异向量")
    plt.xlabel('x')
    plt.ylabel('y', rotation=90)
    plt.legend(loc='upper left')
    plt.show()
    plt.close()

#2d无滤波
def draw_eig_2d2(x, y, i):
    plt.figure(figsize=(10, 4.5))  # 设置画布大小,默认画布序号为零
    plt.rcParams['font.family'] = 'MicroSoft YaHei'  # 设置字体，默认字体显示不了中文
    plt.plot(x,y[0], linestyle='--', lw='1',  color='b', label=f'第{5 * (i - 1) + 1}个特征值')
    plt.plot(x,y[1], linestyle=':', lw='1', color='k', label=f'第{5 * (i - 1) + 2}个特征值')
    plt.plot(x,y[2], linestyle='-', lw='1', color='g', label=f'第{5 * (i - 1) + 3}个特征值')
    plt.plot(x,y[3], linestyle='-', lw='1',color='y', label=f'第{5 * (i - 1) + 4}个特征值')
    plt.plot(x,y[4], linestyle='-.', lw='1', color='r', label=f'第{5 * (i - 1) + 5}个特征值')
    plt.xlim((0, 1))
    plt.ylim((-0.1, 0.1))
    plt.xticks(np.linspace(0, 1, 21))
    plt.yticks(np.linspace(-0.1, 0.1, 11))
    plt.title(f"第{5 * (i - 1) + 1}到第{5 * i}维奇异值")
    plt.xlabel('x')
    plt.ylabel('y', rotation=90)
    plt.legend(loc='upper left')
    plt.show()
    plt.close()

#3d有滤波
def draw_eig_3d(x, y, z, i):
    plt.figure(figsize=(10, 4.5))  # 设置画布大小,默认画布序号为零
    plt.rcParams['font.family'] = 'MicroSoft YaHei'  # 设置字体，默认字体显示不了中文
    for j in range(5):
        y[j] = set.smooth_it(y[j])
    # fig, (axe1,axe2,axe3,axe4,axe5)=plt.subplots(5,1,sharex=True)
    plt.plot(x,y[0],z, linestyle=':', lw='1',  color='b', label=f'第{5 * (i - 1) + 1}个特征值')
    plt.plot(x,y[1],z, linestyle='--', lw='1', color='k', label=f'第{5 * (i - 1) + 2}个特征值')
    plt.plot(x,y[2],z, linestyle='-.', lw='1', color='g', label=f'第{5 * (i - 1) + 3}个特征值')
    plt.plot(x,y[3],z, linestyle='-', lw='1',color='y', label=f'第{5 * (i - 1) + 4}个特征值')
    plt.plot(x,y[4],z, linestyle='-.', lw='1', color='r', label=f'第{5 * (i - 1) + 5}个特征值')
    plt.xlim((0, 1))
    plt.ylim((-0.1, 0.1))
    plt.xticks(np.linspace(0, 1, 21))
    plt.yticks(np.linspace(-0.1, 0.1, 11))
    plt.title(f"第{5 * (i - 1) + 1}到第{5 * i}维奇异值")
    plt.xlabel('x')
    plt.ylabel('y', rotation=90)
    plt.legend(loc='upper left')
    plt.show()
    plt.close()
