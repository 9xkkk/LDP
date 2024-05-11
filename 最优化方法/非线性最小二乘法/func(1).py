#-*- coding:utf-8 -*-
import numpy as np
import matplotlib.pyplot as plt
import math
import time
from scipy.linalg import LinAlgWarning
import warnings
import pulp
# from sympy import re
warnings.filterwarnings(action='ignore', category=LinAlgWarning, module='sklearn')
plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['axes.unicode_minus'] = False


def dis(x1, y1, x2, y2):
    return math.sqrt(pow(x1 - x2, 2) + pow(y1 - y2, 2))



#*********PLP求最优概率扰动矩阵*********#
def PLP(loc, sum, eps, delta = 1, disfunc = dis):
    '''
    个性化隐私预算且线性优化设计随机扰动矩阵
    '''
    # 初始化扰动矩阵perturbed matrix
    num_of_loc = len(loc)                           #离散点数量
    pm = [[0 for i in range(num_of_loc)] for i in range(num_of_loc)]                


    # 初始化约束矩阵，这是一个对称矩阵constraints matrix
    cm = [[0 for i in range(num_of_loc)] for i in range(num_of_loc)]             #这样初始化cm = [[0]*num_of_loc]*num_of_loc是不可以的

    for i in range(num_of_loc):
        for j in range(num_of_loc):
            cm[i][j] = min(loc[i]['epsilon'],loc[j]['epsilon']) * eps * disfunc(loc[i]['x'], loc[i]['y'], loc[j]['x'], loc[j]['y'])
    # print(cm)

    for k in range(num_of_loc):
        for i in range(num_of_loc):
            for j in range(num_of_loc):
                cm[i][j] = min(cm[i][j], cm[i][k] + cm[k][j])
    # 线性优化求概率扰动矩阵，转换成标准形式，定义系数矩阵c,A,b,Aeq,Beq,lb,ub,options
    
    # 1行n*n列
    clist = [0 for i in range(num_of_loc * num_of_loc)]
    for i in range(num_of_loc):
        for j in range(num_of_loc):
            clist[i*num_of_loc+j] = loc[i]['con'] / sum * disfunc(loc[i]['x'], loc[i]['y'], loc[j]['x'], loc[j]['y'])


    # n*n*(n-1)行n*n列
    Alist = [[0 for i in range(num_of_loc*num_of_loc)] for i in range(num_of_loc*num_of_loc*(num_of_loc-1))]
    row = 0
    for k in range(num_of_loc):
        for i in range(num_of_loc):
            for j in range(num_of_loc):
                if i != j:
                    # Alist[row][i*num_of_loc+k] = 1
                    # Alist[row][j*num_of_loc+k] = -pow(math.e,cm[i][j])
                    Alist[row][j*num_of_loc+k] = -1
                    Alist[row][i*num_of_loc+k] = pow(math.e,-1.0 * cm[i][j])
                    row = row + 1

    # n*n*(n-1)行1列，全为0
    blist = [0 for i in range(num_of_loc*num_of_loc*(num_of_loc-1))]

    # n行n*n列
    Aeqlist = [[0 for i in range(num_of_loc*num_of_loc)] for i in range(num_of_loc)]
    for i in range(num_of_loc):
        for j in range(num_of_loc):
            Aeqlist[i][i*num_of_loc+j] = 1

    # n行1列，全为1
    Beqlist = [1 for i in range(num_of_loc)]
  

    # 调用函数库，准备参数
    c = np.array(clist)
    A = np.array(Alist)
    b = np.array(blist)
    Aeq = np.array(Aeqlist)
    beq = np.array(Beqlist)

    #确定最小化问题，最大化只要把Min改成Max即可
    m = pulp.LpProblem(sense=pulp.LpMinimize)
    #定义变量列表
    x = [pulp.LpVariable(f'x{i}', lowBound=0, upBound=1, cat='Continuous') for i in range(num_of_loc*num_of_loc)]
    #定义目标函数，lpDot可以将两个列表的对应位相乘再加和
    m += pulp.lpDot(c, x)
    #设置不等式约束条件
    for i in range(len(A)):
        m += (pulp.lpDot(A[i], x) <= b[i])
    #设置等式约束条件
    for i in range(len(Aeq)):
        m += (pulp.lpDot(Aeq[i], x) == beq[i])
    #求解
    start = time.perf_counter()
    m.solve()
    total_time = time.perf_counter() - start
    #输出结果
    # print(f'优化结果：{pulp.value(m.objective)}')
    # print(f'参数取值：{[pulp.value(var) for var in x]}')

    varpro = [pulp.value(var) for var in x]
    index = 0
    res = []
    for i in range(num_of_loc):
        tmp = []
        for j in range(num_of_loc):
            tmp.append(varpro[index])
            index += 1
        res.append(tmp)

    return total_time, pulp.value(m.objective), pow(num_of_loc, 3) + pow(num_of_loc, 2) + num_of_loc, res



if __name__ == "__main__":
    print('PLP')