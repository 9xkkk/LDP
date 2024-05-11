# min {|x-x1|+|y-y1|+|x-x2|+|y-y2|+|x-x3|+|y-y3|}
# s.t. (x-x1)**2 + (y-y1)**2 <= r1**2
#      (x-x2)**2 + (y-y2)**2 <= r2**2
#      (x-x3)**2 + (y-y3)**2 <= r3**2
# 上述x1,x2,x3,y1,y2,y3，r1,r2,r3都已知，且**2代表平方的意思，请问上述最优化方程是否可以找到最优解，请给出用python求解的代码
from scipy.optimize import minimize
import numpy as np
from scipy.optimize import differential_evolution
from scipy import optimize
from matplotlib import pyplot as plt

c1 = [10,20,10]
c2 = [20,10,10]
c3 = [5,5,10]
c4 = [10,10,10]
x1 = c1[0]
y1=c1[1]
r1=c1[2]
x2 = c2[0]
y2=c2[1]
r2=c2[2]
x3 = c3[0]
y3=c3[1]
r3=c3[2]
x4 = c4[0]
y4 = c4[1]
r4 = c4[2]


def f(X):
    return (X[0] - x1)**2 + (X[1] - y1)**2 + (X[0] -x2)**2 + (X[1] - y2)**2 + (X[0] - x3)**2 + (X[1] - y3)**2 + (X[0] - x4)**2 + (X[1] - y4)**2
    # return abs(X[0] - x1) + abs(X[1] - y1) + abs(X[0] - x2) + abs(X[1] - y2) + abs(X[0] - x3) + abs(X[1] - y3)
def constraints1(X):
    return (X[0] - x1)**2 + (X[1] - y1)**2 - r1**2

def constraints2(X):
    return (X[0] -x2)**2 + (X[1] - y2)**2 - r2**2

def constraints3(X):
    return (X[0] - x3)**2 + (X[1] - y3)**2 - r3**2

def constraints4(X):
    return (X[0] - x4)**2 + (X[1] - y4)**2 - r4**2

constraints = [
    {'type': 'ineq', 'fun': constraints1},
    {'type': 'ineq', 'fun': constraints2},
    {'type': 'ineq', 'fun': constraints3},
    {'type': 'ineq', 'fun': constraints4}
]
x_start = optimize.brute(f, [(0, 20, 0.5), (0, 20, 0.5)], finish=None)
result = optimize.minimize(f, x_start, method='SLSQP', constraints=constraints)

print(result)
# 绘制圆形
def plot_circle(x, y, r, color):
    theta = np.linspace(0, 2*np.pi, 100)
    cx = x + r * np.cos(theta)
    cy = y + r * np.sin(theta)
    plt.plot(cx, cy, color)

# 绘制最优解点和圆形
def plot_solution(x, y):
    plt.scatter(x, y, color='red', label='Optimal Solution')
    plot_circle(x1, y1, r1, 'r')
    plot_circle(x2, y2, r2, 'g')
    plot_circle(x3, y3, r3, 'b')
    plot_circle(x4, y4, r4, 'y')

# 创建一个图形对象
fig, ax = plt.subplots()

# 绘制圆形
plot_circle(x1, y1, r1, 'r')
plot_circle(x2, y2, r2, 'g')
plot_circle(x3, y3, r3, 'b')
plot_circle(x4, y4, r4, 'y')

# 绘制最优解点
plot_solution(result.x[0], result.x[1])

# 设置坐标轴范围
ax.set_xlim([-30, 30])
ax.set_ylim([-30, 30])

# 添加标题和标签
plt.title('Three Circles and Optimal Solution')
plt.xlabel('X-axis')
plt.ylabel('Y-axis')

# 添加图例
plt.legend()

# 显示图形
plt.show()
