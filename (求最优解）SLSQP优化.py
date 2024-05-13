import math
import numpy as np
from scipy import optimize

def opt_0(Epsilon, m):
    def f(X):
        func = 0
        for i in range(len(m)):
            func += m[i] * X[i + len(m)] * (1 - X[i + len(m)]) / ((X[i] - X[i + len(m)]) ** 2)
        max_1 = []
        for i in range(len(m)):
            max_1.append((1 - X[i] - X[i + len(m)]) / (X[i] - X[i + len(m)]))
        max_1 = max(max_1)
        return func + max_1

    def constraint_func(X, i, j):
        epsilon_1, epsilon_2 = Epsilon[i], Epsilon[j]
        epsilon = min(epsilon_1, epsilon_2)
        return math.e ** epsilon - (X[i] * (1 - X[j + len(m)])) / (X[i + len(m)] * (1 - X[j]))

    def constraint_func2(X, i):
        return X[i] - X[i+len(m)]

    def constraint_func3(X, i):
        return X[i] - 0.5

    def constraint_func4(X, i):
        return 0.5 - X[i+len(m)]

    constraints = []
    for i in range(len(m)):
        for j in range(len(m)):
            # 使用偏函数来捕获 i 和 j 的当前值
            constraints.append({'type': 'ineq', 'fun': lambda X, i=i, j=j: constraint_func(X, i, j)})
        constraints.append({'type': 'ineq', 'fun': lambda X, i=i: constraint_func2(X, i)})
        constraints.append({'type': 'ineq', 'fun': lambda X, i=i: constraint_func3(X, i)})
        constraints.append({'type': 'ineq', 'fun': lambda X, i=i: constraint_func4(X, i)})

    # 初始化 x_start，确保长度是 2 * len(m)
    initial_probabilities_x = np.random.uniform(0.5, 1, len(m))
    initial_probabilities_y = np.random.uniform(0, 0.5, len(m))
    initial_probabilities = np.concatenate((initial_probabilities_x, initial_probabilities_y))

    result = optimize.minimize(f, initial_probabilities, method='SLSQP', bounds=[(0, 1)]*len(initial_probabilities), constraints=constraints)
    if not result.success:
        print("Optimization failed:", result.message)
        return None
    X_optimal = result.x
    return X_optimal


if __name__ == '__main__':
    Epsilon = [np.log(4), np.log(6)]
    m = [1, 4]
    probabilities = opt_0(Epsilon, m)
    print(probabilities)
