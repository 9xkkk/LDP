import math
import numpy as np
from scipy.optimize import minimize

def optimize_probabilities(log_epsilon_values, coefficients):
    def objective_function(probabilities):
        probabilities_x = probabilities[:len(coefficients)]
        probabilities_y = probabilities[len(coefficients):]

        term_sum = 0
        for i in range(len(coefficients)):
            term_sum += coefficients[i] * probabilities_x[i] * (1 - probabilities_y[i]) / ((probabilities_x[i] - probabilities_y[i]) ** 2)
        return term_sum

    def constraint_function(probabilities, i, j):
        epsilon_i, epsilon_j = log_epsilon_values[i], log_epsilon_values[j]
        min_epsilon = min(epsilon_i, epsilon_j)
        return (probabilities[i] * (1 - probabilities[j + len(coefficients)])) / (probabilities[i + len(coefficients)] * (1 - probabilities[j])) - math.exp(min_epsilon)

    def max_constraint_function(probabilities, i):
        return 1 - probabilities[i] - probabilities[i + len(coefficients)]

    constraints = []
    for i in range(len(coefficients)):
        for j in range(len(coefficients)):
            constraints.append({'type': 'ineq', 'fun': lambda probabilities, i=i, j=j: constraint_function(probabilities, i, j)})
        constraints.append({'type': 'ineq', 'fun': lambda probabilities, i=i: probabilities[i + len(coefficients)] - probabilities[i]})
        constraints.append({'type': 'ineq', 'fun': lambda probabilities, i=i: max_constraint_function(probabilities, i)})

    # 初始化概率数组，前一半大于0.5小于1，后一半小于0.5大于0
    initial_probabilities_x = np.random.uniform(0.5, 1, len(coefficients))
    initial_probabilities_y = np.random.uniform(0, 0.5, len(coefficients))
    initial_probabilities = np.concatenate((initial_probabilities_x, initial_probabilities_y))

    result = minimize(objective_function, initial_probabilities, method='SLSQP', bounds=[(0, 1)] * len(initial_probabilities), constraints=constraints)

    if not result.success:
        print("Optimization failed:", result.message)
        return None
    return result.x

if __name__ == '__main__':
    log_epsilon_values = [math.log(4), math.log(6)]
    coefficients = [1, 4]
    probabilities = optimize_probabilities(log_epsilon_values, coefficients)
    print(probabilities)
