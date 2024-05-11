import approximate_ldp.frequency_oracles
import pure_ldp.frequency_oracles
from approximate_ldp.frequency_oracles import *
from pure_ldp.frequency_oracles import *

import numpy as np
from collections import Counter
import matplotlib.pyplot as plt

# Super simple synthetic dataset
data = np.concatenate(
    (
        [1] * 8000,
        [2] * 4000,
        [3] * 1000,
        [4] * 500,
        [5] * 1000,
        [6] * 1800,
        [7] * 2000,
        [8] * 300,
        [9] * 400,
        [10] * 1000,
        [11] * 1000,
        [12] * 1000,
        [13] * 1000,
        [14] * 1000,
        [15] * 1000,
        [16] * 1000,
        [17] * 1000,
        [18] * 1000,
        [19] * 500,
        [20] * 500,
    )
)
original_freq = list(Counter(data).values())

# Parameters for experiment
epsilon = 0.4
deta = 0.001
d = 20

is_oue = True

## Direct Encoding
# Pure-ldp Direct Encoding (DE)
pure_client_de = pure_ldp.frequency_oracles.DEClient(epsilon=epsilon, d=d)
pure_server_de = pure_ldp.frequency_oracles.DEServer(epsilon=epsilon, d=d)
# Approximate-ldp Direct Encoding (DE)
approximate_client_de = approximate_ldp.frequency_oracles.DEClient(epsilon=epsilon, deta=deta, d=d)
approximate_server_de = approximate_ldp.frequency_oracles.DEServer(epsilon=epsilon, deta=deta, d=d)

## Unary Encoding
# Pure-ldp Unary Encoding (SUE and OUE)
pure_client_sue = pure_ldp.frequency_oracles.UEClient(epsilon=epsilon, d=d)
pure_server_sue = pure_ldp.frequency_oracles.UEServer(epsilon=epsilon, d=d)
pure_client_oue = pure_ldp.frequency_oracles.UEClient(epsilon=epsilon, d=d, use_oue=is_oue)
pure_server_oue = pure_ldp.frequency_oracles.UEServer(epsilon=epsilon, d=d, use_oue=is_oue)
# Approximate Unary Encoding (SUE and OUE)
approximate_client_sue = approximate_ldp.frequency_oracles.UEClient(epsilon=epsilon, deta=deta, d=d)
approximate_server_sue = approximate_ldp.frequency_oracles.UEServer(epsilon=epsilon, deta=deta, d=d)
approximate_client_oue = approximate_ldp.frequency_oracles.UEClient(epsilon=epsilon, deta=deta, d=d, use_oue=is_oue)
approximate_server_oue = approximate_ldp.frequency_oracles.UEServer(epsilon=epsilon, deta=deta, d=d, use_oue=is_oue)

# Simulate client-side privatisation + server-side aggregation
for item in data:
    priv_pure_de_data = pure_client_de.privatise(item)
    priv_approximate_de_data = approximate_client_de.privatise(item)
    priv_pure_sue_data = pure_client_sue.privatise(item)
    priv_pure_oue_data = pure_client_oue.privatise(item)
    priv_approximate_sue_data = approximate_client_sue.privatise(item)
    priv_approximate_oue_data = approximate_client_oue.privatise(item)

    pure_server_de.aggregate(priv_pure_de_data)
    approximate_server_de.aggregate(priv_approximate_de_data)
    pure_server_sue.aggregate(priv_pure_sue_data)
    pure_server_oue.aggregate(priv_pure_oue_data)
    approximate_server_sue.aggregate(priv_approximate_sue_data)
    approximate_server_oue.aggregate(priv_approximate_oue_data)

# Simulate server-side estimation
pure_de_estimates = []
approximate_de_estimates = []
pure_sue_estimates = []
pure_oue_estimates = []
approximate_sue_estimates = []
approximate_oue_estimates = []
mse_arr = np.zeros(6)

for i in range(0, d):
    pure_de_estimates.append(round(pure_server_de.estimate(i + 1)))
    approximate_de_estimates.append(round(approximate_server_de.estimate(i + 1)))
    pure_sue_estimates.append(round(pure_server_sue.estimate(i + 1)))
    pure_oue_estimates.append(round(pure_server_oue.estimate(i + 1)))
    approximate_sue_estimates.append(round(approximate_server_sue.estimate(i + 1)))
    approximate_oue_estimates.append(round(approximate_server_oue.estimate(i + 1)))


# ------------------------------ Experiment Output (calculating variance) -------------------------

for i in range(0, d):
    mse_arr[0] += (pure_de_estimates[i] - original_freq[i]) ** 2
    mse_arr[1] += (approximate_de_estimates[i] - original_freq[i]) ** 2
    mse_arr[2] += (pure_sue_estimates[i] - original_freq[i]) ** 2
    mse_arr[3] += (pure_oue_estimates[i] - original_freq[i]) ** 2
    mse_arr[4] += (approximate_sue_estimates[i] - original_freq[i]) ** 2
    mse_arr[5] += (approximate_oue_estimates[i] - original_freq[i]) ** 2
mse_arr = mse_arr / d

print("\n")
print(
    "Experiment run on a dataset of size",
    len(data),
    "with d=",
    d,
    "and epsilon=",
    epsilon,
    "and deta=",
    deta,
    "\n",
)
print("Pure Direct Encoding (PDE) Variance: ", mse_arr[0])
print("Approximate Direct Encoding (ADE) Variance: ", mse_arr[1])
print("Pure SUnary Encoding (SUE) Variance: ", mse_arr[2])
print("Pure OUnary Encoding (OUE) Variance: ", mse_arr[3])
print("Approximate SUnary Encoding (SUE) Variance: ", mse_arr[4])
print("Approximate OUnary Encoding (OUE) Variance: ", mse_arr[5])
print("\n")
print("Original Frequencies:", original_freq)
print("PDE Estimates:", pure_de_estimates)
print("ADE Estimates:", approximate_de_estimates)
print("Pure SUE Estimates:", pure_sue_estimates)
print("Pure OUE Estimates:", pure_oue_estimates)
print("Approximate SUE Estimates:", approximate_sue_estimates)
print("Approximate OUE Estimates:", approximate_oue_estimates)
print("Note: We round estimate to the nearest integer")

# Plotting
plt.figure(figsize=(10, 6))
plt.plot(range(1, d + 1), original_freq, label='Original Frequencies', marker='o')
plt.plot(range(1, d + 1), pure_de_estimates, label='Pure DE Estimates', marker='s')
plt.plot(range(1, d + 1), approximate_de_estimates, label='Approximate DE Estimates', marker='D')
# plt.plot(range(1, d + 1), pure_sue_estimates, label='Pure SUE Estimates', marker='p')
# plt.plot(range(1, d + 1), pure_oue_estimates, label='Pure OUE Estimates', marker='*')
# plt.plot(range(1, d + 1), approximate_sue_estimates, label='Approximate SUE Estimates', marker='v')
# plt.plot(range(1, d + 1), approximate_oue_estimates, label='Approximate OUE Estimates', marker='>')


plt.title("Frequency Estimates vs Original Frequencies")
plt.xlabel("Item Index")
plt.ylabel("Frequency")
plt.legend()
plt.grid(True)
plt.show()
