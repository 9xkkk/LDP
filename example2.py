##本程序example用于实现GRR,SUE,OUE,OLH,THE,HRR
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
    )
)
original_freq = list(Counter(data).values())  # True frequencies of the dataset

# Parameters for experiment
epsilon = 1
d = 8
is_the = True
is_oue = True
is_olh = True

# Optimal Local Hashing (OLH)
client_olh = LHClient(epsilon=epsilon, d=d, use_olh=is_olh)
server_olh = LHServer(epsilon=epsilon, d=d, use_olh=is_olh)

# Optimal Unary Encoding (OUE)
client_oue = UEClient(epsilon=epsilon, d=d, use_oue=is_oue)
server_oue = UEServer(epsilon=epsilon, d=d, use_oue=is_oue)

# Threshold Histogram Encoding (THE)
client_the = HEClient(epsilon=epsilon, d=d)
server_the = HEServer(epsilon=epsilon, d=d, use_the=is_the)

# Hadamard Random Response (HRR)
server_hr = HadamardResponseServer(epsilon=epsilon, d=d)
client_hr = HadamardResponseClient(epsilon=epsilon, d=d, hash_funcs=server_hr.get_hash_funcs())


# Simulate client-side privatisation and server-side aggregation
for item in data:
    priv_olh_data = client_olh.privatise(item)
    priv_oue_data = client_oue.privatise(item)
    priv_the_data = client_the.privatise(item)
    priv_hr_data = client_hr.privatise(item)

    server_olh.aggregate(priv_olh_data)
    server_oue.aggregate(priv_oue_data)
    server_the.aggregate(priv_the_data)
    server_hr.aggregate(priv_hr_data)

# Simulate server-side estimation
oue_estimates = []
olh_estimates = []
the_estimates = []
hr_estimates = []
mse_arr = np.zeros(5)

for i in range(0, d):
    olh_estimates.append(round(server_olh.estimate(i + 1)))
    oue_estimates.append(round(server_oue.estimate(i + 1)))
    the_estimates.append(round(server_the.estimate(i + 1)))
    hr_estimates.append(round(server_hr.estimate(i + 1)))

# Calculate mse of server-side estimation
for i in range(0,d):
    mse_arr[0] += (olh_estimates[i] - original_freq[i]) ** 2
    mse_arr[1] += (oue_estimates[i] - original_freq[i]) ** 2
    mse_arr[2] += (the_estimates[i] - original_freq[i]) ** 2
    mse_arr[3] += (hr_estimates[i] - original_freq[i]) ** 2

mse_arr = mse_arr / d

print("\n")
print(
    "Experiment run on a dataset of size",
    len(data),
    "with d=",
    d,
    "and epsilon=",
    epsilon,
    "\n",
)
print("Optimised Local Hashing (OLH) Variance: ", mse_arr[0])
print("Optimised Unary Encoding (OUE) Variance: ", mse_arr[1])
print("Threshold Histogram Encoding (THE) Variance: ", mse_arr[2])
print("Hadamard random response (HR) Variance: ", mse_arr[3])
print(sum(hr_estimates))

print("\n")
print("Original Frequencies:", original_freq)
print("OLH Estimates:", olh_estimates)
print("OUE Estimates:", oue_estimates)
print("THE Estimates:", the_estimates)
print("HR Estimates:", hr_estimates)
print("Note: we round estimates to the nearest integer")

# Plotting
plt.figure(figsize=(10,6))
plt.plot(range(1, d + 1), original_freq, label='Original Frequencies', marker='o')
plt.plot(range(1, d + 1), olh_estimates, label='OLH Estimates', marker='x')
plt.plot(range(1, d + 1), oue_estimates, label='OUE Estimates', marker='^')
plt.plot(range(1, d + 1), the_estimates, label='THE Estimates', marker='s')
plt.plot(range(1, d + 1), hr_estimates, label='HR Estimates', marker='d')

plt.title("Frequency Estimates vs Original Frequencies")
plt.xlabel("Item Index")
plt.ylabel("Frequency")
plt.legend()
plt.grid(True)
plt.show()