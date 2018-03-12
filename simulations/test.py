from scipy import integrate
import matplotlib.pyplot as plt
from process_simulation import ProcessesGenerator
import numpy as np

process_generator = ProcessesGenerator(set_seed = False)
W = process_generator.generate_brownian_motion(n_discr = 500, T = 1, m = 1)
time_steps = np.array([i / (500 - 1) for i in range(500)])

sigma_1_1 = 0.4
sigma_1_2 = -0.06
sigma_2_1 = 0.02
sigma_2_2 = -0.9

alpha = sigma_1_1 - sigma_2_1
beta = np.log((1 +sigma_1_2) / (1 + sigma_2_2))
kappa = 3

def f(s, t, w_t):
    return np.exp(-((alpha * (W[-1] - w_t) - beta * s) ** 2) / (2 * alpha * alpha * (1 - t)))

l1 = []
for i, t in enumerate(time_steps):
    v = np.exp(- kappa * (1 - t)) / np.sqrt((1 - t))
    l1.append(v * integrate.nquad(lambda s : f(s, t, W[i]), [lambda *args : [t, 1]])[0])


plt.figure(1)
plt.plot(time_steps, l1, label = "Z_num,1")
# plt.plot(time_steps, l2, label = "integral2")
# plt.plot(time_steps, l3, label = "integral3")
# plt.plot(time_steps, l4, label = "integral4")
# # v = np.exp(-2 * (1 - t)) / np.sqrt(2 * np.pi * (1 - t) * (sigma_1_1 - sigma_2_1) ** 2)
# g = [np.exp(-(1 - t)) * np.sqrt( (1 - t) ) * np.exp((-(W[-1] - W[i]) ** 2) / (1 - t))  for i, t in enumerate(time_steps)]
# plt.plot(time_steps, g, label = "test")

# l = [np.exp(-3* (1 - t)) * np.exp(-((0.02 * (W[-1] - W[i]) + 0.02)**2)/(0.02 * (1 - t))) * np.sqrt(1 - t) / (0.1)  for i, t in enumerate(time_steps)]
# l2 = [np.exp((-100 - 1) * (1 - t)) * np.sqrt(1 - t) for i, t in enumerate(time_steps)]
# plt.plot(time_steps, l, label = "test")
# plt.plot(time_steps, l2, label = "test2")
plt.title("integral")
plt.xlabel("x")
plt.ylabel("y")
plt.legend(loc = 'best')
plt.show()
