#!/usr/bin/env python
# # -*- coding: utf-8 -*-

"""
The purpose of this script is to make simulations of insider trading.
Contents are based on the articles :
    "Hillairet, Caroline. (2005). Comparison of insiders' optimal strategies
    depending on the type of side-information."
    "Grorud, Axel & Pontier, Monique. (2011). Insider Trading in a Continuous
    Time Market Model. International Journal of Theoretical and Applied Finance"

Python version : 3.*
Authors :  Heang Kitiyavirayuth, Lucas Broux
"""

#######################
# Import model class. #
#######################
from model import Model
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
from process_simulation import ProcessesGenerator


#########################
# Initialize the model. #
#########################
process_generator = ProcessesGenerator()

time_steps = np.array([i  / (499) for i in range(int(500 * 0.9))])
b = np.array([0.1, -0.05])     # Drift of assets.
sigma = np.array([[0.75, 0],
              [0, 1]
              ])

##########################
# Loop over simulations. #
##########################
array_Zs = []
sim_nb = 1000
for s in tqdm(range(sim_nb), desc = 'Computing averages', leave = False):
    Z = []
    W = process_generator.generate_brownian_motion(n_discr = 500, T = 1, m = 2)
    for i, t in enumerate(time_steps):
        Z.append(np.exp((np.dot(sigma[1] - sigma[0], W[-1] - W[i])) **2 ))
    array_Zs.append(np.array(Z))

array_Zs = np.array(array_Zs)
average_Z = np.average(array_Zs, axis=0)
std_Z = np.std(array_Zs, axis = 0)


plt.figure(1)
index_A = int(0.5 * 499)
plt.plot(time_steps[index_A: ], average_Z[index_A: ], label = r"$Z$ (average over " + str(sim_nb) + " simulations)")
plt.plot(time_steps[index_A: ], (average_Z + 2 * std_Z / np.sqrt(sim_nb))[index_A: ], 'r--')
plt.plot(time_steps[index_A: ], (average_Z - 2 * std_Z / np.sqrt(sim_nb))[index_A: ], 'r--')
plt.plot(time_steps[index_A: ], [ np.exp(np.linalg.norm(sigma[1] - sigma[0]) **2 *  (1 - t)) for t in time_steps[index_A: ]], label = r"$Z_{\simeq} $")
plt.title(r"$Z$ (average over " + str(sim_nb) + " simulations)")
plt.xlabel("Time")
plt.ylabel(r"$Z$")
plt.legend(loc = 'best')
plt.show()
