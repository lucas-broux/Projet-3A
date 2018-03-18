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

#########################
# Initialize the model. #
#########################
model = Model()

############################
# Define model parameters. #
############################
# Time input.
model.T = 1                          # Total time (arbitrary unit).
model.A = 0.9                        # Time at which to compute wealth.
model.x = 1                          # Initial wealth of the agent.
model.n_discr = 500                  # Size of discretization.
model.time_steps = np.array([i * model.T / (model.n_discr - 1) for i in range(model.n_discr)])

# Market parameters input.
model.r = 0.1                        # Interest rate of the bond.
model.m = 2                          # Dimension of the Brownian motion.
model.n = 0                          # Dimension of the Poisson process.
model.d = model.m + model.n          # Total dimension.
model.kappa = np.array([])           # Intensity of the Poisson process.
model.b = np.array([0.1, -0.05])     # Drift of assets.
model.sigma = np.array([[0.75, 0],
              [0, 1]
              ])                     # Volatility of assets.

# Insider knowledge input.
model.i_1 = 1
model.i_2 = 2

###################################
# Print model and model validity. #
###################################
print(model)
print("Model validity: " + str(model._check_model_validity()))


##########################
# Loop over simulations. #
##########################
array_Zs = []
sim_nb = 10000
for s in tqdm(range(sim_nb), desc = 'Computing averages', leave = False):
    ##########################
    # Simulate market model. #
    ##########################
    model._simulate_prices(set_seed = False)

    #################################################
    # Compute the optimal strategy for the insider. #
    #################################################
    model._compute_Z()
    array_Zs.append(model.Z)


array_Zs = np.array(array_Zs)
average_Z = np.average(array_Zs, axis=0)
std_Z = np.std(array_Zs, axis = 0)


plt.figure(1)
index_A = int(model.A * (model.n_discr - 1) / model.T)
plt.plot(model.time_steps[:index_A], average_Z[:index_A], label = r"$Z$ (average over " + str(sim_nb) + " simulations)")
plt.plot(model.time_steps[:index_A], (average_Z + 2 * std_Z / np.sqrt(sim_nb))[:index_A], 'r--')
plt.plot(model.time_steps[:index_A], (average_Z - 2 * std_Z / np.sqrt(sim_nb))[:index_A], 'r--')
plt.plot(model.time_steps[:index_A], [np.sqrt(model.T) / np.sqrt(model.T - t) for t in model.time_steps[:index_A]], label = r"$Z_{\simeq} $")
plt.title(r"$Z$ (average over " + str(sim_nb) + " simulations)")
plt.xlabel("Time")
plt.ylabel(r"$Z$")
plt.legend(loc = 'best')
plt.show()
