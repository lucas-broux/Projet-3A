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

#########################
# Initialize the model. #
#########################
model = Model()

############################
# Define model parameters. #
############################
# Time input.
model.T = 1                          # Total time (arbitrary unit).
model.A = 0.9                       # Time at which to compute wealth.
model.x = 1                          # Initial wealth of the agent.
model.n_discr = 500                  # Size of discretization.
model.time_steps = np.array([i * model.T / (model.n_discr - 1) for i in range(model.n_discr)])

# Market parameters input.
model.r = 0.25                        # Interest rate of the bond.
model.m = 1                          # Dimension of the Brownian motion.
model.n = 1                          # Dimension of the Poisson process.
model.d = model.m + model.n          # Total dimension.
model.kappa = np.array([5])          # Intensity of the Poisson process.
model.b = np.array([0.25, 0.01])     # Drift of assets.
model.sigma = np.array([[-0.05, 0.01],
              [0.07, 0.01],
              ])                     # Volatility of assets.
print(np.linalg.inv(model.sigma))

# Insider knowledge input.
model.i_1 = 1
model.i_2 = 2
model.nb_terms_sum = 2               # Number of terms computed in the sum defining Z.


###################################
# Print model and model validity. #
###################################
print(model)
print("Model validity: " + str(model._check_model_validity()))

##########################################
# Simulate market model and plot prices. #
##########################################
model._simulate_prices()
model._plot_prices_evolution()
model._compute_L()
print("Value of the variable known to the insider: L = " + str(model.L))

#################################################
# Compute the optimal strategy for non insider. #
#################################################
model._compute_theta_Q()
print(model.q)
model._compute_Y_non_insider()
model._plot_Y_non_insider()

#################################################
# Compute the optimal strategy for the insider. #
#################################################
model._compute_Z()
model._compute_Y_insider()
model._plot_Z()
model._plot_Y_insider()
model._plot_both_agents()
