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
from model_jump import JumpModel
import numpy as np

#########################
# Initialize the model. #
#########################
model = JumpModel()

############################
# Define model parameters. #
############################
# Time input.
model.T = 1                          # Total time (arbitrary unit).
model.A = 0.95                       # Time at which to compute wealth.
model.x = 1                          # Initial wealth of the agent.
model.n_discr = 500                  # Size of discretization.
model.time_steps = np.array([i * model.T / (model.n_discr - 1) for i in range(model.n_discr)])

# Market parameters input.
model.r = 0.1                        # Interest rate of the bond.
model.m = 2                          # Dimension of the Brownian motion.
model.n = 0                          # Dimension of the Poisson process.
model.d = model.m + model.n            # Total dimension.
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
model._compute_Y_non_insider()
model._plot_Y_non_insider()

#################################################
# Compute the optimal strategy for the insider. #
#################################################
model._compute_Z()
model._compute_Y_insider()
model._plot_Z()
model._plot_Z_with_approximation()
model._plot_Y_insider()
model._plot_both_agents()
