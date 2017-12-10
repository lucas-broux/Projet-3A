#!/usr/bin/env python
# # -*- coding: utf-8 -*-

"""
The purpose of this script is to make simulations of insider trading.
Contents are based on the article :
    "Grorud, Axel & Pontier, Monique. (2011). Insider Trading in a Continuous
    Time Market Model. International Journal of Theoretical and Applied Finance"

Python version : 3.*
Authors :  Heang Kitiyavirayuth, Lucas Broux
"""

# Imports.
import math
import random
import numpy as np
import matplotlib.pyplot as plt
import scipy.stats as st

# Fixing random state for reproducibility
random.seed(42)

# Global variables.
T = 1 # Total time (arbitrary unit).


# Simulation of rademacher variable.
def rademacher():
    """
    Simulates a rademacher variable.

    :return: -1 or 1 with 0.5 probability.
    """
    return 2* random.randint(0, 1) - 1

# Simulation of the brownian motion.
def W(t, m = 10000):
    """
    Simulation of the brownian motion W_tn, according to the formulas pp.41-43 of the book.

    :param t: The array [t1, ... tn], tn being the highest value.
    :param m: The number of steps in the simulation of the brownian motion. Optional parameter.
    :return: The array [W(t1n)m, ..., W(tn)m].
    """
    # Simulation of n rademacher variables.
    Y = [0] + [rademacher() for i in range(math.ceil(t[-1]*m) + 2)]
    # Cumulative sum of the variables.
    M = np.cumsum(Y)
    # Value of the simulated brownian motion.
    return [(M[math.floor(m * v)] + (m * v - math.floor(m * v)) * Y[math.floor(m * v) + 1] ) / math.sqrt(m) for v in t]

def dW(dt):
    """
    Simulation of the variation in brownian motion during the time dt, assumed small.

    :param dt: The variation in time.
    :return: The variation of the brownian motion during dt.
    """
    return np.random.normal(0, 1) * math.sqrt(dt)

# Main part of the script.
if __name__ == '__main__':
    # Discretisation of time.
    n = 10000
    time_steps = [i * T / n for i in range(n + 1)]

    # Parameters.
    r = 0.4
    sigma_1 = 0.2
    sigma_2 = 0.1

    # Compute S0 (bond) and S1/S2 (risky assets) in the Black-Scholes model.
    S0 = np.exp(r * np.array(time_steps))
    S1 = np.exp( (r - sigma_1 * sigma_1 / 2) * np.array(time_steps) + sigma_1 * np.array(W(time_steps)) )
    S2 = np.exp( (r - sigma_2 * sigma_2 / 2) * np.array(time_steps) + sigma_2 * np.array(W(time_steps)) )

    # Show evolution of assets.
    plt.figure(1)
    plt.plot(time_steps, S0, label = "Bond")
    plt.plot(time_steps, S1, label = "Risky asset S1")
    plt.plot(time_steps, S2, label = "Risky asset S2")
    plt.title("Representation of asset prices")
    plt.xlabel("Time")
    plt.ylabel("Asset prices")
    plt.legend(loc = 'best')
    plt.show()

    # To be continued.
