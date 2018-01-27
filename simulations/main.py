#!/usr/bin/env python
# # -*- coding: utf-8 -*-

"""
The purpose of this script is to show the utilisation of the simple implemented model.

Python version : 3.*
Authors :  Heang Kitiyavirayuth, Lucas Broux
"""

# Imports.
from model import SimpleModel
from tqdm import tqdm
import matplotlib.pyplot as plt
import numpy as np


#########################################################
# 1. Initialize model and use it on default parameters. #
#########################################################

# Initialize model.
model = SimpleModel()
# Simulate market.
model.simulate()
# Compute optimal strategies for agents.
model.compute_optimal_strategy()
# Apply statistical test.
model.apply_statistical_test()
# Print results.
regions_failed_outsider = 100 * sum(model.critical_regions_outsider) / len(model.critical_regions_outsider)
regions_failed_insider = 100 * sum(model.critical_regions_insider) / len(model.critical_regions_insider)
print("The outsider fails the statistical test in " + str(regions_failed_outsider) + "% cases.")
print("The insider fails the statistical test in " + str(regions_failed_insider) + "% cases.")


##########################################################################
# 2. Study the evolution of the optimal wealth as A (final time) varies. #
##########################################################################

# Define final times A to study.
frac_values = [0.5, 0.6, 0.7, 0.8, 0.85, 0.9, 0.92, 0.94, 0.95, 0.96, 0.97, 0.98, 0.985, 0.99, 0.995, 0.997, 0.999]
wealth_insider, wealth_outsider = [], []

# Compute wealth at different times.
for frac in tqdm(frac_values, desc = 'Computing wealth at different terminal times', leave = False):
    #  Modify parameters.
    model.A = frac
    # Compute model.
    model.compute_optimal_strategy()
    # Append results.
    wealth_insider.append(model.XA_insider)
    wealth_outsider.append(model.XA_outsider)

# Plot results.
plt.figure(1)
plt.plot(frac_values, wealth_outsider, label = "Non insider wealth")
plt.plot(frac_values, wealth_insider, label = "Insider wealth")
plt.title("Representation of wealth at time A for different terminal times.")
plt.xlabel("Terminal time A (arbitrary unit)")
plt.ylabel("Wealth (arbitrary unit)")
plt.legend(loc = 'best')
# plt.savefig('wealths.png')
plt.show()

plt.figure(2)
plt.plot(frac_values, (np.array(wealth_insider) - np.array(wealth_outsider)) / np.array(wealth_outsider), label = "Relative difference of wealth")
plt.title("Relative difference of wealth at time A for different terminal times.")
plt.xlabel("Terminal time A (arbitrary unit)")
plt.ylabel("Relative difference of wealth")
plt.legend(loc = 'best')
# plt.savefig('relative_difference.png')
plt.show()
