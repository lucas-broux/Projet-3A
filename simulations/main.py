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
from simple_model import SimpleModelParameters, SimpleModel
from tqdm import tqdm
import matplotlib.pyplot as plt
import numpy as np


parameters = SimpleModelParameters()
model = SimpleModel(parameters)

frac_values = [0.5, 0.6, 0.7, 0.8, 0.85, 0.9, 0.92, 0.94, 0.95, 0.96, 0.97, 0.98, 0.985, 0.99, 0.995, 0.997, 0.999]
wealth_insider = []
wealth_outsider = []
for frac in tqdm(frac_values, desc = 'Computing wealth at different terminal times', leave = False):
    #  Modify parameters.
    parameters.A = frac * parameters.T
    model = SimpleModel(parameters)
    # Compute model.
    [XA_outsider, XA_insider] = model.compute_optimal_wealth(print_result = False)
    # Append results.
    wealth_insider.append(XA_insider)
    wealth_outsider.append(XA_outsider)
    tqdm.write(str(frac) + " : " + str(XA_insider) + " " + str(XA_outsider) + " " + str(XA_insider - XA_outsider))

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
