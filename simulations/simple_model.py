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
from tqdm import tqdm

# Fixing random state for reproducibility
random.seed(42)

class SimpleModelParameters:
    """
    Simple class to modelize the parameters of the model.
    """

    def __init__(self):
        """
        Constructor for the class. We put default parameters.
        """
        # Time input.
        self.T = 1                  # Total time (arbitrary unit).
        self.A = 0.9 * self.T       # Time at which to compute wealth (arbitrary unit).
        self.n = 5000               # Size of discretization.

        # Market parameters input.
        self.r = 0.4                # Interest rate of the bond.
        self.b1 = 0.6               # Drift of first asset.
        self.sigma_1 = 0.2          # Volatility of first asset
        self.b2 = 0.75              # Drift of second asset.
        self.sigma_2 = 0.1          # Volatility of second asset.

        # Initial wealth of the agent.
        self.x = 1                  # Initial wealth.


class SimpleModel:
    """
    This class proposes a simple model for simulation of insider trading.

    We suppose :
      - The market consists in one bond(r) and 2 assets (bi, sigma_i).
      - The insider knows the variable L = ln(S1(T)) - ln(S2(T)).
      - The utility function to optimize is logarithmic (U1 = U2 = ln).
    Then, according to the article, we can compute the wealth at time A < T by the following steps.
    """

    def __init__(self, parameters):
        """
        Constructor for the class.

        :param parameters: The parameters used for this modelisation. Instance of the class SimpleModelParameters.
        """
        # Initialize parameters.
        self.parameters = parameters
        # Set seed for reproducibility.
        random.seed(42)


    # Simulation of rademacher variable.
    def __rademacher(self):
        """
        Simulates a rademacher variable.

        :return: -1 or 1 with 0.5 probability.
        """
        return 2* random.randint(0, 1) - 1


    # Simulation of the brownian motion.
    def __W(self, t, m = 10000):
        """
        Simulation of the brownian motion W_tn, according to the formulas pp.41-43 of the book.

        :param t: The array [t1, ... tn], tn being the highest value.
        :param m: The number of steps in the simulation of the brownian motion. Optional parameter.
        :return: The array [W(t1n)m, ..., W(tn)m].
        """
        # Simulation of n rademacher variables.
        Y = [0] + [self.__rademacher() for i in range(math.ceil(t[-1]*m) + 2)]

        # Cumulative sum of the variables.
        M = np.cumsum(Y)

        # Value of the simulated brownian motion.
        return [(M[math.floor(m * v)] + (m * v - math.floor(m * v)) * Y[math.floor(m * v) + 1] ) / math.sqrt(m) for v in t]


    # Simulation of market.
    def __market_2_assets(self, show = False):
            """
            Simulation of a market with 2 risky assets S1 and S2, and a bond S0.

            We create a regular discretization of the time in n equal values between 0 and T.

            :param show: Whether to show the plotted view.
            :return: The market as [[S0, S1, S2], [W1, W2]], each Si being Si = [Si(0), ... Si(T)] and Wi being brownian motion.
            """
            # Discretisation of time.
            time_steps = [i * self.parameters.T / self.parameters.n for i in range(self.parameters.n + 1)]

            # Compute 2 brownian motions.
            W1 = self.__W(time_steps)
            W2 = self.__W(time_steps)

            # Compute S0 (bond) and S1/S2 (risky assets) in the Black-Scholes model.
            S0 = np.exp(self.parameters.r * np.array(time_steps))
            S1 = np.exp( (self.parameters.b1 - self.parameters.sigma_1 * self.parameters.sigma_1 / 2) * np.array(time_steps) + self.parameters.sigma_1 * np.array(W1) )
            S2 = np.exp( (self.parameters.b2 - self.parameters.sigma_2 * self.parameters.sigma_2 / 2) * np.array(time_steps) + self.parameters.sigma_2 * np.array(W2) )

            # Plot evolution of assets if asked.
            if show:
                plt.figure(1)
                plt.plot(time_steps, S0, label = "Bond")
                plt.plot(time_steps, S1, label = "Risky asset S1")
                plt.plot(time_steps, S2, label = "Risky asset S2")
                plt.title("Representation of asset prices")
                plt.xlabel("Time")
                plt.ylabel("Asset prices")
                plt.legend(loc = 'best')
                plt.show()

            # Return result.
            return [[S0, S1, S2], [W1, W2]]


    # Computation of optimal wealth in model.
    def compute_optimal_wealth(self, print_result = False):
        """
        Compute the optimal wealth of an insider and of a non-insider in a simplified case.

        We create a regular discretization of the time in n equal values between 0 and T.
        We suppose :
          - The market consists in one bond(r) and 2 assets (bi, sigma_i).
          - The insider knows the variable L = ln(S1(T)) - ln(S2(T)).
          - The utility function to optimize is logarithmic (U1 = U2 = ln).
        Then, according to the article, we can compute the wealth at time A < T by the following steps.

        :param print_result: Whether to print obtained result in console.
        :return: Optimal wealth of the agents as an array [XA_outsider, XA_insider].
        """

        ##########################
        # Step 0 : Gather input. #
        ##########################
        # Time input.
        T = self.parameters.T               # Total time (arbitrary unit).
        A = self.parameters.A               # Time at which to compute wealth (arbitrary unit).
        n = self.parameters.n               # Size of discretization.

        # Market parameters input.
        r = self.parameters.r               # Interest rate of the bond.
        b1 = self.parameters.b1             # Drift of first asset.
        sigma_1 = self.parameters.sigma_1   # Volatility of first asset
        b2 = self.parameters.b2             # Drift of second asset.
        sigma_2 = self.parameters.sigma_2   # Volatility of second asset.

        # Initial wealth of the agent.
        x = self.parameters.x               # Initial wealth.


        ###################################
        # Step 1 : Compute eta and gamma. #
        ###################################
        eta = np.array([(b1 - r) / sigma_1, (b2 - r) / sigma_2])
        gamma = np.array([sigma_1, - sigma_2])


        #################################################
        # Step 2 : Simulate Brownian Motion and Market. #
        #################################################
        # Get market values over time.
        [[S0, S1, S2], [W1, W2]] = self.__market_2_assets()


        ##########################################################
        # Step 3 : Compute 2-d vector l.                         #
        # We compute l on [0, ..., A] with the formula:          #
        # l(r) = 1 / (T - r) * int(gamma . dWs, [r, T]) * gamma. #
        ##########################################################
        l = [ ((gamma[0] * (W1[-1] - W1[r]) + gamma[1] * (W2[-1] - W2[r])) / (n - r)) * gamma for r in range(int(A * n / T)) ]


        #################################################
        # Step 4 : Compute B.                           #
        # We compute B on [0, ..., A] with the formula: #
        # B(t) = W(t) - int(l(u) . du, [0, t])          #
        #################################################
        B = [np.array([W1[r], W2[r]]) - sum(l[:(r)]) * (T / n) for r in tqdm(range(int(A * n / T)), desc = 'Computing Bt', leave = False)]


        ###################################
        # Step 5 : Compute M and M_tilda. #
        ###################################
        # Computation of M.
        eta_sqare = np.dot(eta, eta)
        M = [np.exp( - np.dot( eta, np.array([W1[r], W2[r]]) ) - .5 * (r * T / n) * eta_sqare ) for r in range(int(A * n / T)) ]

        # Computation of M_tilda.
        f = [v + eta for v in l]
        int_1 = np.array( [sum([np.dot(f[s], B[s + 1] - B[s]) for s in range(r)]) for r in tqdm(range(int(A * n / T)), desc = 'Computing M_tilda', leave = False)] )
        int_2 = np.array( [(sum([np.dot(f[s], f[s]) for s in range(r)]) * (T / n)) for r in tqdm(range(int(A * n / T)), desc = 'Computing M_tilda', leave = False)] )
        M_tilda = np.exp(-int_1 - 0.5 * int_2)


        ######################################
        # Step 6 : Compute optimal strategy. #
        ######################################
        # Optimal wealth at time A for the non-insider.
        y = x / (A + 1)
        XA_outsider = np.exp(r * A) * y / M[-1]

        # Optimal wealth at time A for the insider.
        XA_insider = np.exp(r * A) * y / M_tilda[-1]

        # Output result if asked.
        if print_result:
            print("The optimal wealth at time A is (knowing X_0 = " + str(x) + "): ")
            print("\t For the non_insider: " + str(XA_outsider))
            print("\t For the insider: " + str(XA_insider))

        # Return values.
        return [XA_outsider, XA_insider]
