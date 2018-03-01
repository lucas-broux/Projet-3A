#!/usr/bin/env python
# # -*- coding: utf-8 -*-

"""
The purpose of this script is to make simulations of insider trading.
Contents are based on the article :
    "Hillairet, Caroline. (2005). Comparison of insiders' optimal strategies
    depending on the type of side-information."

Python version : 3.*
Authors :  Heang Kitiyavirayuth, Lucas Broux
"""

# Imports.
import math
import numpy as np
import matplotlib.pyplot as plt
import scipy.stats as st
from tqdm import tqdm

from process_simulation import ProcessesGenerator
from integrate import Integrate


class JumpModel:
    """
    This class proposes a simple model for simulation of insider trading.

    We suppose :
      - The insider knows the variable L = ln(S1(T)) - ln(S2(T)).
      - The utility function to optimize is logarithmic (U1 = U2 = ln).
    Then, according to the article, we can compute the wealth and consumption at time A < T by the following steps.
    """

    def __init__(self, parameters = 'default'):
        """
        Constructor for the class. Initialize variables of the class by setting them to false.
        """
        # Set seed for reproductibility.
        np.random.seed(42)

        # Parameters of the model.
        self.T = False                          # Total time (arbitrary unit).
        self.A = False                          # Time at which to compute wealth.
        self.x = False                          # Initial wealth.
        self.n_discr = False                    # Size of discretization.
        self.time_steps = False                 # Discretisation.

        self.m = False                          # Dimension of the Brownian motion.
        self.n = False                          # Dimension of the Poisson process.
        self.d = False                          # Total dimension.
        self.r = False                          # Interest rate of the bond.
        self.kappa = False                      # Intensity of the poisson process.
        self.b = False                          # Drift of assets.
        self.sigma = False                      # Volatility of assets.

        self.i_1 = False                        # The insider knows L = ln(P_{i_1}(T)) - ln(P_{i_2}(T)).
        self.i_2 = False                        # The insider knows L = ln(P_{i_1}(T)) - ln(P_{i_2}(T)).

        # Other useful variables.

        self.theta = False                      # Theta.
        self.q = False                          # q.
        self.W = False                          # Brownian motion.
        self.N = False                          # Poisson process.
        self.prices = False                     # Prices of the process.
        self.L = False                          # Value of the variable L = ln(P_{i_1}(T)) - ln(P_{i_2}(T)) known by the insider.
        self.y_0 = False                        # Strategy of non_insider.

        # Actually initialize parameters of the model if parameters == 'default'.
        if (parameters == 'default'):
            # Time input.
            self.T = 1                          # Total time (arbitrary unit).
            self.A = 0.95                       # Time at which to compute wealth.
            self.x = 1                          # Initial wealth of the agent.
            self.n_discr = 500                  # Size of discretization.
            self.time_steps = np.array([i / (self.n_discr - 1) for i in range(self.n_discr)])

            # Market parameters input.
            self.r = 0.02                       # Interest rate of the bond.
            self.m = 2                          # Dimension of the Brownian motion.
            self.n = 2                          # Dimension of the Poisson process.
            self.d = self.m + self.n            # Total dimension.
            self.kappa = np.array([3, 2])       # Intensity of the Poisson process.
            self.b = np.array([0.15, 0.1, 0.084, 0.1])    # Drift of assets.
            self.sigma = np.array([[-0.4, -0.1, -0.15, 0.17],
                          [-0.09, -0.4, -0.03, 0.035],
                          [0.048, -0.12, 0.1, -0.12],
                          [0.075, 0.26, 0.31, -0.28],
                          ])                   # Volatility of assets.

            # Insider knowledge input.
            self.i_1 = 1
            self.i_2 = 2


    def __str__(self):
        """
        Redefines underlying string of class.
        """
        s = "Jump model with following parameters :"
        s += "\n\t-Total time: " + str(self.T)
        s += "\n\t-Terminal time: " + str(self.A)
        s += "\n\t-Initial wealth: " + str(self.x)
        s += "\n\t-Size of discretization: " + str(self.n_discr)
        s += "\n\t-Dimension of the Brownian motion: " + str(self.m)
        s += "\n\t-Dimension of the Poisson process: " + str(self.n)
        s += "\n\t-Interest rate of bond: " + str(self.r)
        s += "\n\t-Intensity of Poisson process: " + str(self.kappa)
        s += "\n\t-Drift of assets: " + str(self.b)
        s += "\n\t-Volatility of assets: " + str(self.sigma)
        return s


    def _check_model_validity(self):
        """
        Checks that the dimensions provided in parameter agree with the dimensions
        of drifts, volatilities, ...
        """
        bool_0 = self.d == self.m + self.n
        bool_1 = np.shape(self.kappa) == (self.n, )
        bool_2 = np.shape(self.b) == (self.d, )
        bool_3 = np.shape(self.sigma) == (self.d, self.d)
        bool_4 = np.shape(self.time_steps)[0] == self.n_discr
        bool_5 = (self.i_1 > 0) and (self.i_2 > 0) and (self.i_1 <= self.d) and (self.i_2 <= self.d)
        return (bool_0 and bool_1 and bool_2 and bool_3 and bool_4 and bool_5)


    def _compute_theta_Q(self):
        """
        Compute variables Theta and q from the parameters of the model.
        """
        aux_var = np.dot(np.linalg.inv(self.sigma), self.b - self.r)
        self.theta = aux_var[:self.m]
        self.q = - (aux_var[self.m:].T / self.kappa).T


    def _simulate_prices(self, set_seed = True):
        """
        Simulate the prices of the assets by solving the equation with doleans exponential.

        :param set_seed: Wheter to set a seed for the generation of brownian and poisson processes.
        :return: A np array of arrays P = [[P0], [P1], ...] with P0 = bond and Pi (i > 0) = risky assets.
        """
        # Initialize the generator of the processes and the integrator.
        integrator = Integrate()
        process_generator = ProcessesGenerator(set_seed = set_seed)

        # Generate brownian motion and poisson process.
        self.W = process_generator.generate_brownian_motion(n_discr = self.n_discr, T = self.T, m = self.m)
        self.N = process_generator.generate_poisson_process(n_discr = self.n_discr, T = self.T, n = self.n, kappa = self.kappa)

        # Compute bond.
        self.prices = [np.array([np.exp(self.r * t) for t in self.time_steps])]

        # Compute other arrays.
        for i in range(self.d):
            riemann_fun = self.b[i] * np.ones(self.n_discr)
            jump_fun = self.sigma[i][self.m:] * np.ones((self.n_discr, self.n))
            ito_fun = self.sigma[i][:self.m] * np.ones((self.n_discr, self.m))
            self.prices.append(integrator.doleans_exponential(riemann_fun, ito_fun, self.W, jump_fun, self.N, T = self.T))

        # End computation.
        self.prices = np.array(self.prices)


    def _plot_prices_evolution(self):
        """
        Plots the evolution of the prices, assuming they have been computed.
        """
        counter = 0
        label = "Bond"
        for prices in self.prices:
            plt.figure(counter)
            counter += 1
            plt.plot(self.time_steps, prices, label = "Price evolution: " + label)
            plt.title("Price evolution: " + label)
            plt.xlabel("Time")
            plt.ylabel("Price: " + label)
            plt.legend(loc = 'best')
            label = "Asset " + str(counter)
        plt.show()


    def _compute_L(self):
        """
        Compute the value of the variable L that is known by the insider.
        """
        self.L = np.log(self.prices[self.i_1][-1]) - np.log(self.prices[self.i_2][-1])


    def _compute_Y_non_insider(self):
        """
        Compute the process Y0 which gives all wanted informations about the strategy of the non insider.

        This can be done by solving a doleans exponential.
        """
        # Initialize integrator.
        integrator = Integrate()

        # Compute different terms of the equation.
        ito_fun = - self.theta * np.ones((self.n_discr, self.m))
        jump_fun = (self.q - 1) * np.ones((self.n_discr, self.n))
        riemann_fun = - np.dot(jump_fun, self.kappa)

        # Compute Y0.
        self.y_0 = integrator.doleans_exponential(riemann_fun, ito_fun, self.W, jump_fun, self.N, T = self.T)


    def _plot_Y_non_insider(self):
        """
        Plots the evolution of the strategy of the non insider.
        """
        plt.figure(1)
        plt.plot(self.time_steps, 1 / self.y_0, label = r"$\frac{1}{Y_0}$ (non insider)")
        plt.title(r"$\frac{1}{Y_0}$ (non insider)")
        plt.xlabel("Time")
        plt.ylabel(r"$\frac{1}{Y_0}$ (non insider)")
        plt.legend(loc = 'best')
        plt.show()


    def _compute_Z(self):
        """
        Compute the process Z as described numerically.

        It is an approximation since we can not compute infinity sums.
        """
        # Initialize integrator.
        integrator = Integrate()

        # Z depends on whether there is an underlying jump process or not.
        if self.n == 0:
            pass # TODO: Implement formulas in the case of purely diffusive market model.
        else:
            # We make the approximation that the first term in the sum over kj is dominant.
            # First : we compute m_t as per the cited paper.
            aux_int = self.T * (self.b[self.i_1] - self.b[self.i_2] - 0.5 * (np.linalg.norm(self.sigma[self.i_1][:self.m]) ** 2 - np.linalg.norm(self.sigma[self.i_2][:self.m]) ** 2))
            ito_fun = (self.sigma[self.i_1][:self.m] - self.sigma[self.i_2][:self.m]) * np.ones((self.n_discr, self.m))
            jump_fun = np.log((1 + self.sigma[self.i_1][self.m:]) / (1 + self.sigma[self.i_2][self.m:])) * np.ones((self.n_discr, self.n))
            ito_int = integrator.integrate(ito_fun, self.W, 0, self.T, self.T, return_all_values = True)
            jump_int = integrator.integrate(jump_fun, self.N, 0, self.T, self.T, return_all_values = True)
            m_t = aux_int + ito_int + jump_int

            # Second : We compute the value of Sigma_t as per the cited paper.
            sum_t = self.T * (np.linalg.norm(self.sigma[self.i_1][:self.m] - self.sigma[self.i_2][:self.m]) ** 2)
            riemann_fun = (np.linalg.norm(self.sigma[self.i_1][:self.m] - self.sigma[self.i_2][:self.m]) ** 2) * np.ones(self.n_discr)
            sum_t = sum_t - integrator.riemann_integrate(fun = riemann_fun, low = 0, upp = self.T, T = self.T, return_all_values = True)

            # Compute the numerator of the formula that computes Z.
            product = 1
            for j in range(self.n):
                pass



if __name__ == "__main__":
    jump_model = JumpModel()
    print(jump_model)
    print("Model validity: " + str(jump_model._check_model_validity()))
    jump_model._simulate_prices()
    # jump_model._plot_prices_evolution()
    jump_model._compute_L()
    print("Value of the variable known to the insider: L = " + str(jump_model.L))
    jump_model._compute_theta_Q()
    jump_model._compute_Y_non_insider()
    # jump_model._plot_Y_non_insider()
    jump_model._compute_Z()
