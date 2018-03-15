#!/usr/bin/env python
# # -*- coding: utf-8 -*-

"""
The purpose of this script is to make simulations of brownian motion and poisson processes.
Python version : 3.*
Authors :  Heang Kitiyavirayuth, Lucas Broux
"""

# Imports.
import math
import numpy as np
from tqdm import tqdm


class ProcessesGenerator:
    """
    The purpose of the class is to provide simulations of Brownian motion and Poisson processes.
    """

    def __init__(self, set_seed = True):
        """
        Constructor for the class.

        :param set_seed: Wheter to set a seed for reproductibility.
        """
        if set_seed:
            np.random.seed(42)


    def generate_brownian_motion(self, n_discr, T, m):
        """
        Simulates brownian motion.

        :param n_discr: The size of the discretization.
        :param T: Total time (arbitrary unit).
        :param m: The dimension of the brownian movement.
        :return: The array [W(t1n)m, ..., W(tn)m].
        """
        # Discretisation of time.
        time_steps = [i * T / (n_discr - 1) for i in range(n_discr)]
        # Define array to contain the values.
        W = []
        for i in range(m):
            W_aux = [0]
            for i in range(1, len(time_steps)):
                W_aux.append(W_aux[-1] + np.random.normal(0, np.sqrt(time_steps[i] - time_steps[i-1])))
            W.append(W_aux)
        return np.swapaxes(np.array(W), 0, 1)


    def generate_poisson_process(self, n_discr, T, n, kappa):
        """
        Simulates poisson process.

        :param n_discr: The size of the discretization.
        :param T: Total time (arbitrary unit).
        :param n: The dimension of the process.
        :param kappa: The intensity of the model. Must be np array with shape (n, )
        :return: The array [N(t1n)m, ..., N(tn)m].
        """
        # Assert the validity of model.
        assert(np.shape(kappa) == (n, ))
        # Discretisation of time.
        time_steps = [i * T / (n_discr - 1) for i in range(n_discr)]
        # Define array to contain the values.
        N = []
        for k in kappa:
            # Compute number of jumps.
            number_jumps = np.random.poisson(k)
            # Compute times of jumps.
            time_jumps = np.sort(np.random.uniform(0, T, number_jumps))
            # Define and fill array of values.
            N_aux = np.zeros(n_discr)
            for t in time_jumps:
                N_aux[int(t * (n_discr - 1) / T)] = 1
            current_value = 0
            for i in range(len(N_aux)):
                if N_aux[i] == 1:
                    current_value += 1
                N_aux[i] = current_value
            N.append(N_aux)
        if N != []:
            N = np.swapaxes(np.array(N), 0, 1)
        else:
            N = np.array([])
        return N
