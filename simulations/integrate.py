#!/usr/bin/env python
# # -*- coding: utf-8 -*-

"""
The purpose of this script is to implement integration of functions.
Python version : 3.*
Authors :  Heang Kitiyavirayuth, Lucas Broux
"""

# Imports.
import math
import numpy as np
from tqdm import tqdm
import itertools


class Integrate:
    """
    The purpose of the class is to provide integration methods.
    """

    def __init__(self):
        """
        Constructor for the class.
        """
        pass


    def integrate(self, fun, ker, low, upp, T, return_all_values = False):
        """
        Integrate the function fun with the kernel ker in the interval [low, upp]

        :param fun: A np array corresponding to the values between 0 and T of the function to integrate.
        :param ker: A np array corresponding to the values between 0 and T of the integration kernel.
        :param low: Real value between 0 and T corresponding to the lower bound.
        :param upp: Real value between 0 and T corresponding to the upper bound.
        :param T: Total time.
        :param return_all_values: Whether to return the whole array [0, ... I(low, t), ... I] or not (faster computation in that case).
        :return: The value of the integral.
        """
        # First, check the validity of the arguments.
        bool_1 = (np.shape(fun)[0] == np.shape(ker)[0])
        bool_2 = (low >= 0  and low <= upp and upp <= T)
        assert(bool_1 and bool_2)

        # Get the discrete values of the lower and upper bounds.
        length = np.shape(fun)[0]
        low_disc = int(low * (length - 1) / T)
        upp_disc = int(upp * (length - 1) / T)

        # Do the computation.
        s = 0
        all_values = [s]
        for i in range(low_disc, upp_disc):
            s += np.dot(fun[i], ker[i + 1] - ker[i])
            all_values.append(s)

        # Return corresponging value.
        if return_all_values:
            return np.array(all_values)
        else:
            return s


    def riemann_integrate(self, fun, low, upp, T, return_all_values = False):
        """
        Compute riemann integral of the function fun in the interval [low, upp]

        :param fun: A np array corresponding to the values between 0 and T of the function to integrate.
        :param low: Real value between 0 and T corresponding to the lower bound.
        :param upp: Real value between 0 and T corresponding to the upper bound.
        :param T: Total time.
        :param return_all_values: Whether to return the whole array [0, ... I(low, t), ... I] or not (faster computation in that case).
        :return: The value of the integral.
        """
        # Compute corresponding kernel by hand and use self.integrate().
        length = np.shape(fun)[0]
        ker = np.array([T * i / (length - 1) for i in range(length)])
        return self.integrate(fun, ker, low, upp, T, return_all_values)


    def doleans_exponential(self, riemann_fun, ito_fun, ito_ker, jump_fun, jump_ker, T):
        """
        Compute the doleans exponential of X(t) =  R(t) + I(t) + J(t), with intitial condition E(0) = 1.

        :param riemann_fun: A np array corresponding to the values between 0 and T of the riemann part of the process X.
        :param ito_fun: A np array corresponding to the values between 0 and T of the ito part of the process X.
        :param ito_ker: A np array corresponding to the values between 0 and T of the brownian motion with respect to which to integrate the ito part of the process X.
        :param jump_fun: A np array corresponding to the values between 0 and T of the jump part of the process X.
        :param jump_ker: A np array corresponding to the values between 0 and T of the poisson process with respect to which to integrate the jump part of the process X. It is a Poisson process, i.e. jumps are of size 1.
        :param T: Total time.
        :return: The array [E(0), E(t1), E(t2), ..] of values of the exponential.
        """
        # First, we check the validity of the arguements.
        bool_1 = (np.shape(riemann_fun)[0] == np.shape(ito_fun)[0]) and (np.shape(riemann_fun)[0] == np.shape(jump_fun)[0])
        bool_2 = (np.shape(ito_fun)[0] == np.shape(ito_ker)[0]) and ((np.shape(jump_fun)[0] == np.shape(jump_ker)[0]) or np.shape(jump_ker) == (0, ))
        assert(bool_1)
        assert(bool_2)

        # Compute the riemann part of the exponential.
        length = np.shape(riemann_fun)[0]
        time_steps = np.array([T * i / (length - 1) for i in range(length)])
        rie = self.riemann_integrate(fun = riemann_fun - 0.5 * np.linalg.norm(ito_fun, axis = 1) ** 2,  low = 0, upp = T, T = T, return_all_values = True)

        # Compute ito part of exponential.
        ito = self.integrate(fun = ito_fun, ker = ito_ker, low = 0, upp = T, T = T, return_all_values = True)

        # Compute poisson part of the exponential (product computation) if needed.
        if np.shape(jump_ker) == (0, ):
            poi = 1
        else:
            poi = [1]
            current_value = 1
            for i in range(length - 1):
                current_value = current_value * (1 + np.dot(jump_fun[i], jump_ker[i + 1] - jump_ker[i]))
                poi.append(current_value)
            poi = np.array(poi)

        # Return result.
        return np.exp(rie + ito) * poi
