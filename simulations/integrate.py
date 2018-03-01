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
        bool_2 = (np.shape(ito_fun)[0] == np.shape(ito_ker)[0]) and (np.shape(jump_fun)[0] == np.shape(jump_ker)[0])
        assert(bool_1 and bool_2)

        # Compute the riemann part of the exponential.
        length = np.shape(riemann_fun)[0]
        time_steps = np.array([T * i / (length - 1) for i in range(length)])
        rie = self.riemann_integrate(fun = riemann_fun - 0.5 * np.linalg.norm(ito_fun, axis = 1) ** 2,  low = 0, upp = T, T = T, return_all_values = True)

        # Compute ito part of exponential.
        ito = self.integrate(fun = ito_fun, ker = ito_ker, low = 0, upp = T, T = T, return_all_values = True)

        # Compute poisson part of the exponential (product computation).
        poi = [1]
        current_value = 1
        for i in range(length - 1):
            current_value = current_value * (1 + np.dot(jump_fun[i], jump_ker[i + 1] - jump_ker[i]))
            poi.append(current_value)
        poi = np.array(poi)

        # Return result.
        return np.exp(rie + ito) * poi


    def riemann_integrate_nd(self, fun, low, upp, T, return_all_values = False):
        """
        Compupte riemann integral for a multidimensional function.

        :param fun: A np array corresponding to the values between [0; T]^n of the function to integrate.
        :param low: Real value between 0 and T corresponding to the lower bound.
        :param upp: Real value between 0 and T corresponding to the upper bound.
        :param T: Total time.
        :param return_all_values: Whether to return the whole array [0, ... I(low, t), ... I] or not (faster computation in that case).
        :return: The value of the integral.
        """
        # Check validity of input data : we take only data with same size of discretisation.
        dim = len(np.shape(fun))
        bool_1 = len(set(np.shape(fun))) <= 1
        assert(bool_1)

        # Get the discrete values of the lower and upper bounds.
        length = np.shape(fun)[0]
        low_disc = int(low * (length - 1) / T)
        upp_disc = int(upp * (length - 1) / T)

        # Do the computation.
        s = 0
        all_values = [s]
        for i in tqdm(range(low_disc, upp_disc), leave = False):
            # Build the set of elements to add for this iteration.
            n_tuples = list(itertools.product(range(low_disc, i + 1), repeat=dim))
            valid_n_tuples = [x for x in n_tuples if (i in x)]
            s += sum([fun[t] for t in valid_n_tuples]) * ((T / (length - 1)) ** dim)
            all_values.append(s)

        # Return corresponging value.
        if return_all_values:
            return np.array(all_values)
        else:
            return s

if __name__ == "__main__":
    """
    from process_simulation import ProcessesGenerator
    integrator = Integrate()
    process_generator = ProcessesGenerator(set_seed = True)
    s = 500  # Size of discretisation.
    n = 2   # Dimension of poisson process.
    m = 3   # Dimension of brownian motion.
    time_steps = np.array([i / (s - 1) for i in range(s)])
    riemann_fun = 0.3 * np.ones(s)
    jump_ker = process_generator.generate_poisson_process(n_discr = s, T = 1, n = n, kappa = np.array([3, 2]))
    jump_fun =  np.array([0.4, -0.2]) * np.ones((s, 2))
    ito_ker = process_generator.generate_brownian_motion(n_discr = s, T = 1, m = m)
    ito_fun = np.array([0.3, -0.025, -0.7]) * np.ones((s, 3))
    dol_exp = integrator.doleans_exponential(riemann_fun, ito_fun, ito_ker, jump_fun, jump_ker, T = 1)

    import matplotlib.pyplot as plt
    plt.figure(1)
    plt.plot(time_steps, dol_exp, label = "Price evolution")
    plt.title("Price evolution")
    plt.xlabel("Time")
    plt.ylabel("Price")
    plt.legend(loc = 'best')
    # plt.savefig('wealths.png')
    plt.show() """

    integrator = Integrate()
    s = 100  # Size of discretisation.
    T = 1  # Total time.
    time_steps = np.array([T * i / (s - 1) for i in range(s)])
    riemann_fun_1 = np.array([[ (x + y ) ** 2 for x in time_steps] for y in time_steps])
    int_1 = integrator.riemann_integrate_nd(riemann_fun_1, low = 0, upp = T, T = T, return_all_values = False)
    print(int_1)
    riemann_fun_2 = np.array([[[ (x + y + z) ** 2 for x in time_steps] for y in time_steps] for z in time_steps])
    int_2 = integrator.riemann_integrate_nd(riemann_fun_2, low = 0, upp = T, T = T, return_all_values = False)
    print(int_2)
