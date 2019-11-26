import numpy as np
import math


def distance(p1, p2):
    Angs2Bohr = 1 / 5.2917726E-1
    p1 = [p * Angs2Bohr for p in p1]
    p2 = [p * Angs2Bohr for p in p2]
    return np.sqrt((p1[0] - p2[0]) ** 2 + (p1[1] - p2[1]) ** 2 + (p1[2] - p2[2]) ** 2)


def number_of_sum(s, t):
    if (s+t)%2 == 0:
        return int(0.5*(s+t))
    else:
        return int(0.5*(s+t-1))


def combination(n, k):
    if k > n:
        return 0
    return math.factorial(n)/(math.factorial(k)*math.factorial(n - k))


def binomial_products(t, m, n, a, b):
    fn = 0
    for i in range(t+1):
        if m-i < 0 or n-t+i < 0:
            fn += 0
        else:
            fn += combination(m, i)*combination(n, t - i)*(a**(m-i)*b**(n-t+i))
    return fn


def double_factorial(n):
    if n == 0 or n == 1 or n == -1:
        return 1
    return n * double_factorial(n - 2)


def overlap_integral(orbital_type1, orbital_type2, coordinates_atom1, coordinates_atom2, a1, a2):

    orbital_type_to_vector = {'s': [0, 0, 0],
                              'px': [1, 0, 0],
                              'py': [0, 1, 0],
                              'pz': [0, 0, 1],
                              'dxx': [2, 0, 0],
                              'dxy': [1, 1, 0],
                              'dxz': [1, 0, 1],
                              'dyy': [0, 2, 0],
                              'dyz': [0, 1, 1],
                              'dzz': [0, 0, 2]}

    R12 = distance(coordinates_atom1, coordinates_atom2)
    orbital1 = orbital_type_to_vector[orbital_type1]
    orbital2 = orbital_type_to_vector[orbital_type2]
    exponential = ((np.pi/(a1 + a2))**(3/2)*np.exp(-(R12**2)*(a1*a2/(a1 + a2))))

    sum_of_binomial_products = 1
    Angs2Bohr = 1 / 5.2917726E-1
    for i in range(0,3):
        max_number_of_sum = number_of_sum(orbital1[i], orbital2[i])
        fxyz = 0
        for j in range(max_number_of_sum+1):
            a = Angs2Bohr*(coordinates_atom2[i] - coordinates_atom1[i])*a2/(a1 + a2)
            b = Angs2Bohr*(coordinates_atom1[i] - coordinates_atom2[i])*a1/(a1 + a2)
            fxyz += binomial_products(2 * j, orbital1[i], orbital2[i], a, b) * \
                    double_factorial(2 * j - 1) / ((2 ** j) * (a1 + a2) ** j)
        sum_of_binomial_products *= fxyz

    overlap = exponential*sum_of_binomial_products

    return overlap


