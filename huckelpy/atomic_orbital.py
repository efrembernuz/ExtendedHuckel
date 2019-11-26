import numpy as np
import copy
from huckelpy import tools


class AtomicOrbital():

    def __init__(self, atom, atomic_basis, position, orbital_type, eh_energy):

        self._atom = atom
        self._shell = atomic_basis
        self._position = position
        self._orbital_type = orbital_type
        self._energy = eh_energy
        self.linear_combination = None
        self.linear_combination_coefficients = None

    def get_linear_combination(self):
        if self.linear_combination is None:
            pure_basis = {'dz2': {'orbitals': ['dzz', 'dxx', 'dyy'], 'lcc': [1, -1/2, -1/2]},
                          'dx2-y2': {'orbitals': ['dxx', 'dyy'], 'lcc': [np.sqrt(3/4), -np.sqrt(3/4)]}}
            if self._orbital_type in pure_basis:
                self.linear_combination = pure_basis[self._orbital_type]['orbitals']
                self.linear_combination_coefficients = pure_basis[self._orbital_type]['lcc']
            else:
                self.linear_combination = [self._orbital_type]
                self.linear_combination_coefficients = [1]
        return self.linear_combination, self.linear_combination_coefficients

    def double_factorial(self, n):
        if n == 0 or n == 1 or n == -1:
            return 1
        return n * self.double_factorial(n - 2)

    def get_gaussian_norm(self):

        def double_factorial(n):
            if n == 0 or n == 1 or n == -1:
                return 1
            return n * double_factorial(n - 2)

        n1 = 0
        if self._orbital_type == 's':
            n1 = 1
        elif 'p' in self._orbital_type:
            n1 = 2
        elif 'd' in self._orbital_type:
            n1 = 3
        elif 'f' in self._orbital_type:
            n1 = 4

        norm_vector = []
        for alpha in self.ao_orbital_shell['p_exponents']:
            norm_vector.append((alpha ** ((2 * n1 + 1) / 4)) * (2 ** (n1 + 1)) / (
                    (double_factorial(2 * n1 - 1)) * (2 * np.pi) ** (-1 / 4)))
        return norm_vector

    def get_real_spherical_harmonic(self):
        if 's' in self._orbital_type:
            return np.sqrt(1 / (4 * np.pi))
        elif 'p' in self._orbital_type:
            return np.sqrt(3 / (4 * np.pi))
        elif 'd' in self._orbital_type:
            return np.sqrt(15 / (4 * np.pi))
        elif 'f' in self._orbital_type:
            return np.sqrt(105 / (4 * np.pi))

    def get_atomic_orbital_coeff(self):
        if 'p' in self._orbital_type:
            coefficients = copy.deepcopy(self.ao_orbital_shell['p_con_coefficients'])
        else:
            coefficients = copy.deepcopy(self.ao_orbital_shell['con_coefficients'])
        return coefficients

    def get_atomic_orbital_exponents(self):
        return copy.deepcopy(self.ao_orbital_shell['p_exponents'])

    def get_ao_row(self):

        if 'd' in self._orbital_type:
            row = str(tools.get_element_row(self.atom) - 1)
        elif 'f' in self._orbital_type:
            row = str(tools.get_element_row(self.atom) - 2)
        else:
            row = str(tools.get_element_row(self.atom))
        return row

    @property
    def ao_coefficients(self):
        return self.get_atomic_orbital_coeff()

    @property
    def ao_exponents(self):
        return self.get_atomic_orbital_exponents()

    @property
    def gaussian_norm(self):
        return self.get_gaussian_norm()

    def real_spherical_harmonic(self):
        return self.get_real_spherical_harmonic()

    @property
    def ao_orbital_shell(self):
        return self._shell

    @property
    def ao_linear_combination(self):
        return self.get_linear_combination()

    @property
    def position(self):
        return self._position

    @property
    def atom(self):
        return self._atom

    @property
    def type(self):
        return self._orbital_type

    @property
    def energy(self):
        return self._energy

    @property
    def row(self):
        return self.get_ao_row()
