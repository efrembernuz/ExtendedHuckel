__version__ = '0.1.1'

import os
import copy
import yaml
import numpy as np
import scipy.linalg as la
from huckelpy import tools, atomic_orbital
from huckelpy.overlap_integral import overlap_integral


class ExtendedHuckel:

    def __init__(self, coordinates, symbols, charge=0, cartesian_orbitals=False):

        self.coordinates = coordinates
        self.symbols = symbols
        self.charge = charge
        self.pure_orbitals = not cartesian_orbitals
        self.n_electrons = 0
        with open(os.path.dirname(os.path.abspath(__file__)) + '/basis_set.yaml') as file:
            self.basis = yaml.load(file)
        self.overlap = np.zeros((self.matrices_dimensions(), self.matrices_dimensions()))
        self.hamiltonian = None
        self.eigenvalues = None
        self.eigenvectors = None
        self.total_energy = None
        self.coefficients = []
        self.molecular_basis = None
        self.orbitals_list = []
        self.orbital_vector = []
        self.orbitals_atom = []
        self.atomic_numbers = []

    def set_atom_energy(self, symbol, sp_energy, d_energy=None, f_energy=None):

        self.basis[symbol][0]['VSIP'] = sp_energy
        if d_energy is not None:
            self.basis[symbol][1]['VSIP'] = d_energy
        if f_energy is not None:
            self.basis[symbol][2]['VSIP'] = f_energy
        self.overlap = None
        self.hamiltonian = None
        self.eigenvalues = None
        self.eigenvectors = None

    def get_coordinates(self):
        return self.coordinates

    def get_symbols(self):
        return self.symbols

    def get_basis(self):
        return self.basis

    def get_overlap_matrix(self):
        if not np.any(self.overlap):
            self.overlap_matrix()
        return self.overlap

    def get_hamiltonian_matrix(self):
        if self.hamiltonian is None:
            self.hamiltonian_matrix()
        return self.hamiltonian

    def get_mo_coefficients(self):
        if not self.coefficients:
            self.norm_coefficients()
        return self.coefficients

    def norm_coefficients(self):
        for mo_coefficients in self.get_eigenvectors():
            norm = 0
            for coef in mo_coefficients:
                norm += coef**2
            norm = 1 / np.sqrt(norm)
            self.coefficients.append(norm * mo_coefficients)

    def get_mo_energies(self):
        if self.eigenvalues is None:
            self.calculate_energy()
        return self.eigenvalues

    def get_eigenvectors(self):
        if self.eigenvectors is None:
            self.calculate_energy()
        return self.eigenvectors

    def get_total_energy(self):
        if self.total_energy is None:
            self.total_energy = [ei * 2 for ei in self.get_mo_energies()[:int((self.get_number_of_electrons()) / 2)]]
            if self.get_number_of_electrons() % 2 == 1:
                self.total_energy += self.get_mo_energies()[int((self.get_number_of_electrons()) / 2)]
            self.total_energy = sum(self.total_energy)
        return self.total_energy

    def matrices_dimensions(self):
        matrices_dimensions = 0
        for symbol in self.symbols:
            atomic_orbitals = 0
            for shell in self.basis[symbol]:
                if shell['shell_type'] == 'sp':
                    atomic_orbitals += 4
                elif shell['shell_type'] == 'd':
                    if self.pure_orbitals:
                        atomic_orbitals += 5
                    else:
                        atomic_orbitals += 6
                elif shell['shell_type'] == 'f':
                    if self.pure_orbitals:
                        atomic_orbitals += 7
                    else:
                        atomic_orbitals += 10
                else:
                    atomic_orbitals += 1
            matrices_dimensions += atomic_orbitals
        return matrices_dimensions

    # def basis_transformation(self):
    #     orbital_df_positions = [[]]
    #     new_basis = ['dxx', 'dyy', 'dzz', 'dxy', 'dxz', 'dyz']
    #     new_eigenvectors = np.ndarray.tolist(self.eigenvectors.copy())
    #
    #     nd = 0
    #     yz_positions = []
    #     for ido, orbital in enumerate(self.orbital_vector):
    #         current_atom = orbital.split('_')[0]
    #         if 'd' in orbital:
    #
    #             if nd == 4:
    #                 yz_positions.append(ido+1)
    #             elif nd > 4:
    #                 nd = 0
    #                 orbital_df_positions.append([current_atom, [ido]])
    #             else:
    #                 if current_atom in orbital_df_positions[-1]:
    #                     orbital_df_positions[-1][1].append(ido)
    #                 else:
    #                     orbital_df_positions.append([current_atom, [ido]])
    #             self.orbital_vector[ido] = current_atom + '_' + new_basis[nd]
    #             nd += 1
    #     orbital_df_positions.pop(0)
    #     for yz in yz_positions:
    #         current_atom = self.orbital_vector[yz-1].split('_')[0]
    #         self.orbital_vector.insert(yz, current_atom + '_dyz')
    #
    #     new_eigenvalues = np.ndarray.tolist(self.eigenvalues.copy())
    #     last_eigenvalue = new_eigenvalues[-1]
    #     for current_atom, suborbital in orbital_df_positions:
    #         #cal arreglar que passa quan hi ha més de dos metalls suborbital+1?
    #         for ide, mo in enumerate(self.eigenvectors):
    #             # new_eigenvectors[ide][suborbital[0]] = (mo[suborbital[0]]*np.sqrt(3/4) - mo[suborbital[1]]/2 +
    #             #                                         mo[suborbital[2]]/np.sqrt(5))
    #             # new_eigenvectors[ide][suborbital[1]] = (-mo[suborbital[0]]*np.sqrt(3/4) - mo[suborbital[1]]/2 +
    #             #                                         mo[suborbital[2]]/np.sqrt(5))
    #             # new_eigenvectors[ide][suborbital[2]] = (2*(mo[suborbital[1]])/2 +
    #             #                                         mo[suborbital[2]]/np.sqrt(5))
    #             new_eigenvectors[ide][suborbital[0]] = (mo[suborbital[0]] * np.sqrt(3 / 4) - mo[suborbital[1]] / 2)
    #             new_eigenvectors[ide][suborbital[1]] = (-mo[suborbital[0]] * np.sqrt(3 / 4) - mo[suborbital[1]] / 2)
    #             new_eigenvectors[ide].insert(suborbital[2], (2 * (mo[suborbital[1]]) / 2))
    #
    #         new_eigenvectors.append([0]*len(new_eigenvectors[-1]))
    #         new_eigenvectors[-1][suborbital[0]] = 1
    #         new_eigenvectors[-1][suborbital[1]] = 1
    #         new_eigenvectors[-1][suborbital[2]] = 1
    #         new_eigenvalues.append(1.1*last_eigenvalue)
    #
    #     self.eigenvalues = np.asarray(new_eigenvalues)
    #     self.eigenvectors = np.asarray(new_eigenvectors)

    def calculate_energy(self):
        self.eigenvalues, self.eigenvectors = la.eigh(self.get_hamiltonian_matrix(), b=self.get_overlap_matrix())
        idx = np.argsort(self.eigenvalues)
        self.eigenvalues = self.eigenvalues[idx]
        self.eigenvectors = self.eigenvectors[:, idx].T
        # if not self.pure_orbitals:
        #     self.basis_transformation()

    def orbitalxorbital(self, orbital_obj1, orbital_obj2):

        g0 = 0.
        for ido1, orbital1 in enumerate(orbital_obj1.ao_linear_combination[0]):
            for ido2, orbital2 in enumerate(orbital_obj2.ao_linear_combination[0]):
                g1g2 = 0
                for g1_parameters in range(3):
                    for g2_parameters in range(3):
                        g1g2 += orbital_obj1.ao_coefficients[g1_parameters]*orbital_obj2.ao_coefficients[g2_parameters] * \
                                orbital_obj1.gaussian_norm[g1_parameters]*orbital_obj2.gaussian_norm[g2_parameters] * \
                                orbital_obj1.real_spherical_harmonic() * orbital_obj2.real_spherical_harmonic() * \
                                overlap_integral(orbital1, orbital2, orbital_obj1.position, orbital_obj2.position,
                                                 orbital_obj1.ao_exponents[g1_parameters],
                                                 orbital_obj2.ao_exponents[g2_parameters])
                g0 += orbital_obj1.ao_linear_combination[1][ido1]*orbital_obj2.ao_linear_combination[1][ido2]*g1g2

        return g0

    def return_shell_position(self, shell_list, type):
        for idsh, shell_type in enumerate(shell_list):
            if type in shell_type['shell_type']:
                return idsh

    def overlap_matrix(self):

        for ido1, orbital_obj1 in enumerate(self.get_ao_list()):
            for ido2, orbital_obj2 in enumerate(self.get_ao_list()[ido1:]):
                self.overlap[ido1][ido1+ido2] = self.orbitalxorbital(orbital_obj1, orbital_obj2)

        self.overlap = self.overlap + self.overlap.T - np.diag(np.diag(self.overlap))

        norm_vector = [1 / np.sqrt(x) for x in np.diag(self.overlap)]
        for i in range(len(self.overlap)):
            for j in range(len(self.overlap[i])):
                self.overlap[i][j] *= norm_vector[i] * norm_vector[j]

    def hamiltonian_matrix(self):

        self.hamiltonian = np.zeros((self.matrices_dimensions(), self.matrices_dimensions()))
        ev_to_ha = 3.67493095E-2
        k = 0.75
        if not np.any(self.overlap):
            self.get_overlap_matrix()
        for ido1, orbital_obj1 in enumerate(self.get_ao_list()):
            for ido2, orbital_obj2 in enumerate(self.get_ao_list()):
                if ido1 == ido2:
                    h = orbital_obj1.energy
                    self.hamiltonian[ido1][ido2] = h * ev_to_ha
                else:
                    h1 = orbital_obj1.energy
                    h2 = orbital_obj2.energy
                    a = (h1 - h2) / (h1 + h2)
                    eh_constant = 1 + k + a ** 2 - k * a ** 4
                    self.hamiltonian[ido1][ido2] = eh_constant * (h1 + h2) * (1 / 2) * ev_to_ha * self.overlap[ido1][ido2]

    def get_ao_list(self):
        cartesian_basis = {'s': ['s'],
                           'sp': ['s', 'px', 'py', 'pz'],
                           'd': ['dxx', 'dyy', 'dzz', 'dxy', 'dxz', 'dyz'],
                           'f': ['fxxx', 'fyyy', 'fzzz', 'fxyy', 'fxxy', 'fxxz', 'fxzz', 'fyzz', 'fyyz', 'fxyz']}
        # cartesian_basis = {'s': ['s'],
        #                    'sp': ['s', 'px', 'py', 'pz'],
        #                    'd': ['dxx - dyy', 'dzz + dzz - dxx - dyy', 'dxy', 'dxz', 'dyz']}
        pure_basis = {'s': ['s'],
                      'sp': ['s', 'px', 'py', 'pz'],
                      # 'd': ['dzz + dzz - dxx - dyy', 'dxz', 'dyz', 'dxx - dyy', 'dxy']
                      'd': ['dz2', 'dxz', 'dyz', 'dx2-y2', 'dxy']
                      }

        if self.pure_orbitals:
            basis = pure_basis
        else:
            basis = cartesian_basis

        if not self.orbitals_list:
            for ids, symbol in enumerate(self.symbols):
                for shell in self.basis[symbol]:
                    for orbital_type in basis[shell['shell_type']]:
                        if 'p' in orbital_type:
                            eh_energy = shell['VSIP'][1]
                        else:
                            eh_energy = shell['VSIP'][0]
                        self.orbitals_list.append([atomic_orbital.AtomicOrbital(symbol, shell, self.coordinates[ids],
                                                                                orbital_type, eh_energy)])
            self.orbitals_list = [y for x in self.orbitals_list for y in x]
        return self.orbitals_list

    def get_ao_labels(self):
        cartesian_basis = {'s': ['s'],
                           'sp': ['s', 'px', 'py', 'pz'],
                           'd': ['dxx', 'dyy', 'dzz', 'dxy', 'dxz', 'dyz'],
                           'f': ['fxxx', 'fyyy', 'fzzz', 'fxyy', 'fxxy', 'fxxz', 'fxzz', 'fyzz', 'fyyz', 'fxyz']}
        # cartesian_basis = {'s': ['s'],
        #                    'sp': ['s', 'px', 'py', 'pz'],
        #                    'd': ['dxx - dyy', 'dzz + dzz - dxx - dyy', 'dxy', 'dxz', 'dyz']}
        pure_basis = {'s': ['s'],
                      'sp': ['s', 'px', 'py', 'pz'],
                      # 'd': ['dzz + dzz - dxx - dyy', 'dxz', 'dyz', 'dxx - dyy', 'dxy']
                      'd': ['dz2', 'dxz', 'dyz', 'dx2-y2', 'dxy']
                      }

        if self.pure_orbitals:
            basis = pure_basis
        else:
            basis = cartesian_basis

        if not self.orbital_vector:
            for ids, symbol in enumerate(self.symbols):
                n_atomic_orbitals = 0
                for shell in self.basis[symbol]:
                    self.orbital_vector.append([symbol + '_' + x for x in basis[shell['shell_type']]])
                    n_atomic_orbitals += len(basis[shell['shell_type']])
                self.orbitals_atom.append(n_atomic_orbitals)
            self.orbital_vector = [y for x in self.orbital_vector for y in x]
        return self.orbital_vector

    def get_number_of_electrons(self):
        if self.n_electrons == 0:
            for symbol in self.symbols:
                symbol = symbol.split('_')[0]
                self.n_electrons += tools.element_valence_electron(symbol)
            self.n_electrons += self.charge
        return self.n_electrons

    def get_multiplicity(self):
        if self.get_number_of_electrons() % 2 == 0:
            self.multiplicity = 1
        else:
            self.multiplicity = 2
        return self.multiplicity

    def get_atomic_numbers(self):
        if not self.atomic_numbers:
            for symbol in self.symbols:
                symbol = symbol.split('_')[0]
                self.atomic_numbers.append(tools.element_to_atomic_number(symbol))
        return self.atomic_numbers

    def get_charge(self):
        return self.charge

    def get_molecular_basis(self):
        if self.molecular_basis is None:
            self.molecular_basis = {'name': 'STO-3G',
                                    'primitive_type': 'gaussian',
                                    'atoms': []}
            for symbol in self.symbols:
                pure_basis = copy.deepcopy(self.basis[symbol])
                if self.pure_orbitals and len(self.basis[symbol]) > 1:
                    pure_basis[1]['shell_type'] = 'd_'
                self.molecular_basis['atoms'].append({'shells': pure_basis})
        return self.molecular_basis
