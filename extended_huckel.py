import os
import yaml
import numpy as np
import scipy.linalg as la
import tools
from overlap_integral import overlap_integral


def rotation_matrix(axis, angle):
    a = angle * np.pi / 180
    cos = np.cos(a)
    sin = np.sin(a)
    u = axis / np.linalg.norm(axis)

    # Rotation matrix
    r = np.zeros((3, 3))
    r[0, 0] = cos + (u[0] ** 2) * (1 - cos)
    r[0, 1] = u[0] * u[1] * (1 - cos) - u[2] * sin
    r[0, 2] = u[0] * u[2] * (1 - cos) + u[1] * sin
    r[1, 0] = u[1] * u[0] * (1 - cos) + u[2] * sin
    r[1, 1] = cos + (u[1] ** 2) * (1 - cos)
    r[1, 2] = u[1] * u[2] * (1 - cos) - u[0] * sin
    r[2, 0] = u[2] * u[0] * (1 - cos) - u[1] * sin
    r[2, 1] = u[2] * u[1] * (1 - cos) + u[0] * sin
    r[2, 2] = cos + (u[2] ** 2) * (1 - cos)

    return r


class ExtendedHuckel:
    def __init__(self, coordinates, symbols, charge=0, pure_orbitals=False):
        stream = open(os.path.dirname(os.path.abspath(__file__)) + '/basis_set.yaml')
        self.coordinates = coordinates
        self.symbols = symbols
        self.charge = charge
        self.pure_orbitals = pure_orbitals
        self.n_electrons = 0

        self.basis = yaml.load(stream)
        matrices_dimensions = self.matrices_dimensions()
        self.S = np.zeros((matrices_dimensions, matrices_dimensions))
        self.H = np.zeros((matrices_dimensions, matrices_dimensions))
        self.eigenvalues = None
        self.eigenvectors = None
        self.total_energy = None
        self.coefficients = []
        self.molecular_basis = None
        self.orbital_vector = []
        self.orbitals_atom = []
        self.atomic_numbers = []
        self.S1H = np.zeros((matrices_dimensions, matrices_dimensions))

    def get_coordinates(self):
        return self.coordinates

    def get_symbols(self):
        return self.symbols

    def get_basis(self):
        return self.basis

    def get_overlap_matrix(self):
        if not np.any(self.S):
            self.overlap_matrix()
        return self.S

    def get_hamiltonian_matrix(self):
        if not np.any(self.H):
            self.hamiltonian_matrix()
        return self.H

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

    def get_eigenvalues(self):
        if self.eigenvalues is None:
            self.calculate_energy()
        return self.eigenvalues

    def get_eigenvectors(self):
        if self.eigenvectors is None:
            self.calculate_energy()
        return self.eigenvectors

    def get_total_energy(self):
        if self.total_energy is None:
            self.total_energy = [ei * 2 for ei in self.get_eigenvalues()[:int((self.get_number_of_electrons()) / 2)]]
            if self.get_number_of_electrons() % 2 == 1:
                self.total_energy += self.get_eigenvalues()[int((self.get_number_of_electrons()) / 2)]
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
                    atomic_orbitals += 5
                    # if self.pure_orbitals:
                    #     atomic_orbitals += 5
                    # else:
                    #     atomic_orbitals += 6
                elif shell['shell_type'] == 'f':
                    atomic_orbitals += 7
                    # if self.pure_orbitals:
                    #     atomic_orbitals += 7
                    # else:
                    #     atomic_orbitals += 10
                else:
                    atomic_orbitals += 1
            matrices_dimensions += atomic_orbitals
        return matrices_dimensions

    def basis_transformation(self):
        orbital_df_positions = [[]]
        new_basis = ['dxx', 'dyy', 'dzz', 'dxy', 'dxz', 'dyz']
        new_eigenvectors = np.ndarray.tolist(self.eigenvectors.copy())

        nd = 0
        yz_positions = []
        for ido, orbital in enumerate(self.orbital_vector):
            current_atom = orbital.split('_')[0]
            if 'd' in orbital:

                if nd == 4:
                    yz_positions.append(ido+1)
                elif nd > 4:
                    nd = 0
                    orbital_df_positions.append([current_atom, [ido]])
                else:
                    if current_atom in orbital_df_positions[-1]:
                        orbital_df_positions[-1][1].append(ido)
                    else:
                        orbital_df_positions.append([current_atom, [ido]])
                self.orbital_vector[ido] = current_atom + '_' + new_basis[nd]
                nd += 1
        orbital_df_positions.pop(0)
        for yz in yz_positions:
            current_atom = self.orbital_vector[yz-1].split('_')[0]
            self.orbital_vector.insert(yz, current_atom + '_dyz')

        new_eigenvalues = np.ndarray.tolist(self.eigenvalues.copy())
        last_eigenvalue = new_eigenvalues[-1]
        for current_atom, suborbital in orbital_df_positions:
            #cal arreglar que passa quan hi ha més de dos metalls suborbital+1?
            for ide, mo in enumerate(self.eigenvectors):
                # new_eigenvectors[ide][suborbital[0]] = (mo[suborbital[0]]*np.sqrt(3/4) - mo[suborbital[1]]/2 +
                #                                         mo[suborbital[2]]/np.sqrt(5))
                # new_eigenvectors[ide][suborbital[1]] = (-mo[suborbital[0]]*np.sqrt(3/4) - mo[suborbital[1]]/2 +
                #                                         mo[suborbital[2]]/np.sqrt(5))
                # new_eigenvectors[ide][suborbital[2]] = (2*(mo[suborbital[1]])/2 +
                #                                         mo[suborbital[2]]/np.sqrt(5))
                new_eigenvectors[ide][suborbital[0]] = (mo[suborbital[0]] * np.sqrt(3 / 4) - mo[suborbital[1]] / 2)
                new_eigenvectors[ide][suborbital[1]] = (-mo[suborbital[0]] * np.sqrt(3 / 4) - mo[suborbital[1]] / 2)
                new_eigenvectors[ide].insert(suborbital[2], (2 * (mo[suborbital[1]]) / 2))

            new_eigenvectors.append([0]*len(new_eigenvectors[-1]))
            new_eigenvectors[-1][suborbital[0]] = 1
            new_eigenvectors[-1][suborbital[1]] = 1
            new_eigenvectors[-1][suborbital[2]] = 1
            new_eigenvalues.append(1.1*last_eigenvalue)

        self.eigenvalues = np.asarray(new_eigenvalues)
        self.eigenvectors = np.asarray(new_eigenvectors)

    def calculate_energy(self):
        self.eigenvalues, self.eigenvectors = la.eigh(self.get_hamiltonian_matrix(), b=self.get_overlap_matrix(),
                                                      type=1)
        idx = np.argsort(self.eigenvalues)
        self.eigenvalues = self.eigenvalues[idx]
        self.eigenvectors = self.eigenvectors[:, idx].T
        if not self.pure_orbitals:
            self.basis_transformation()

    def gaussianxgaussian(self, shell1, shell2, atom1_coordinates, atom2_coordinates, orbital_type1, orbital_type2):

        def double_factorial(n):
            if n == 0 or n == 1 or n == -1:
                return 1
            return n * double_factorial(n - 2)

        def norm_gaussian(orbital_type1, a1, orbital_type2, a2):

            n1, n2 = 0, 0
            if orbital_type1 == 's':
                n1 = 1
            elif 'p' in orbital_type1:
                n1 = 2
            elif 'd' in orbital_type1:
                n1 = 3
            elif 'f' in orbital_type1:
                n1 = 4
            if orbital_type2 == 's':
                n2 = 1
            elif 'p' in orbital_type2:
                n2 = 2
            elif 'd' in orbital_type2:
                n2 = 3
            elif 'f' in orbital_type2:
                n2 = 4

            norm1 = (a1 ** ((2 * n1 + 1) / 4)) * (2 ** (n1 + 1)) / (
                    (double_factorial(2 * n1 - 1)) * (2 * np.pi) ** (-1 / 4))
            norm2 = (a2 ** ((2 * n2 + 1) / 4)) * (2 ** (n2 + 1)) / (
                    (double_factorial(2 * n2 - 1)) * (2 * np.pi) ** (-1 / 4))

            return norm1 * norm2

        def real_spherical_harmonic(orbital_type):
            if 's' in orbital_type:
                return np.sqrt(1 / (4 * np.pi))
            elif 'p' in orbital_type:
                return np.sqrt(3 / (4 * np.pi))
            elif 'd' in orbital_type:
                return np.sqrt(15 / (4 * np.pi))
            elif 'f' in orbital_type:
                return np.sqrt(105 / (4 * np.pi))

        alphas1 = shell1['p_exponents']
        if 'p' in orbital_type1[1]:
            coefs1 = shell1['p_con_coefficients']
        else:
            coefs1 = shell1['con_coefficients']

        alphas2 = shell2['p_exponents']
        if 'p' in orbital_type2[1]:
            coefs2 = shell2['p_con_coefficients']
        else:
            coefs2 = shell2['con_coefficients']

        g0 = 0.
        if orbital_type1[1] == 'dxx - dyy':
            norm1 = np.sqrt(3/4)
        elif orbital_type1[1] == 'dzz + dzz - dxx - dyy':
            norm1 = 1/2
        elif orbital_type1[1] == 'dxx + dyy + dzz':
            norm1 = 1 / np.sqrt(5)
        else:
            norm1 = 1

        if orbital_type2[1] == 'dxx - dyy':
            norm2 = np.sqrt(3/4)
        elif orbital_type2[1] == 'dzz + dzz - dxx - dyy':
            norm2 = 1/2
        elif orbital_type2[1] == 'dxx + dyy + dzz':
            norm2 = 1 / np.sqrt(5)
        else:
            norm2 = 1

        operation1 = '+'
        for ido1, orbital1 in enumerate(orbital_type1[1].split(' ')):
            operation2 = '+'
            if ido1 % 2 != 0:
                operation1 = orbital1
                pass
            else:
                for ido2, orbital2 in enumerate(orbital_type2[1].split(' ')):
                    g1g2 = 0
                    if ido2 % 2 != 0:
                        operation2 = orbital2
                        pass
                    else:
                        for idf1, coef1 in enumerate(coefs1):
                            a1 = alphas1[idf1]
                            for idf2, coef2 in enumerate(coefs2):
                                a2 = alphas2[idf2]
                                coef = coef1 * coef2
                                g1g2 += norm1*norm2*coef * norm_gaussian(orbital_type1[1], a1, orbital_type2[1], a2) * \
                                        real_spherical_harmonic(orbital_type1[1]) * \
                                        real_spherical_harmonic(orbital_type2[1]) * \
                                        overlap_integral(orbital1, orbital2, atom1_coordinates,
                                                         atom2_coordinates, a1, a2)
                    if operation1 == operation2:
                        g0 += g1g2
                    else:
                        g0 -= g1g2

        return g0

    def return_shell_position(self, shell_list, type):
        for idsh, shell_type in enumerate(shell_list):
            if type in shell_type['shell_type']:
                return idsh

    def overlap_matrix(self):

        atom1 = self.get_ao_labels()[0].split('_')[0]
        atom1_position = 0
        for ido1, orbital1 in enumerate(self.get_ao_labels()):
            atom2 = self.get_ao_labels()[0].split('_')[0]
            atom2_position = 0

            if ido1 >= sum(self.orbitals_atom[:atom1_position + 1]):
                atom1 = orbital1.split('_')[0]
                atom1_position += 1
            for ido2, orbital2 in enumerate(self.get_ao_labels()):
                if ido2 >= sum(self.orbitals_atom[:atom2_position + 1]):
                    atom2 = orbital2.split('_')[0]
                    atom2_position += 1

                orbital_type1 = orbital1.split('_')[-1]
                orbital_type2 = orbital2.split('_')[-1]
                if 's' in orbital_type1 or 'p' in orbital_type1:
                    shell1 = self.basis[atom1][self.return_shell_position(self.basis[atom1], 's')]
                elif 'd' in orbital_type1:
                    shell1 = self.basis[atom1][self.return_shell_position(self.basis[atom1], 'd')]
                elif 'f' in orbital_type1:
                    shell1 = self.basis[atom1][self.return_shell_position(self.basis[atom1], 'f')]

                if 's' in orbital_type2 or 'p' in orbital_type2:
                    shell2 = self.basis[atom2][self.return_shell_position(self.basis[atom2], 's')]
                elif 'd' in orbital_type2:
                    shell2 = self.basis[atom2][self.return_shell_position(self.basis[atom2], 'd')]
                elif 'f' in orbital_type2:
                    shell2 = self.basis[atom2][self.return_shell_position(self.basis[atom2], 'f')]

                row1 = tools.get_element_row(atom1)
                row2 = tools.get_element_row(atom2)
                if 'd' in orbital_type1:
                    row1 -= 1
                elif 'f' in orbital_type1:
                    row1 -= 2

                if 'd' in orbital_type2:
                    row2 -= 1
                elif 'f' in orbital_type2:
                    row2 -= 2

                self.S[ido1][ido2] = self.gaussianxgaussian(shell1, shell2, self.coordinates[atom1_position],
                                                            self.coordinates[atom2_position], [row1, orbital_type1],
                                                            [row2, orbital_type2])

        norm_vector = [1 / np.sqrt(x) for x in np.diag(self.S)]
        for i in range(len(self.S)):
            for j in range(len(self.S[i])):
                self.S[i][j] *= norm_vector[i] * norm_vector[j]

    def hamiltonian_matrix(self):
        eV_to_Ha = 3.67493095E-2
        k = 0.75
        if not np.any(self.S):
            self.get_overlap_matrix()
        for ido1, orbital1 in enumerate(self.get_ao_labels()):
            for ido2, orbital2 in enumerate(self.get_ao_labels()):
                if ido1 == ido2:
                    h = self.get_VSIP(orbital1)
                    self.H[ido1][ido2] = h * eV_to_Ha
                else:
                    h1 = self.get_VSIP(orbital1)
                    h2 = self.get_VSIP(orbital2)
                    A = (h1 - h2) / (h1 + h2)
                    K = 1 + k + A ** 2 - k * A ** 4
                    self.H[ido1][ido2] = K * (h1 + h2) * (1 / 2) * eV_to_Ha * self.S[ido1][ido2]

    def get_VSIP(self, orbital_type):
        atom = orbital_type.split('_')[0]
        if orbital_type.split('_')[1] == 's':
            return self.basis[atom][self.return_shell_position(self.basis[atom], 's')]['VSIP'][0]
        elif 'p' in orbital_type.split('_')[1]:
            return self.basis[atom][self.return_shell_position(self.basis[atom], 's')]['VSIP'][1]
        elif 'd' in orbital_type.split('_')[1]:
            return self.basis[atom][self.return_shell_position(self.basis[atom], 'd')]['VSIP'][0]
        elif 'f' in orbital_type.split('_')[1]:
            return self.basis[atom][self.return_shell_position(self.basis[atom], 'p')]['VSIP'][0]

    def get_ao_labels(self):
        # cartesian_basis = {'s': ['s'],
        #                    'sp': ['s', 'px', 'py', 'pz'],
        #                    'd': ['dxx - dyy', 'dzz + dzz - dxx - dyy', 'dxx + dyy + dzz', 'dxy', 'dxz', 'dyz'],
        #                    'f': ['fxxx', 'fyyy', 'fzzz', 'fxyy', 'fxxy', 'fxxz', 'fxzz', 'fyzz', 'fyyz', 'fxyz']}
        pure_basis = {'s': ['s'],
                      'sp': ['s', 'px', 'py', 'pz'],
                      'd': ['dxx - dyy', 'dzz + dzz - dxx - dyy', 'dxy', 'dxz', 'dyz']}

        basis = pure_basis
        # if self.pure_orbitals:
        #     basis = pure_basis
        # else:
        #     basis = cartesian_basis

        if not self.orbital_vector:
            for symbol in self.symbols:
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
                self.molecular_basis['atoms'].append({'shells': self.basis[symbol]})
        return self.molecular_basis
