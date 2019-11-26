import numpy as np

eV_to_Ha = 3.67493095E-2


def build_output(parsed_data, filename):

    output = open(filename+'.out', 'w')

    structure = np.array(parsed_data.get_coordinates())  # ['structure']
    symbols = parsed_data.get_symbols()
    alpha_mo_coeff = parsed_data.get_eigenvectors()  # ['coefficients']['alpha']
    # alpha_mo_coeff = parsed_data.get_mo_coefficients()
    alpha_mo_energies = np.real(parsed_data.get_mo_energies())  # ['mo_energies']['alpha']
    ao_list = parsed_data.get_ao_list()
    overlap = parsed_data.get_overlap_matrix()
    hamiltonian = parsed_data.get_hamiltonian_matrix()

    output.write('Extended Huckel calculation\n')
    output.write('Current cartesian coordinates:\n')
    for ids, symbol in enumerate(symbols):
        output.write('{:4} {:18.8f} {:18.8f} {:18.8f}'.format(symbol,
                                                              structure[ids][0],
                                                              structure[ids][1],
                                                              structure[ids][2]))
        output.write('\n')

    output.write('Molecular Orbital Coefficients: \n')
    for i in range(int(len(alpha_mo_energies)/5) + (len(alpha_mo_energies)%5 > 0)):

        n_orbitals = list(range(5*i, 5*i + 5))
        if i == int(len(alpha_mo_energies)/5):
            if len(alpha_mo_energies)%5 > 0:
                n_orbitals = n_orbitals[:len(alpha_mo_energies[n_orbitals[0]:])]

        width = 20
        for orbital in n_orbitals:
            output.write('{:{width}}'.format(orbital + 1, width=width))
            width = 18
        output.write('\n')
        width = 14
        output.write('Eigenvalues')
        for orbital in n_orbitals:
            output.write('{:{width}.8f}'.format(alpha_mo_energies[orbital]/eV_to_Ha, width=width))
            width = 18
        output.write('\n')
        output.write('\n')

        for j in range(len(alpha_mo_coeff[i])):
            width = 15
            output.write('{:4} {:5}'.format(j+1, ao_list[j].row+ao_list[j].type))
            if ao_list[j].type == 'dx2-y2':
                width = 13

            for orbital in n_orbitals:
                output.write('{:{width}.8f}'.format(
                    alpha_mo_coeff[orbital][j], width=width))
                width = 18
            output.write('\n')

    output.write('\n')
    output.write('Total Energy = {:18.8f}\n'.format(parsed_data.get_total_energy()/eV_to_Ha))
    output.write('\n')

    # La llargada dels mo no es la mateixa que el solapament i l'hamiltonià
    output.write('Overlap matrix: \n')
    for i in range(int(len(alpha_mo_energies)/5) + (len(alpha_mo_energies)%5 > 0)):

        n_orbitals = list(range(5*i, 5*i + 5))
        if i == int(len(alpha_mo_energies)/5):
            if len(alpha_mo_energies)%5 > 0:
                n_orbitals = n_orbitals[:len(alpha_mo_energies[n_orbitals[0]:])]

        width = 20
        for orbital in n_orbitals:
            output.write('{:{width}}'.format(orbital + 1, width=width))
            width = 18
        output.write('\n')
        output.write('\n')

        for j in range(len(alpha_mo_coeff[i])):
            output.write('{:3}'.format(j+1))
            width = 21
            for orbital in n_orbitals:
                output.write('{:{width}.8f}'.format(
                    overlap[orbital][j], width=width))
                width = 18
            output.write('\n')

    output.write('\n')

    output.write('Hamiltonian matrix: \n')
    for i in range(int(len(alpha_mo_energies)/5) + (len(alpha_mo_energies)%5 > 0)):

        n_orbitals = list(range(5*i, 5*i + 5))
        if i == int(len(alpha_mo_energies)/5):
            if len(alpha_mo_energies)%5 > 0:
                n_orbitals = n_orbitals[:len(alpha_mo_energies[n_orbitals[0]:])]

        width = 20
        for orbital in n_orbitals:
            output.write('{:{width}}'.format(orbital + 1, width=width))
            width = 18
        output.write('\n')
        output.write('\n')

        for j in range(len(alpha_mo_coeff[i])):
            output.write('{:3}'.format(j+1))
            width = 21
            for orbital in n_orbitals:
                output.write('{:{width}.8f}'.format(
                    hamiltonian[orbital][j], width=width))
                width = 18
            output.write('\n')

    output.write('\n')


def get_array_txt(label, type, array, row_size=5):

    formats = {'R': '15.8e',
               'I': '11'}

    n_elements = len(array)
    rows = int(np.ceil(n_elements/row_size))

    txt_fchk = '{:40}   {}   N=       {:5}\n'.format(label, type, n_elements)

    for i in range(rows):
        if (i+1)*row_size > n_elements:
            txt_fchk += (' {:{fmt}}'* (n_elements - i*row_size) + '\n').format(*array[i * row_size:n_elements],
                                                                               fmt=formats[type])
        else:
            txt_fchk += (' {:{fmt}}'* row_size  + '\n').format(*array[i * row_size: (i+1)*row_size],
                                                               fmt=formats[type])

    return txt_fchk


def build_fchk(parsed_data):

    structure = np.array(parsed_data.get_coordinates())#['structure']
    basis = parsed_data.get_molecular_basis()#['basis']
    alpha_mo_coeff = parsed_data.get_eigenvectors()#['coefficients']['alpha']
    alpha_mo_energies = parsed_data.get_mo_energies()#['mo_energies']['alpha']

    #overlap = parsed_data['overlap']
    #coor_shell = parsed_data['coor_shell']
    #core_hamiltonian = parsed_data['core_hamiltonian']
    #scf_density = parsed_data['scf_density']

    number_of_basis_functions = len(alpha_mo_coeff)
    number_of_electrons = parsed_data.get_number_of_electrons()
    # number_of_electrons = np.sum(structure.get_atomic_numbers()) - structure.charge
    alpha_electrons = number_of_electrons//2
    beta_electrons = number_of_electrons//2
    if parsed_data.get_multiplicity() > 1:
        alpha_electrons += 1

    #print(alpha_electrons)
    #print(number_of_electrons)

    alpha_mo_coeff = np.array(alpha_mo_coeff).flatten().tolist()

    # if 'beta' in parsed_data.get_mo_coefficients(): #['coefficients']:
    #     beta_mo_coeff = parsed_data['coefficients']['beta']
    #     beta_mo_coeff = np.array(beta_mo_coeff).flatten().tolist()
    #
    #     beta_mo_energies = parsed_data['alpha_mo_energies']['beta']

    shell_type_list = {'s':  {'type':  0, 'angular_momentum': 0},
                       'p':  {'type':  1, 'angular_momentum': 1},
                       'd':  {'type':  2, 'angular_momentum': 2},
                       'f':  {'type':  3, 'angular_momentum': 3},
                       'sp': {'type': -1, 'angular_momentum': 1},  # hybrid
                       'dc': {'type': -2, 'angular_momentum': 2},  # pure
                       'fc': {'type': -3, 'angular_momentum': 3}}  # pure

    shell_type = []
    p_exponents = []
    c_coefficients = []
    p_c_coefficients = []
    n_primitives = []
    atom_map = []

    largest_degree_of_contraction = 0
    highest_angular_momentum = 0
    number_of_contracted_shells = 0

    for i, atoms in enumerate(basis['atoms']):
        for shell in atoms['shells']:
            number_of_contracted_shells += 1
            st = shell['shell_type']
            shell_type.append(shell_type_list[st]['type'])
            n_primitives.append(len(shell['p_exponents']))
            atom_map.append(i+1)
            if highest_angular_momentum < shell_type_list[st]['angular_momentum']:
                highest_angular_momentum = shell_type_list[st]['angular_momentum']

            if len(shell['con_coefficients']) > largest_degree_of_contraction:
                    largest_degree_of_contraction = len(shell['con_coefficients'])

            for p in shell['p_exponents']:
                p_exponents.append(p)
            for c in shell['con_coefficients']:
                c_coefficients.append(c)
            for pc in shell['p_con_coefficients']:
                p_c_coefficients.append(pc)

    angstrom_to_bohr = 1/0.529177249
    coordinates_list = angstrom_to_bohr*structure.flatten()

    txt_fchk = '{}\n'.format('Extended Huckel calculation over filename')
    txt_fchk += 'SP        RHuckel                                                     {}\n'.format('STO-3G')
    txt_fchk += 'Number of atoms                            I               {}\n'.format(len(structure))
    txt_fchk += 'Charge                                     I               {}\n'.format(parsed_data.get_charge())
    txt_fchk += 'Multiplicity                               I               {}\n'.format(parsed_data.get_multiplicity())
    txt_fchk += 'Number of electrons                        I               {}\n'.format(number_of_electrons)
    txt_fchk += 'Number of alpha electrons                  I               {}\n'.format(alpha_electrons)
    txt_fchk += 'Number of beta electrons                   I               {}\n'.format(beta_electrons)
    txt_fchk += 'Number of basis functions                  I               {}\n'.format(number_of_basis_functions)

    txt_fchk += get_array_txt('Atomic numbers', 'I', parsed_data.get_atomic_numbers(), row_size=6)
    txt_fchk += get_array_txt('Nuclear charges', 'R', parsed_data.get_atomic_numbers())
    txt_fchk += get_array_txt('Current cartesian coordinates', 'R', coordinates_list)

    txt_fchk += 'Number of contracted shells                I               {}\n'.format(number_of_contracted_shells)
    txt_fchk += 'Number of primitive shells                 I               {}\n'.format(np.sum(n_primitives))
    txt_fchk += 'Highest angular momentum                   I               {}\n'.format(highest_angular_momentum)
    txt_fchk += 'Largest degree of contraction              I               {}\n'.format(largest_degree_of_contraction)

    txt_fchk += get_array_txt('Shell types', 'I', shell_type, row_size=6)
    txt_fchk += get_array_txt('Number of primitives per shell', 'I', n_primitives, row_size=6)
    txt_fchk += get_array_txt('Shell to atom map', 'I', atom_map, row_size=6)
    txt_fchk += get_array_txt('Primitive exponents', 'R', p_exponents)
    txt_fchk += get_array_txt('Contraction coefficients', 'R', c_coefficients)
    txt_fchk += get_array_txt('P(S=P) Contraction coefficients', 'R', p_c_coefficients)
    # txt_fchk += get_array_txt('Coordinates of each shell', 'R', coor_shell) #
    # txt_fchk += get_array_txt('Overlap Matrix', 'R', overlap)
    #txt_fchk += get_array_txt('Core Hamiltonian Matrix', 'R', core_hamiltonian)
    txt_fchk += 'Total Energy                               R             {}\n'.format(parsed_data.get_total_energy())
    txt_fchk += get_array_txt('Alpha Orbital Energies', 'R', alpha_mo_energies)
    # txt_fchk += get_array_txt('Beta Orbital Energies', 'R', beta_mo_energies)
    # txt_fchk += get_array_txt('Total SCF Density', 'R', scf_density)
    txt_fchk += get_array_txt('Alpha MO coefficients', 'R', alpha_mo_coeff)
    # txt_fchk += get_array_txt('Beta MO coefficients', 'R', beta_mo_coeff)

    return txt_fchk