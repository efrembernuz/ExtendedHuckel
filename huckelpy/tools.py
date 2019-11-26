# Store the periodic table's information
# Z Atomic number, A Mass number, valence electrons
periodic_table_info = dict(H=[1, 1.0079, 1], He=[2, 4.0026, 2], Li=[3, 6.941, 1], Be=[4, 9.0122, 2], B=[5, 10.811, 3],
                           C=[6, 12.0107, 4], N=[7, 14.0067, 5], O=[8, 15.9994, 6], F=[9, 18.9984, 7],
                           Ne=[10, 20.1797, 8], Na=[11, 22.9897, 1], Mg=[12, 24.305, 2], Al=[13, 26.9815, 3],
                           Si=[14, 28.0855, 4], P=[15, 30.9738, 5], S=[16, 32.065, 6], Cl=[17, 35.453, 7],
                           Ar=[18, 39.948, 8], K=[19, 39.0983, 1], Ca=[20, 40.078, 2], Sc=[21, 44.9559, 3],
                           Ti=[22, 47.867, 4], V=[23, 50.9415, 5], Cr=[24, 51.9961, 6], Mn=[25, 54.938, 7],
                           Fe=[26, 55.845, 8], Co=[27, 58.9332, 9], Ni=[28, 58.6934, 10], Cu=[29, 63.546, 11],
                           Zn=[30, 65.390, 12], Ga=[31, 69.723, 13], Ge=[32, 72.640, 14], As=[33, 74.9216, 15],
                           Se=[34, 78.960, 16], Br=[35, 79.904, 17], Kr=[36, 83.800, 18],  Rb=[37, 85.4678, 1],
                           Sr=[38, 87.620, 2], Y=[39, 88.9059, 3], Zr=[40, 91.224, 4], Nb=[41, 92.9064, 5],
                           Mo=[42, 95.940, 6], Tc=[43, 98.000, 7],  Ru=[44, 101.070, 8], Rh=[45, 102.9055, 9],
                           Pd=[46, 106.420, 10], Ag=[47, 107.8682, 11], Cd=[48, 112.411, 12], In=[49, 114.818, 13],
                           Sn=[50, 118.71, 14],  Sb=[51, 121.76, 15], Te=[52, 127.600, 16], I=[53, 126.9045, 17],
                           Xe=[54, 131.293, 18], Cs=[55, 132.9055, 1], Ba=[56, 137.327, 2], La=[57, 138.9055, 3],
                           Ce=[58, 140.116, 4], Pr=[59, 140.9077, 5], Nd=[60, 144.240, 6],
                           Pm=[61, 145.000, 7], Sm=[62, 150.360, 8], Eu=[63, 151.964, 9], Gd=[64, 157.250, 10],
                           Tb=[65, 158.9253, 11], Dy=[66, 162.500, 12], Ho=[67, 164.9303, 13], Er=[68, 167.259, 14],
                           Tm=[69, 168.9342, 15], Yb=[70, 173.04, 16], Lu=[71, 174.967, 17], Hf=[72, 178.49, 18],
                           Ta=[73, 180.9479, 19], W=[74, 183.84, 20], Re=[75, 186.207, 21], Os=[76, 190.23, 22],
                           Ir=[77, 192.217, 23], Pt=[78, 195.078, 24], Au=[79, 196.9665, 25], Hg=[80, 200.59, 26],
                           Tl=[81, 204.3833, 27], Pb=[82, 207.200, 28], Bi=[83, 208.9804, 29], Po=[84, 209.000, 30],
                           At=[85, 210.000, 31], Rn=[86, 222.000, 32], Fr=[87, 223.000, 1], Ra=[88, 226.000, 2],
                           Ac=[89, 227.000, 3], Th=[90, 232.0381, 4], Pa=[91, 231.0359, 5], U=[92, 238.0289, 6],
                           Np=[93, 237.000, 7], Pu=[94, 244.000, 8], Am=[95, 243.000, 9], Cm=[96, 247.000, 10],
                           Bk=[97, 247.000, 11], Cf=[98, 251.000, 12], Es=[99, 252.000, 13], Fm=[100, 257.000, 14],
                           Md=[101, 258.000, 15], No=[102, 259.000, 16], Lr=[103, 262.000, 17],
                           Rf=[104, 261.000, 18], Db=[105, 268.000, 19], Sg=[106, 269.000, 20], Bh=[107, 270.000, 21],
                           Hs=[108, 277.000, 22], Mt=[109, 278.000, 23], Ds=[110, 281.000, 24], Rg=[111, 282.000, 25],
                           Cn=[112, 285.000, 26], Nh=[113, 286.000, 27], Fl=[114, 289.000, 28], Mc=[115, 289.000, 29],
                           Lv=[116, 293.000, 30], Ts=[117, 294.000, 31], Og=[118, 294.000, 32])


def atomic_number_to_element(z):
    for element, info in periodic_table_info.items():
        if int(z) == info[0]:
            return element


def element_to_atomic_number(symbol):
    for element, info in periodic_table_info.items():
        if symbol.capitalize() == element:
            return info[0]


def element_mass(symbol):
    for element, info in periodic_table_info.items():
        if symbol.capitalize() == element:
            return info[1]


def element_valence_electron(symbol):
    for element, info in periodic_table_info.items():
        if symbol.capitalize() == element:
            return info[2]


def center_mass(elements, coordinates):
    cm = [0., 0., 0.]
    m = 0
    for ide, element in enumerate(elements):
        cm += coordinates[ide] * element_mass(element)
        m += element_mass(element)
    cm = cm/m
    return cm


def get_atomic_number(symbol):
    return periodic_table_info[symbol][0]


def get_element_row(symbol):
    if get_atomic_number(symbol) <= 2:
        return 1
    elif get_atomic_number(symbol) <= 10:
        return 2
    elif get_atomic_number(symbol) <= 18:
        return 3
    elif get_atomic_number(symbol) <= 36:
        return 4
    elif get_atomic_number(symbol) <= 54:
        return 5
    elif get_atomic_number(symbol) <= 86:
        return 6


def rotation_matrix(axis, angle):
    import numpy as np

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