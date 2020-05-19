from huckelpy import ExtendedHuckel, file_io


coordinates = [[0.00000, 0.00000, 0.00000],
               [0.00000, 0.00000, 1.10000],
               [1.03709, 0.00000, -0.36667],
               [-0.51855, 0.89815, -0.36667],
               [-0.51855, -0.89815, -0.36667]]
symbols = ['Ti', 'H', 'H', 'H', 'H']

structure = ExtendedHuckel(coordinates=coordinates, symbols=symbols)
file_io.build_output(structure, 'tih4')