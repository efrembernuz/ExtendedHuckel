import unittest
from huckelpy import ExtendedHuckel
import numpy as np


class Testehmethod(unittest.TestCase):

    def setUp(self):
        coordinates = [[0.00000, 0.00000, 0.00000],
                       [0.00000, 0.00000, 1.10000],
                       [1.03709, 0.00000, -0.36667],
                       [-0.51855, 0.89815, -0.36667],
                       [-0.51855, -0.89815, -0.36667]]
        symbols = ['Ti', 'H', 'H', 'H', 'H']

        self.tih4 = ExtendedHuckel(coordinates=coordinates, symbols=symbols)

    def test_overlap(self):
        tih4_overlap = [[1., 0., 0., 0., 0.,
                         0., 0., 0., 0., 0.54795242,
                         0.54795213, 0.5479508, 0.5479508],
                        [0., 1., 0., 0., 0.,
                         0., 0., 0., 0., 0.,
                         0.56385007, -0.28192643, -0.28192643],
                        [0., 0., 1., 0., 0.,
                         0., 0., 0., 0., 0.,
                         0., 0.48830822, -0.48830822],
                        [0., 0., 0., 1., 0.,
                         0., 0., 0., 0., 0.59805389,
                         -0.19935291, -0.19935197, -0.19935197],
                        [0., 0., 0., 0., 1.,
                         0., 0., 0., 0., 0.37639193,
                         -0.12546296, -0.12546353, -0.12546353],
                        [0., 0., 0., 0., 0.,
                         1., 0., 0., 0., 0.,
                         -0.20488324, 0.10244157, 0.10244157],
                        [0., 0., 0., 0., 0.,
                         0., 1., 0., 0., 0.,
                         0., -0.17743302, 0.17743302],
                        [0., 0., 0., 0., 0.,
                         0., 0., 1., 0., 0.,
                         0.28974604, -0.14487195, -0.14487195],
                        [0., 0., 0., 0., 0.,
                         0., 0., 0., 1., 0.,
                         0., -0.25092834, 0.25092834],
                        [0.54795242, 0., 0., 0.59805389, 0.37639193,
                         0., 0., 0., 0., 1.,
                         0.16503888, 0.16503804, 0.16503804],
                        [0.54795213, 0.56385007, 0., -0.19935291, -0.12546296,
                         -0.20488324, 0., 0.28974604, 0., 0.16503888,
                         1., 0.16503798, 0.16503798],
                        [0.5479508, -0.28192643, 0.48830822, -0.19935197, -0.12546353,
                         0.10244157, -0.17743302, -0.14487195, -0.25092834, 0.16503804,
                         0.16503798, 1., 0.16503766],
                        [0.5479508, -0.28192643, -0.48830822, -0.19935197, -0.12546353,
                         0.10244157, 0.17743302, -0.14487195, 0.25092834, 0.16503804,
                         0.16503798, 0.16503766, 1.]]

        np.testing.assert_allclose(self.tih4.get_overlap_matrix(), tih4_overlap, rtol=1e-7)

    def test_hamiltonian(self):
        tih4_hamiltonian = [[-0.2396055, -0., -0., -0., -0.,
                             -0., -0., -0., -0., -0.35410067,
                             -0.35410048, -0.35409963, -0.35409963],
                            [-0., -0.13994137, -0., -0., -0.,
                             -0., -0., -0., -0., -0.,
                             -0.33497251, 0.16748709, 0.16748709],
                            [-0., -0., -0.13994137, -0., -0.,
                             -0., -0., -0., -0., -0.,
                             -0., -0.29009454, 0.29009454],
                            [-0., -0., -0., -0.13994137, -0.,
                             -0., -0., -0., -0., -0.35529234,
                             0.11843174, 0.11843118, 0.11843118],
                            [-0., -0., -0., -0., -0.26374979,
                             -0., -0., -0., -0., -0.24854182,
                             0.0828466, 0.08284698, 0.08284698],
                            [-0., -0., -0., -0., -0.,
                             -0.26374979, -0., -0., -0., -0.,
                             0.13528997, -0.06764495, -0.06764495],
                            [-0., -0., -0., -0., -0.,
                             -0., -0.26374979, -0., -0., -0.,
                             -0., 0.11716385, -0.11716385],
                            [-0., -0., -0., -0., -0.,
                             -0., -0., -0.26374979, -0., -0.,
                             -0.19132719, 0.09566289, 0.09566289],
                            [-0., -0., -0., -0., -0.,
                             -0., -0., -0., -0.26374979, -0.,
                             -0., 0.1656948, -0.1656948],
                            [-0.35410067, -0., -0., -0.35529234, -0.24854182,
                             -0., -0., -0., -0., -0.46175507,
                             -0.1333632, -0.13336252, -0.13336252],
                            [-0.35410048, -0.33497251, -0., 0.11843174, 0.0828466,
                             0.13528997, -0., -0.19132719, -0., -0.1333632,
                             -0.46175507, -0.13336247, -0.13336247],
                            [-0.35409963, 0.16748709, -0.29009454, 0.11843118, 0.08284698,
                             -0.06764495, 0.11716385, 0.09566289, 0.1656948, -0.13336252,
                             -0.13336247, -0.46175507, -0.13336221],
                            [-0.35409963, 0.16748709, 0.29009454, 0.11843118, 0.08284698,
                             -0.06764495, -0.11716385, 0.09566289, -0.1656948, -0.13336252,
                             -0.13336247, -0.13336221, -0.46175507]]

        np.testing.assert_allclose(self.tih4.get_hamiltonian_matrix(), tih4_hamiltonian, rtol=1e-7)

    def test_energies(self):
        tih4_energies = [-0.58512015, -0.46521299, -0.46521295, -0.46521274, -0.26374979,
                         -0.26374979, -0.22441406, -0.22441400, -0.22441380, 1.71448118,
                         2.47970697, 2.47973523, 2.47975843]
        np.testing.assert_allclose(self.tih4.get_mo_energies(), tih4_energies, rtol=1e-7)

    # def test_eigenvectors(self):
    #     tih4_eigenvectors = [[1.38526998e-01, -1.21416688e-08, -4.51243478e-17,
    #                           -1.46415219e-08, 2.08987363e-07, -4.24969502e-08,
    #                           3.38474663e-17, 2.26795904e-07, 2.77555756e-17,
    #                           3.57373242e-01, 3.57373155e-01, 3.57372454e-01,
    #                           3.57372454e-01],
    #                          [-1.49083578e-17, 1.58432518e-10, 2.02205402e-01,
    #                           1.76222305e-10, 2.71116151e-10, -1.40728460e-10,
    #                           -1.79608954e-01, 1.99017335e-10, -2.54005572e-01,
    #                           5.57804761e-10, 2.86878591e-10, 5.22601194e-01,
    #                           -5.22601195e-01],
    #                          [-4.27973694e-08, 1.70723583e-01, -2.28196083e-10,
    #                           1.08354569e-01, 1.66702614e-01, -1.51645652e-01,
    #                           2.02694729e-10, 2.14456687e-01, 2.86653895e-10,
    #                           3.42980329e-01, 3.95166512e-01, -3.69074454e-01,
    #                           -3.69074453e-01],
    #                          [1.61015410e-08, -1.08354697e-01, -6.38876018e-11,
    #                           1.70723986e-01, 2.62656147e-01, 9.62465907e-02,
    #                           5.67465384e-11, -1.36112039e-01, 8.02525260e-11,
    #                           5.40399642e-01, -5.03498613e-01, -1.84506822e-02,
    #                           -1.84506819e-02],
    #                          [7.61538022e-10, -1.83961650e-15, -1.61630512e-15,
    #                           1.23887538e-15, -1.13792198e-09, -1.21553611e-03,
    #                           8.16495840e-01, -8.59523563e-04, -5.77349397e-01,
    #                           2.19336888e-11, 2.19324891e-11, 2.19322100e-11,
    #                           2.19341037e-11],
    #                          [5.11535739e-07, -2.22875622e-13, 5.31768164e-16,
    #                           -2.36479043e-13, -7.64356928e-07, -8.16492592e-01,
    #                           -1.21554094e-03, -5.77353991e-01, 8.59516724e-04,
    #                           1.47327292e-08, 1.47327544e-08, 1.47328290e-08,
    #                           1.47328289e-08],
    #                          [1.87832156e-06, 1.47900548e-01, -9.33971491e-11,
    #                           5.03591357e-01, -8.02594127e-01, 1.36090689e-01,
    #                           -8.59398577e-11, -1.92458055e-01, -1.21534723e-10,
    #                           1.44312621e-01, -8.14504070e-03, -6.80840548e-02,
    #                           -6.80840547e-02],
    #                          [9.55852995e-07, 5.03591682e-01, -1.44612565e-10,
    #                           -1.47900781e-01, 2.35713152e-01, 4.63381170e-01,
    #                           -1.33061554e-10, -6.55312822e-01, -1.88182811e-10,
    #                           -4.23833788e-02, 1.50187121e-01, -5.39020219e-02,
    #                           -5.39020218e-02],
    #                          [-6.12152309e-16, -1.65070043e-10, -5.24862447e-01,
    #                           -4.88620384e-11, 7.78730914e-11, -1.51890824e-10,
    #                           -4.82947813e-01, 2.14801504e-10, -6.82991759e-01,
    #                           -1.40025342e-11, -3.99307931e-11, -1.22807737e-01,
    #                           1.22807737e-01],
    #                          [-2.25040405e+00, 8.25374843e-06, -1.34573620e-16,
    #                           7.47117036e-06, 6.77770279e-07, -1.74789692e-06,
    #                           3.68055591e-17, 4.38270141e-07, 5.55111512e-17,
    #                           8.49880714e-01, 8.49881506e-01, 8.49889053e-01,
    #                           8.49889053e-01],
    #                          [2.31247840e-16, 1.09625755e-11, -1.87149813e+00,
    #                           -9.35522904e-13, -5.74262546e-13, -3.88562567e-12,
    #                           6.63338466e-01, 5.49499128e-12, 9.38102822e-01,
    #                           9.99697054e-13, -1.13798703e-11, 1.63318162e+00,
    #                           -1.63318162e+00],
    #                          [-1.30620355e-05, -1.68559002e+00, -9.46705005e-12,
    #                           -8.13222364e-01, -4.99247228e-01, 5.97446717e-01,
    #                           3.35537047e-12, -8.44907015e-01, 4.74537076e-12,
    #                           8.69165274e-01, 1.40878452e+00, -1.13896751e+00,
    #                           -1.13896751e+00],
    #                          [3.44992381e-06, -8.13224420e-01, -5.60604341e-12,
    #                           1.68559588e+00, 1.03480257e+00, 2.88243269e-01,
    #                           1.98667343e-12, -4.07634291e-01, 2.81002305e-12,
    #                           -1.80154133e+00, 1.41996967e+00, 1.90784023e-01,
    #                           1.90784023e-01]]

        # np.testing.assert_allclose(self.tih4.get_eigenvectors(), tih4_eigenvectors, rtol=1e-7)

    def test_multiplicity(self):
        self.assertEqual(self.tih4.get_multiplicity(), 1)
