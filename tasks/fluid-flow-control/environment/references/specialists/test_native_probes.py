"""Check the public 3D PPO-example sensor coordinates and ordering, without CFD."""

import unittest

import numpy as np

from native import (author_pinball_3d_probes, author_cylinder_1000_probes,
                    author_square_200_probes, paper_coupling, logged_force_coefficients)


class AuthorProbeTests(unittest.TestCase):
    def test_square_200_grid(self):
        x, y = np.meshgrid(np.linspace(1., 8., 8), np.linspace(-.75, .75, 5))
        actual = np.array(author_square_200_probes())
        self.assertEqual(actual.shape, (40, 2))
        np.testing.assert_array_equal(actual, np.stack([x.ravel(), y.ravel()], axis=1))

    def test_square_has_its_own_lift_weight_and_substeps(self):
        self.assertEqual(paper_coupling('SquareCylinder_2D_Re200'), (.2, 225, 200))
        self.assertEqual(paper_coupling('Cylinder_2D_Re1000'), (1., 350, 200))
        self.assertEqual(paper_coupling('Cylinder_2D_Re3900'), (1., 600, 200))
        self.assertEqual(paper_coupling('Pinball_2D_Re100'), (1., 170, 200))
        self.assertEqual(paper_coupling('Pinball_3D_Re150'), (1., 200, 200))
        self.assertEqual(paper_coupling('NACA0012Gust_3D_Re1000_AOA20'), (.25, 640, 25))

    def test_square_lattice_forces_are_converted_not_reported_as_cd(self):
        from types import SimpleNamespace
        from unittest.mock import Mock
        convert = Mock(side_effect=lambda force, **kwargs: force * 2)
        env = SimpleNamespace(Ma=.1, referenceLength=1., dX=.01, zLength=.01,
                              compute_nondim_coefficients=convert)
        np.testing.assert_array_equal(logged_force_coefficients(
            env, 'SquareCylinder_2D_Re200', [[.1, .2]]), [[.2, .4]])
        self.assertEqual(convert.call_args.kwargs['projectionLength'], 100.)
        self.assertAlmostEqual(convert.call_args.kwargs['referenceVelocity'], .1 / np.sqrt(3))
        convert.reset_mock()
        np.testing.assert_array_equal(logged_force_coefficients(
            env, 'Pinball_2D_Re100', [[.1, .2]]), [[.1, .2]])
        convert.assert_not_called()

    def test_cylinder_1000_uses_its_own_public_sensor_grid(self):
        x, y = np.meshgrid(np.linspace(1., 7., 13), np.linspace(-.75, .75, 5))
        expected = np.stack([x.ravel(), y.ravel()], axis=1)
        actual = np.array(author_cylinder_1000_probes())
        self.assertEqual(actual.shape, (65, 2))
        np.testing.assert_array_equal(actual, expected)
        self.assertEqual(len(np.unique(actual, axis=0)), 65)

    def test_matches_public_meshgrid_order(self):
        x, y, z = np.meshgrid(np.linspace(2., 8., 7),
                             np.linspace(-.75, .75, 5),
                             np.linspace(-1.5, 1.5, 5))
        expected = np.stack([x.ravel(), y.ravel(), z.ravel()], axis=1)
        actual = np.array(author_pinball_3d_probes())
        self.assertEqual(actual.shape, (175, 3))
        np.testing.assert_array_equal(actual, expected)
        self.assertEqual(len(np.unique(actual, axis=0)), 175)


if __name__ == "__main__":
    unittest.main()
