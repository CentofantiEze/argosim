import numpy as np
import numpy.testing as npt

from argosim import antenna_utils as au


class TestAntennaUtils:

    antenna_lims = {"east": (-500.0, 500.0), "north": (-500.0, 500.0), "up": 0.0}

    circ_antenna_exp = np.array(
        [[300.0, 0.0, 0.0], [-150.0, 259.80762114, 0.0], [-150.0, -259.80762114, 0.0]]
    )
    circ_antenna_tolerance = 1e-7

    def test_random_antenna_pos_default(self):

        antenna_pos = au.random_antenna_pos()

        assert np.logical_and(
            antenna_pos[0] >= self.antenna_lims["east"][0],
            antenna_pos[0] <= self.antenna_lims["east"][1],
        )
        assert np.logical_and(
            antenna_pos[1] >= self.antenna_lims["north"][0],
            antenna_pos[1] <= self.antenna_lims["north"][1],
        )
        assert antenna_pos[2] == self.antenna_lims["up"]

    def test_circular_antenna_arr(self):

        circ_antenna_out = au.circular_antenna_arr()

        npt.assert_allclose(
            circ_antenna_out,
            self.circ_antenna_exp,
            atol=self.circ_antenna_tolerance,
            err_msg="Circular antenna outputs do not match.",
        )
