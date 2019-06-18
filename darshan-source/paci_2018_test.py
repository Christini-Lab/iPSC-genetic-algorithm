import os
import unittest
import random

import paci_2018
import ga_config

TEST_DATA_DIR = 'test_data'
ORIGINAL_SAP = 'original_single_action_potential.txt'
ORIGINAL_SP = 'original_stochastic_pacing.txt'
ORIGINAL_VC = 'original_voltage_clamp.txt'


class TestPaci2018(unittest.TestCase):

    def test_single_action_potential(self):
        expected_trace = _read_in_trace(os.path.join(ORIGINAL_SAP))

        test_trace = paci_2018.PaciModel().generate_response(
            ga_config.SingleActionPotentialProtocol())

        self.assertEqual(expected_trace, test_trace)

    def test_voltage_clamp(self):
        expected_trace = _read_in_trace(ORIGINAL_VC)

        steps = [
            ga_config.VoltageClampSteps(duration=0.1, voltage=-0.08),
            ga_config.VoltageClampSteps(duration=0.1, voltage=-0.12),
            ga_config.VoltageClampSteps(duration=0.5, voltage=-0.06),
            ga_config.VoltageClampSteps(duration=0.05, voltage=-0.04),
            ga_config.VoltageClampSteps(duration=0.15, voltage=0.02),
            ga_config.VoltageClampSteps(duration=0.025, voltage=-0.08),
            ga_config.VoltageClampSteps(duration=0.3, voltage=0.04),
        ]
        protocol = ga_config.VoltageClampProtocol(steps=steps)
        test_trace = paci_2018.PaciModel().generate_response(protocol=protocol)

        self.assertEqual(expected_trace, test_trace)

    def test_stochastic_pacing(self):
        expected_trace = _read_in_trace(ORIGINAL_SP)

        random.seed(3)
        protocol = ga_config.StochasticPacingProtocol(duration=10, stimulations=4)
        test_trace = paci_2018.PaciModel().generate_response(protocol=protocol)

        self.assertEqual(expected_trace, test_trace)


def _read_in_trace(filename, test_dir=TEST_DATA_DIR):
    expected_t = []
    expected_y = []
    with open(os.path.join(test_dir, filename), 'r') as f:
        for line in f:
            expected_t.append(float(line.split(',')[0]))
            expected_y.append(float(line.split(',')[1]))
    return paci_2018.Trace(expected_t, [expected_y])


if __name__ == '__main__':
    unittest.main()
