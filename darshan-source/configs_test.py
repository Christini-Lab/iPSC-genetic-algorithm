import unittest

import configs
import random


class TestConfigs(unittest.TestCase):

    def test_stochastic_pacing_protocol(self):
        random.seed(3)
        protocol = configs.StochasticPacingProtocol(duration=10, stimulations=4)

        expected_timestamps = [2.3796462709189137, 5.442292252959518,
                               3.6995516654807927, 6.039200385961944]
        self.assertListEqual(
            protocol.stimulation_timestamps,
            expected_timestamps)

    def test_voltage_clamp_protocol(self):
        step_one = configs.VoltageClampSteps(duration=1.0, voltage=0.02)
        step_two = configs.VoltageClampSteps(duration=2.5, voltage=-0.03)
        step_three = configs.VoltageClampSteps(duration=0.3, voltage=0.05)
        steps = [step_one, step_two, step_three]
        protocol = configs.VoltageClampProtocol(steps=steps)

        self.assertListEqual(protocol.voltage_change_endpoints, [1, 3.5, 3.8])
        self.assertEqual(-0.03, protocol.get_voltage_at_time(time=3.3))
        self.assertEqual(0.02, protocol.get_voltage_at_time(time=0.01))
        self.assertRaises(ValueError, protocol.get_voltage_at_time, 3.9)


if __name__ == '__main__':
    unittest.main()
