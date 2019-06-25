import unittest

import voltage_clamp


class TestVoltageClamp(unittest.TestCase):

    def test_voltage_clamp_protocol(self):
        step_one = voltage_clamp.VoltageClampSteps(duration=1.0, voltage=0.02)
        step_two = voltage_clamp.VoltageClampSteps(duration=2.5, voltage=-0.03)
        step_three = voltage_clamp.VoltageClampSteps(duration=0.3, voltage=0.05)
        steps = [step_one, step_two, step_three]
        protocol = voltage_clamp.VoltageClampProtocol(steps=steps)

        self.assertListEqual(protocol.voltage_change_endpoints, [1, 3.5, 3.8])
        self.assertEqual(-0.03, protocol.get_voltage_at_time(time=3.3))
        self.assertEqual(0.02, protocol.get_voltage_at_time(time=0.01))
        self.assertRaises(ValueError, protocol.get_voltage_at_time, 3.9)

    def test_get_time_interval_for_voltage_raises_value_error(self):
        step_one = voltage_clamp.VoltageClampSteps(duration=1.0, voltage=0.02)
        step_two = voltage_clamp.VoltageClampSteps(duration=2.5, voltage=-0.03)
        step_three = voltage_clamp.VoltageClampSteps(duration=0.3, voltage=0.05)
        steps = [step_one, step_two, step_three]
        protocol = voltage_clamp.VoltageClampProtocol(steps=steps)

        self.assertRaises(
            ValueError,
            protocol.get_time_interval_for_voltage, 0.6)

    def test_get_time_interval_for_voltage(self):
        step_one = voltage_clamp.VoltageClampSteps(duration=1.0, voltage=0.02)
        step_two = voltage_clamp.VoltageClampSteps(duration=2.5, voltage=-0.03)
        step_three = voltage_clamp.VoltageClampSteps(duration=0.3, voltage=0.05)
        steps = [step_one, step_two, step_three]
        protocol = voltage_clamp.VoltageClampProtocol(steps=steps)

        self.assertEqual(
            protocol.get_time_interval_for_voltage(voltage=0.02),
            (0.0, 1.0))
        self.assertEqual(
            protocol.get_time_interval_for_voltage(voltage=0.05),
            (3.5, 3.8))


if __name__ == '__main__':
    unittest.main()
