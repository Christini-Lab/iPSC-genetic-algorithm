import unittest

import irregular_pacing


class IrregularPacingTest(unittest.TestCase):

    def test_init_protocol_raises_value_error(self):
        duration = 10
        stimulation_offsets = [1.6, 0.3, 5]

        with self.assertRaises(ValueError):
            irregular_pacing.IrregularPacingProtocol(
                duration=duration,
                stimulation_offsets=stimulation_offsets)

    def test_should_stimulate_returns_true(self):
        stimulation_offsets = [1.2, 0.3, 0.5]
        ip_protocol = irregular_pacing.IrregularPacingProtocol(
            duration=10,
            stimulation_offsets=stimulation_offsets)
        ip_protocol.stimulation_times = [1.2, 0.3, 0.5]

        time_param_list = [1.2001, 0.3004, 0.5]
        for i in time_param_list:
            with self.subTest():
                self.assertTrue(ip_protocol.should_stimulate(i))


if __name__ == '__main__':
    unittest.main()
