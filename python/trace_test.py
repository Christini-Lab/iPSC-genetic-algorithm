import unittest

import pandas as pd

import trace


class TraceFake:

    def __init__(self, t, y):
        self.t = t
        self.y = y


class TraceTest(unittest.TestCase):

    # Tests for IrregularPacingInfo
    def test_add_apd90(self):
        pacing_info = trace.IrregularPacingInfo()
        pacing_info.apd_90_end_voltage = 10
        pacing_info.add_apd_90(apd_90=1.5)

        self.assertEqual(pacing_info.apd_90_end_voltage, -1)

    def test_should_stimulate_returns_false(self):
        pacing_info = trace.IrregularPacingInfo()
        pacing_info.stimulations = [1.1, 1.3005, 1.7]

        test_times = [1.2, 1.0999, 1.8]
        for i in test_times:
            with self.subTest():
                self.assertFalse(pacing_info.should_stimulate(t=i))

    def test_should_stimulate_returns_true(self):
        pacing_info = trace.IrregularPacingInfo()
        pacing_info.stimulations = [1.1, 1.3005, 1.7]

        test_times = [1.10001, 1.3006, 1.7003]
        for i in test_times:
            with self.subTest():
                self.assertTrue(pacing_info.should_stimulate(t=i))

    def test_detect_peak_returns_false_too_close_to_past_peak(self):
        pacing_info = trace.IrregularPacingInfo()
        pacing_info.peaks.append(0.3)

        detected_peak = pacing_info.detect_peak(
            t=[.1, .2, .3],
            y_voltage=0.03,
            d_y_voltage=[1.5, -1.5])

        self.assertFalse(detected_peak)

    def test_detect_peak_returns_false_no_switch_in_d_y(self):
        pacing_info = trace.IrregularPacingInfo()
        pacing_info.peaks.append(0.1)

        detected_peak = pacing_info.detect_peak(
            t=[.1, .2, 3],
            y_voltage=0.03,
            d_y_voltage=[-1.5, -1.5])

        self.assertFalse(detected_peak)

    def test_detect_peak_returns_false_under_voltage_threshold(self):
        pacing_info = trace.IrregularPacingInfo()
        pacing_info.peaks.append(0.1)

        detected_peak = pacing_info.detect_peak(
            t=[.1, .2, 3],
            y_voltage=0.01,
            d_y_voltage=[1.5, -1.5])

        self.assertFalse(detected_peak)

    def test_detect_peak_returns_true(self):
        pacing_info = trace.IrregularPacingInfo()
        pacing_info.peaks.append(0.1)

        detected_peak = pacing_info.detect_peak(
            t=[.1, .2, 3],
            y_voltage=0.03,
            d_y_voltage=[1.5, -1.5])

        self.assertTrue(detected_peak)

    def test_detect_apd90_returns_false_apd_end_not_set(self):
        pacing_info = trace.IrregularPacingInfo()

        detected_apd90 = pacing_info.detect_apd_90(y_voltage=10)

        self.assertFalse(detected_apd90)

    def test_detect_apd90_returns_false_different_y_voltage(self):
        pacing_info = trace.IrregularPacingInfo()
        pacing_info.apd_90_end_voltage = 5

        detected_apd90 = pacing_info.detect_apd_90(y_voltage=10)

        self.assertFalse(detected_apd90)

    def test_detect_apd90_returns_true(self):
        pacing_info = trace.IrregularPacingInfo()
        pacing_info.apd_90_end_voltage = 5

        detected_apd90 = pacing_info.detect_apd_90(y_voltage=5.0001)

        self.assertTrue(detected_apd90)

    def test_find_trace_y_values(self):
        trace_t = [0.5, 1., 1.3, 1.5]
        trace_y = [10, 20, 25, 30]
        trace_fake = TraceFake(t=trace_t, y=trace_y)
        timings = [0.6, 1.1, 1.7]

        y_values = trace._find_trace_y_values(trace_fake, timings)
        expected_y_values = [10, 20, 30]

        self.assertListEqual(y_values, expected_y_values)

    def test_find_closest_t_index(self):
        array = [1, 6, 9]
        t = 7.1

        index = trace._find_closest_t_index(array, t)
        expected_index = 1

        self.assertEqual(index, expected_index)

    def test_calculate_current_contribution(self):
        _ = self  # To remove pycharm warning about static test.
        currents = [
            [
                trace.Current(name='i_na', value=15),
                trace.Current(name='i_ca', value=10),
            ],
            [
                trace.Current(name='i_na', value=20),
                trace.Current(name='i_ca', value=-5),
            ],
        ]
        # Protocol is not needed for this test.
        current_response_info = trace.CurrentResponseInfo(protocol=None)
        current_response_info.currents = currents
        timings = [0., 1.]
        start_t = 0.
        end_t = 1.

        curr_contrib = current_response_info.calculate_current_contribution(
            timings=timings,
            start_t=start_t,
            end_t=end_t)

        expected_curr_contrib = pd.DataFrame(
            data={'Parameter': ['i_na', 'i_ca'],
                  'Max Percent Contribution': [0.7, 0.3]}
        )

        pd.testing.assert_frame_equal(curr_contrib, expected_curr_contrib)

    def test_get_current(self):
        currents = [
            [
                trace.Current(name='i_na', value=15),
                trace.Current(name='i_ca', value=10),
            ],
            [
                trace.Current(name='i_na', value=20),
                trace.Current(name='i_ca', value=-5),
            ],
        ]
        current_response_info = trace.CurrentResponseInfo(protocol=None)
        current_response_info.currents = currents

        i_na_values = current_response_info.get_single_current_values(
            current_name='i_na')
        i_ca_values = current_response_info.get_single_current_values(
            current_name='i_ca')

        expected_i_na_values = [15, 20]
        expected_i_ca_values = [10, -5]

        self.assertListEqual(i_na_values, expected_i_na_values)
        self.assertListEqual(i_ca_values, expected_i_ca_values)


if __name__ == '__main__':
    unittest.main()
