import unittest

import pandas as pd

import paci_2018


class TestPaci2018(unittest.TestCase):

    def test_add_currents_timestep(self):
        current_response_info = paci_2018.CurrentResponseInfo()
        timestep_1 = [paci_2018.Current(name='i_na', value=10),
                      paci_2018.Current(name='i_ka', value=-1)]
        timestep_2 = [paci_2018.Current(name='i_na', value=3),
                      paci_2018.Current(name='i_ka', value=-4)]
        current_response_info.add_currents_timestep(timestep_1)
        current_response_info.add_currents_timestep(timestep_2)

        self.assertEqual(len(current_response_info._currents), 2)

    def test_calc_frac_contribution_currents(self):
        current_response_info = paci_2018.CurrentResponseInfo()
        timestep_1 = [paci_2018.Current(name='i_na', value=2),
                      paci_2018.Current(name='i_ka', value=8)]
        timestep_2 = [paci_2018.Current(name='i_na', value=-2),
                      paci_2018.Current(name='i_ka', value=2)]
        timestep_3 = [paci_2018.Current(name='i_na', value=-2),
                      paci_2018.Current(name='i_ka', value=4)]
        current_response_info.add_currents_timestep(timestep_1)
        current_response_info.add_currents_timestep(timestep_2)
        current_response_info.add_currents_timestep(timestep_3)
        current_response_info.add_t(t=0.5)
        current_response_info.add_t(t=1.3)
        current_response_info.add_t(t=2.7)
        expected_dataframe = pd.DataFrame(data={'i_na': [.3], 'i_ka': [0.7]})

        dataframe = current_response_info.calc_frac_contribution_currents(
            start_t=0.5,
            end_t=2.7)

        pd.testing.assert_frame_equal(expected_dataframe, dataframe)


if __name__ == '__main__':
    unittest.main()
