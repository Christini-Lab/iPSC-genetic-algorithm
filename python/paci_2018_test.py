import unittest
import paci_2018
import protocols
import numpy.testing as tst
import ga_configs
import pdb


class TestPaci2018(unittest.TestCase):
    def test_generate_trace_SAP(self):
        protocol = protocols.SingleActionPotentialProtocol()

        baseline_trace = paci_2018.generate_trace(protocol)

        self.assertTrue(len(baseline_trace.t) > 100, \
                'Paci errored in less than .4s')
        self.assertTrue(min(baseline_trace.y) < -.01, \
                'baseline Paci min is greater than .01')
        self.assertTrue(max(baseline_trace.y)<.06, \
                'baseline Paci max is greater than .06')

    def test_update_parameters(self):
        protocol = protocols.SingleActionPotentialProtocol()
        tunable_parameters = [
               ga_configs.Parameter(name='G_Na', default_value=3671.2302),
               ga_configs.Parameter(name='G_F', default_value=30.10312),
               ga_configs.Parameter(name='G_Ks', default_value=2.041),
               ga_configs.Parameter(name='G_Kr', default_value=29.8667),
               ga_configs.Parameter(name='G_K1', default_value=28.1492),
               ga_configs.Parameter(name='G_bNa', default_value=0.95),
               ga_configs.Parameter(name='G_NaL', default_value=17.25),
               ga_configs.Parameter(name='G_CaL', default_value=8.635702e-5),
               ga_configs.Parameter(name='G_pCa', default_value=0.4125),
               ga_configs.Parameter(name='G_bCa', default_value=0.727272)]
        new_params = [3671.2,
                      30.1,
                      2.04,
                      29.8,
                      28.1,
                      0.95,
                      17.2,
                      8.63,
                      0.41,
                      0.72] 
        new_params = [x * .7 for x in new_params]

        baseline_trace = paci_2018.generate_trace(protocol)
        new_trace = paci_2018.generate_trace(protocol, \
                    tunable_parameters, new_params)

        tst.assert_raises(AssertionError, tst.assert_array_equal, \
                baseline_trace.y, new_trace.y, \
                'updating parameters does not change trace')

    def test_no_ion_selective(self):


    def test_voltage_protocol(self):
        pass




    def test_stim_protocol(self):
        pass


if __name__ == '__main__':
    unittest.main()
