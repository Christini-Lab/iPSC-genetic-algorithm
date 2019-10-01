import unittest
import kernik
import protocols
import numpy.testing as tst
import ga_configs
import pdb


class TestKernik(unittest.TestCase):
    def test_generate_trace_SAP(self):
        protocol = protocols.SingleActionPotentialProtocol(1800)

        baseline_model = kernik.KernikModel()
        baseline_model.generate_response(protocol)
        pdb.set_trace()

        self.assertTrue(len(baseline_trace.t) > 100,
                'Kernik errored in less than .4s')
        self.assertTrue(min(baseline_trace.y) < -10,
                'baseline Kernik min is greater than -.01')
        self.assertTrue(max(baseline_trace.y) < 60,
                'baseline Kernik max is greater than .06')


if __name__ == '__main__':
    unittest.main()
