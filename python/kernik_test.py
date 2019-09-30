import unittest
import kernik
import protocols
import numpy.testing as tst
import ga_configs
import pdb


class TestKernik(unittest.TestCase):
    def test_generate_trace_SAP(self):
        protocol = protocols.SingleActionPotentialProtocol()

        baseline_trace = kernik.generate_trace(protocol)

        self.assertTrue(len(baseline_trace.t) > 100,
                'Kernik errored in less than .4s')
        self.assertTrue(min(baseline_trace.y) < -.01,
                'baseline Kernik min is greater than -.01')
        self.assertTrue(max(baseline_trace.y) < .06,
                'baseline Kernik max is greater than .06')


if __name__ == '__main__':
    unittest.main()

