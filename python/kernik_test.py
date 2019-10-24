import unittest
import kernik
import protocols
import numpy as np
import numpy.testing as tst
import ga_configs
import pdb


class TestKernik(unittest.TestCase):
    def test_generate_trace_SAP(self):
        protocol = protocols.SingleActionPotentialProtocol(1800)

        baseline_model = kernik.KernikModel()
        baseline_model.generate_response(protocol)
        import seaborn as sns
        plt.plot(baseline_model.t, baseline_model.y_voltage)
        pdb.set_trace()

        self.assertTrue(len(baseline_model.t) > 100,
                'Kernik errored in less than .4s')
        self.assertTrue(min(baseline_model.y_voltage) < -10,
                'baseline Kernik min is greater than -.01')
        self.assertTrue(max(baseline_model.y_voltage) < 60,
                'baseline Kernik max is greater than .06')
    
    def test_kernik_py_vs_mat(self):
        protocol = protocols.SingleActionPotentialProtocol(1800)

        baseline_model = kernik.KernikModel()
        baseline_model.generate_response(protocol)
        mat_baseline = np.loadtxt('model_data/original_baseline_3000ms.csv')
        compare_voltage_plots(baseline_model, mat_baseline)


def compare_voltage_plots(individual, original_matlab):
    import matplotlib.pyplot as plt
    plt.plot(original_matlab[:, 0], original_matlab[:, 1], label='Matlab')
    plt.plot(individual.t, individual.y_voltage, label='Python')
    axes = plt.gca()
    axes.set_xlabel('Time (ms)', fontsize=20)
    axes.set_ylabel('Voltage', fontsize=20)
    plt.legend(fontsize=16)
    plt.show()
    pdb.set_trace()


if __name__ == '__main__':
    unittest.main()
