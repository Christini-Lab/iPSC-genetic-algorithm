import unittest

import pandas as pd

import genetic_algorithm_main


class ConfigFake:

    max_generations = 0


class IndividualFake:

    def __init__(self, error):
        self.error = error


class GAResultFake:

    config = ConfigFake()

    def __init__(self, best_individual):
        self.best_individual = best_individual

    def get_best_individual(self, generation):
        del generation
        return self.best_individual


class TestGeneticAlgorithmMain(unittest.TestCase):
    def test_generate_error_strip_plot_data_frame(self):
        sap_results = [
           GAResultFake(best_individual=IndividualFake(error=1.1)),
           GAResultFake(best_individual=IndividualFake(error=2.3)),
           GAResultFake(best_individual=IndividualFake(error=7.4)),
        ]
        ip_results = [
           GAResultFake(best_individual=IndividualFake(error=2.4)),
           GAResultFake(best_individual=IndividualFake(error=6.5)),
           GAResultFake(best_individual=IndividualFake(error=4.5)),
        ]

        dataframe = genetic_algorithm_main._generate_error_strip_plot_data_frame(
            sap_results=sap_results,
            ip_results=ip_results)

        expected_dataframe = pd.DataFrame(
           {'Error': [1.1, 2.3, 7.4, 2.4, 6.5, 4.5],
            'Protocol Type': ['Single AP', 'Single AP', 'Single AP',
                              'Irregular Pacing', 'Irregular Pacing',
                              'Irregular Pacing']})

        pd.testing.assert_frame_equal(dataframe, expected_dataframe)


if __name__ == '__main__':
    unittest.main()
