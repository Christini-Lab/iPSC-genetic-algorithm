import unittest

import ga_config
import genetic_algorithm
import single_action_potential


class TestGeneticAlgorithm(unittest.TestCase):

    def setUp(self):
        parameters = [
            ga_config.Parameter(name='g_na', default_value=3671.2302),
            ga_config.Parameter(name='g_f_s_per_f', default_value=30.10312),
            ga_config.Parameter(name='g_ks_s_per_f', default_value=2.041),
            ga_config.Parameter(name='g_kr_s_per_f', default_value=29.8667),
            ga_config.Parameter(name='g_k1_s_per_f', default_value=28.1492),
            ga_config.Parameter(name='g_b_na', default_value=0.95),
            ga_config.Parameter(name='g_na_lmax', default_value=17.25),
        ]

        parameters.sort(key=lambda x: x.name)
        self.config = ga_config.GeneticAlgorithmConfig(
            population_size=7,
            max_generations=7,
            protocol=single_action_potential.SingleActionPotentialProtocol(),
            tunable_parameters=parameters,
            params_lower_bound=0.5,
            params_upper_bound=1.5,
            crossover_probability=0.9,
            parameter_swap_probability=0.5,
            gene_mutation_probability=0.1,
            tournament_size=2)
        self.ga = genetic_algorithm.GeneticAlgorithm(config=self.config)

    def test_calculate_error(self):
        low_error_param = [0.97, 30.10312, 20.1492, 29.8667, 2.041, 3671.2302,
                           17.25]
        high_error_param = [0.97, 40.10312, 20.1492, 29.8667, 1.041, 2000.2302,
                            12.25]

        low_error_trace = genetic_algorithm._generate_trace(
            config=self.config,
            params=low_error_param)
        high_error_trace = genetic_algorithm._generate_trace(
            config=self.config,
            params=high_error_param)

        expected_low_error = self.ga._calculate_error(low_error_trace)
        expected_high_error = self.ga._calculate_error(high_error_trace)

        self.assertLess(expected_low_error, expected_high_error)


if __name__ == '__main__':
    unittest.main()
