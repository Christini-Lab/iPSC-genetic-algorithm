import unittest
import random

import numpy as np

import ga_configs
import parameter_tuning_genetic_algorithm
import protocols


class CustomAssertions:

    def assertInBounds(self, bounds, values):
        for i in range(len(bounds)):
            if values[i] < bounds[i][0] or values[i] > bounds[i][1]:
                raise AssertionError('Values exceed bounds.')


class TestParameterTuningGeneticAlgorithm(unittest.TestCase, CustomAssertions):

    def setUp(self):
        config = ga_configs.ParameterTuningConfig(
            population_size=2,
            max_generations=2,
            protocol=protocols.SingleActionPotentialProtocol(),
            tunable_parameters=[
                ga_configs.Parameter(name='g_b_na', default_value=0.95),
                ga_configs.Parameter(name='g_na', default_value=3671.2302)],
            params_lower_bound=0.9,
            params_upper_bound=1.1,
            mutate_probability=1.0,
            mate_probability=0.9,
            gene_swap_probability=0.5,
            gene_mutation_probability=0.15,
            tournament_size=2)
        self.ga = parameter_tuning_genetic_algorithm.\
            ParameterTuningGeneticAlgorithm(config=config)

    def test_evaluate_performance_with_default(self):
        error = self.ga._evaluate_performance()

        # Expected error should be 0, since we are comparing trace with default
        # parameters with itself.
        expected_error = 0

        self.assertAlmostEqual(error, expected_error, 4)

    def test_evaluate_performance_with_varied_parameters(self):
        error = self.ga._evaluate_performance(new_parameters=[0.94, 3000.])

        # Because of parameter variation, error is expected to be > 0.
        self.assertGreater(error, 0)

    def test_evaluate_performance_with_combined_protocol(self):
        config = ga_configs.ParameterTuningConfig(
            population_size=2,
            max_generations=2,
            protocol=protocols.SingleActionPotentialProtocol(),
            tunable_parameters=[
                ga_configs.Parameter(name='g_b_na', default_value=0.95),
                ga_configs.Parameter(name='g_na', default_value=3671.2302)],
            params_lower_bound=0.9,
            params_upper_bound=1.1,
            mutate_probability=1.,
            mate_probability=0.9,
            gene_swap_probability=0.5,
            gene_mutation_probability=0.15,
            tournament_size=2)
        single_ap_ga = parameter_tuning_genetic_algorithm.\
            ParameterTuningGeneticAlgorithm(config=config)
        single_ap_error = single_ap_ga._evaluate_performance(
            new_parameters=[0.94, 3000.])

        config.secondary_protocol = protocols.IrregularPacingProtocol(
            duration=5,
            stimulation_offsets=[0.1, 0.2])
        combined_protocol_ga = parameter_tuning_genetic_algorithm.\
            ParameterTuningGeneticAlgorithm(config=config)
        combined_protocol_error = combined_protocol_ga._evaluate_performance(
            new_parameters=[0.94, 3000.])

        # Combined protocol error should be larger than a single protocol error.
        self.assertGreater(combined_protocol_error, single_ap_error)

    def test_mate(self):
        individual_one = [[0.96, 3400.]]
        individual_two = [[0.92, 3200.]]

        # With seed set to 3, the next to calls to random.random() will return
        # 0.23 and 0.54, respectively. Since the parameter swap probability is
        # set to 0.5, only the first parameter will be swapped.
        random.seed(3)
        self.ga._mate(individual_one, individual_two)

        self.assertListEqual(individual_one, [[0.92, 3400.]])
        self.assertListEqual(individual_two, [[0.96, 3200.]])

    def test_mutate(self):
        individual = [[0.96, 3400.]]

        for i in range(500):
            random.seed(i)
            r_num_one = random.random()
            r_num_two = random.random()
            if r_num_two < self.ga.config.gene_mutation_probability < r_num_one:
                random.seed(i)
                break
        else:
            self.fail(msg='Could not find seed to meet required behavior.')

        np.random.seed(4)
        random_val = np.random.normal(3400)
        np.random.seed(4)

        self.ga._mutate(individual)

        self.assertListEqual(individual, [[0.96, random_val]])

    def test_initialize_parameters(self):
        param_bounds = []
        for i in self.ga.config.tunable_parameters:
            param_bounds.append(
                (i.default_value * self.ga.config.params_lower_bound,
                 i.default_value * self.ga.config.params_upper_bound))

        for i in range(10):
            new_parameters = self.ga._initialize_parameters()
            self.assertInBounds(bounds=param_bounds, values=new_parameters)

    def test_configure_toolbox(self):
        toolbox = self.ga._configure_toolbox()

        self.assertIn('init_param', toolbox.__dict__)
        self.assertIn('individual', toolbox.__dict__)
        self.assertIn('population', toolbox.__dict__)
        self.assertIn('evaluate', toolbox.__dict__)
        self.assertIn('select', toolbox.__dict__)
        self.assertIn('mate', toolbox.__dict__)
        self.assertIn('mutate', toolbox.__dict__)
        self.assertNotIn('random_key', toolbox.__dict__)


if __name__ == '__main__':
    unittest.main()
