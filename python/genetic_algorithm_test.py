import unittest
import random

import numpy as np

import ga_config
import genetic_algorithm
import protocols


class TestGeneticAlgorithm(unittest.TestCase):

    def setUp(self):
        config = ga_config.GeneticAlgorithmConfig(
            population_size=2,
            max_generations=2,
            protocol=protocols.SingleActionPotentialProtocol(),
            tunable_parameters=[
                ga_config.Parameter(name='g_b_na', default_value=0.95),
                ga_config.Parameter(name='g_na', default_value=3671.2302)],
            params_lower_bound=0.9,
            params_upper_bound=1.1,
            crossover_probability=0.9,
            parameter_swap_probability=0.5,
            gene_mutation_probability=0.15,
            tournament_size=2)
        self.ga = genetic_algorithm.GeneticAlgorithm(config=config)

    def test_evaluate_performance_with_default(self):
        error = self.ga._evaluate_performance(new_parameters=None)

        # Expected error should be 0, since we are comparing trace with default
        # parameters with itself.
        expected_error = 0

        self.assertEqual(error, expected_error)

    def test_evaluate_performance_with_varied_parameters(self):
        error = self.ga._evaluate_performance(new_parameters=[0.94, 3000.])

        # Because of parameter variation, error is expected to be > 0.
        self.assertGreater(error, 0)

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

        # With seed set to 4, the next two calls to random.random() will return
        # 0.23 and 0.10, respectively. Because gene mutation probability is set
        # to 0.15, only the second parameter will be mutated.
        random.seed(4)

        # With seed set to 4, the next random number, drawn from the normal
        # distribution centered around 3400., will be 3400.050561707143.
        np.random.seed(4)
        self.ga._mutate(individual)

        self.assertListEqual(individual, [[0.96, 3400.050561707143]])

    def test_initialize_parameters(self):
        # With seed set to 5, random parameters initialized from default values
        # according to lower and upper bounds, will be 0.9733513220290433 and
        # 3848.7613393882134, respectively.
        random.seed(5)
        new_parameters = self.ga._initialize_parameters()

        self.assertListEqual(
            new_parameters,
            [0.9733513220290433, 3848.7613393882134])

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
