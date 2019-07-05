import random
import unittest

import ga_config
import genetic_algorithm_result
import protocols


class TestGeneticAlgorithmResult(unittest.TestCase):

    def setUp(self):
        config = ga_config.ParameterTuningConfig(
            population_size=2,
            max_generations=2,
            protocol=protocols.SingleActionPotentialProtocol(),
            tunable_parameters=[
                ga_config.Parameter(name='g_b_na', default_value=1.),
                ga_config.Parameter(name='g_na', default_value=3600.)],
            params_lower_bound=0.9,
            params_upper_bound=1.1,
            mate_probability=0.9,
            mutate_probability=1.,
            gene_swap_probability=0.5,
            gene_mutation_probability=0.15,
            tournament_size=2)
        self.ga_result = genetic_algorithm_result.GeneticAlgorithmResult(
            config=config,
            baseline_trace=None)  # No need for baseline trace for testing.
        generation_one = [
            genetic_algorithm_result.IndividualResult(
                param_set=[0.95, 3671.2302],
                error=0),
            genetic_algorithm_result.IndividualResult(
                param_set=[0.1, 3400.],
                error=10),
        ]
        generation_two = [
            genetic_algorithm_result.IndividualResult(
                param_set=[0.6, 3500.],
                error=25),
            genetic_algorithm_result.IndividualResult(
                param_set=[0.9, 3600.],
                error=29),
        ]
        self.ga_result.generations.append(generation_one)
        self.ga_result.generations.append(generation_two)

    def test_get_individual_raises_value_error(self):
        self.assertRaises(ValueError, self.ga_result.get_individual, -1, 0)
        self.assertRaises(ValueError, self.ga_result.get_individual, 0, 3)

    def test_get_individual_returns_successfully(self):
        individual = self.ga_result.get_individual(generation=0, index=0)

        expected_individual = genetic_algorithm_result.IndividualResult(
                param_set=[0.95, 3671.2302],
                error=0)

        self.assertEqual(individual, expected_individual)

    def test_get_random_individual(self):
        # With seed set to 2, we will select individual at index 0.
        random.seed(2)

        individual = self.ga_result.get_random_individual(generation=0)

        expected_individual = genetic_algorithm_result.IndividualResult(
            param_set=[0.95, 3671.2302],
            error=0)

        self.assertEqual(individual, expected_individual)

    def test_get_best_individual(self):
        individual_first_gen = self.ga_result.get_best_individual(generation=0)
        individual_second_gen = self.ga_result.get_best_individual(generation=1)

        expected_ind_first_gen = genetic_algorithm_result.IndividualResult(
            param_set=[0.95, 3671.2302],
            error=0)
        expected_ind_second_gen = genetic_algorithm_result.IndividualResult(
            param_set=[0.6, 3500.],
            error=25)

        self.assertEqual(individual_first_gen, expected_ind_first_gen)
        self.assertEqual(individual_second_gen, expected_ind_second_gen)

    def test_get_worst_individual(self):
        individual_first_gen = self.ga_result.get_worst_individual(generation=0)
        individual_second_gen = self.ga_result.get_worst_individual(
            generation=1)

        expected_ind_first_gen = genetic_algorithm_result.IndividualResult(
            param_set=[0.1, 3400.],
            error=10)
        expected_ind_second_gen = genetic_algorithm_result.IndividualResult(
            param_set=[0.9, 3600.],
            error=29)

        self.assertEqual(individual_first_gen, expected_ind_first_gen)
        self.assertEqual(individual_second_gen, expected_ind_second_gen)

    def test_get_parameter_scales(self):
        individual = genetic_algorithm_result.IndividualResult(
            param_set=[0.5, 4500.],
            error=29)
        parameter_scaling = self.ga_result.get_parameter_scales(
            individual=individual)

        expected_parameter_scaling = [0.5, 1.25]

        self.assertListEqual(parameter_scaling, expected_parameter_scaling)


if __name__ == '__main__':
    unittest.main()
