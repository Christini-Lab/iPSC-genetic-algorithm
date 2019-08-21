import random
import unittest

import pandas as pd

import ga_configs
import genetic_algorithm_results
import protocols
import trace


class TestGeneticAlgorithmResults(unittest.TestCase):

    def setUp(self):
        config = ga_configs.ParameterTuningConfig(
            population_size=2,
            max_generations=2,
            protocol=protocols.SingleActionPotentialProtocol(),
            tunable_parameters=[
                ga_configs.Parameter(name='g_b_na', default_value=1.),
                ga_configs.Parameter(name='g_na', default_value=3600.)],
            params_lower_bound=0.9,
            params_upper_bound=1.1,
            mate_probability=0.9,
            mutate_probability=1.,
            gene_swap_probability=0.5,
            gene_mutation_probability=0.15,
            tournament_size=2)
        self.ga_result = genetic_algorithm_results.GAResultParameterTuning(
            config=config)
        generation_one = [
            genetic_algorithm_results.ParameterTuningIndividual(
                parameters=[0.95, 3671.2302],
                fitness=0),
            genetic_algorithm_results.ParameterTuningIndividual(
                parameters=[0.1, 3400.0],
                fitness=2),
        ]
        generation_two = [
            genetic_algorithm_results.ParameterTuningIndividual(
                parameters=[0.6, 3500.0],
                fitness=5),
            genetic_algorithm_results.ParameterTuningIndividual(
                parameters=[0.9, 3600.0],
                fitness=9),
        ]
        self.ga_result.generations.append(generation_one)
        self.ga_result.generations.append(generation_two)

    def test_get_individual_raises_value_error(self):
        self.assertRaises(ValueError, self.ga_result.get_individual, -1, 0)
        self.assertRaises(ValueError, self.ga_result.get_individual, 0, 3)

    def test_get_individual_returns_successfully(self):
        individual = self.ga_result.get_individual(generation=0, index=0)

        expected_individual = genetic_algorithm_results.\
            ParameterTuningIndividual(
                parameters=[0.95, 3671.2302],
                fitness=0)

        self.assertEqual(individual, expected_individual)

    def test_get_random_individual(self):
        for i in range(5):
            random.seed(i)
            expected_index = random.randint(
                0, len(self.ga_result.generations[0]) - 1)
            random.seed(i)
            individual = self.ga_result.get_random_individual(generation=0)
            expected_individual = self.ga_result.get_individual(
                generation=0,
                index=expected_index)
            self.assertEqual(individual, expected_individual)

    def test_get_high_fitness_individual(self):
        individual_first_gen = self.ga_result.get_high_fitness_individual(
            generation=0)
        individual_second_gen = self.ga_result.get_high_fitness_individual(
            generation=1)

        expected_ind_first_gen = genetic_algorithm_results.\
            ParameterTuningIndividual(
                parameters=[0.1, 3400.0],
                fitness=2)
        expected_ind_second_gen = genetic_algorithm_results.\
            ParameterTuningIndividual(
                parameters=[0.9, 3600.],
                fitness=9)

        self.assertEqual(individual_first_gen, expected_ind_first_gen)
        self.assertEqual(individual_second_gen, expected_ind_second_gen)

    def test_get_low_fitness_individual(self):
        individual_first_gen = self.ga_result.get_low_fitness_individual(
            generation=0)
        individual_second_gen = self.ga_result.get_low_fitness_individual(
            generation=1)

        expected_ind_first_gen = genetic_algorithm_results.\
            ParameterTuningIndividual(
                parameters=[0.95, 3671.2302],
                fitness=0)
        expected_ind_second_gen = genetic_algorithm_results.\
            ParameterTuningIndividual(
                parameters=[0.6, 3500.0],
                fitness=5)

        self.assertEqual(individual_first_gen, expected_ind_first_gen)
        self.assertEqual(individual_second_gen, expected_ind_second_gen)

    def test_get_parameter_scales(self):
        individual = genetic_algorithm_results.ParameterTuningIndividual(
            parameters=[0.5, 4500.],
            fitness=29)
        parameter_scaling = self.ga_result.get_parameter_scales(
            individual=individual)

        expected_parameter_scaling = [0.5, 1.25]

        self.assertListEqual(parameter_scaling, expected_parameter_scaling)

    def test_calculate_fitness_score_from_contributions(self):
        df_one = pd.DataFrame(
            {'Parameter': ['i_k1', 'i_ks'], 'Max Percent Contribution': [0.6, 0.4]})
        df_two = pd.DataFrame(
            {'Parameter': ['i_k1', 'i_ks'], 'Max Percent Contribution': [0.2, 0.8]})

        fitness_score = genetic_algorithm_results._calc_fitness_score(
            contributions=[df_one, df_two])

        expected_fitness_score = 1.4

        self.assertAlmostEqual(fitness_score, expected_fitness_score)

    def test_get_max_contributions(self):
        df_one = pd.DataFrame(
            {'Parameter': ['i_k1', 'i_ks'],
             'Max Percent Contribution': [0.6, 0.4]})
        df_two = pd.DataFrame(
            {'Parameter': ['i_k1', 'i_ks'],
             'Max Percent Contribution': [0.2, 0.8]})

        max_contributions = genetic_algorithm_results.get_max_contributions(
            contributions=[df_one, df_two])
        expected_max_contributions_dict = {
            'Max Percent Contribution': {'i_k1': 0.6, 'i_ks': 0.8}
        }
        self.assertDictEqual(
            max_contributions.to_dict(),
            expected_max_contributions_dict)


if __name__ == '__main__':
    unittest.main()
