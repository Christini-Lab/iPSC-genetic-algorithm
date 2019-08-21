import random
import unittest

import ga_configs
import genetic_algorithm_results
import protocols
import voltage_clamp_optimization_experiments


class TestVoltageClampOptimizationExperiments(unittest.TestCase):

    def setUp(self):
        vc_ga_config = ga_configs.VoltageOptimizationConfig(
            window=100,
            step_size=1,
            steps_in_protocol=6,
            step_duration_bounds=(0.1, 2.0),
            step_voltage_bounds=(-1.2, 0.6),
            population_size=2,
            max_generations=2,
            mate_probability=0.9,
            mutate_probability=1.0,
            gene_swap_probability=0.5,
            gene_mutation_probability=0.1,
            tournament_size=3)
        self.ga_result = genetic_algorithm_results. \
            GAResultVoltageClampOptimization(config=vc_ga_config)

        sample_vc_protocol = protocols.VoltageClampProtocol(
            steps=[
                protocols.VoltageClampStep(duration=0.1, voltage=-0.08),
                protocols.VoltageClampStep(duration=0.1, voltage=-0.12),
                protocols.VoltageClampStep(duration=0.5, voltage=-0.06),
                protocols.VoltageClampStep(duration=0.05, voltage=-0.04),
                protocols.VoltageClampStep(duration=0.15, voltage=0.02),
                protocols.VoltageClampStep(duration=0.025, voltage=-0.08),
                protocols.VoltageClampStep(duration=0.3, voltage=0.04),
            ]
        )
        generation_one = [
            genetic_algorithm_results.VCOptimizationIndividual(
                protocol=sample_vc_protocol,
                fitness=0),
            genetic_algorithm_results.VCOptimizationIndividual(
                protocol=sample_vc_protocol,
                fitness=2),
        ]
        generation_two = [
            genetic_algorithm_results.VCOptimizationIndividual(
                protocol=sample_vc_protocol,
                fitness=5),
            genetic_algorithm_results.VCOptimizationIndividual(
                protocol=sample_vc_protocol,
                fitness=9),
        ]
        self.ga_result.generations.append(generation_one)
        self.ga_result.generations.append(generation_two)

    def test_get_highest_fitness_individual_overall(self):
        best_individual = voltage_clamp_optimization_experiments. \
            get_highest_fitness_individual_overall(self.ga_result)

        self.assertEqual(best_individual.fitness, 9)

    def test_construct_optimal_protocol_returns_successfully_fixed_steps(self):
        vc_ga_config = ga_configs.VoltageOptimizationConfig(
            window=100,
            step_size=1,
            steps_in_protocol=-1,  # Will be set later
            step_duration_bounds=(0.05, 0.6),
            step_voltage_bounds=(-.12, .06),
            target_currents=None,  # Will be set later
            population_size=2,
            max_generations=2,
            mate_probability=0.9,
            mutate_probability=0.9,
            gene_swap_probability=0.2,
            gene_mutation_probability=0.2,
            tournament_size=2)
        vc_optimization_config = ga_configs.CombinedVCConfig(
            currents=['i_na', 'i_k1', 'i_kr'],
            step_range=range(5, 6, 1),
            adequate_fitness_threshold=0.4,
            ga_config=vc_ga_config)

        random.seed(2)
        optimized_vc_protocol = voltage_clamp_optimization_experiments.\
            construct_optimal_protocol(
                vc_protocol_optimization_config=vc_optimization_config)
        self.assertEqual(len(optimized_vc_protocol.steps), 18)
        self.assertEqual(
            optimized_vc_protocol.steps[0],
            protocols.VoltageClampProtocol.HOLDING_STEP)
        self.assertEqual(
            optimized_vc_protocol.steps[6],
            protocols.VoltageClampProtocol.HOLDING_STEP)
        self.assertEqual(
            optimized_vc_protocol.steps[12],
            protocols.VoltageClampProtocol.HOLDING_STEP)

    def test_find_single_current_optimal_protocol_exceeds_threshold(self):
        vc_ga_config = ga_configs.VoltageOptimizationConfig(
            window=100,
            step_size=1,
            steps_in_protocol=-1,  # Will be set later
            step_duration_bounds=(0.05, 0.6),
            step_voltage_bounds=(-.12, .06),
            target_currents=None,  # Will be set later
            population_size=2,
            max_generations=2,
            mate_probability=0.9,
            mutate_probability=0.9,
            gene_swap_probability=0.2,
            gene_mutation_probability=0.2,
            tournament_size=2)
        vc_optimization_config = ga_configs.CombinedVCConfig(
            currents=['i_na'],
            step_range=range(5, 6, 1),
            adequate_fitness_threshold=0.4,
            ga_config=vc_ga_config)

        random.seed(2)
        optimized_vc_protocol = voltage_clamp_optimization_experiments.\
            find_single_current_optimal_protocol(
                current=vc_optimization_config.currents[0],
                vc_opt_config=vc_optimization_config)

        self.assertEqual(len(optimized_vc_protocol.steps), 6)

    def test_find_single_current_optimal_protocol_below_threshold(self):
        vc_ga_config = ga_configs.VoltageOptimizationConfig(
            window=100,
            step_size=1,
            steps_in_protocol=-1,  # Will be set later
            step_duration_bounds=(0.05, 0.6),
            step_voltage_bounds=(-.12, .06),
            target_currents=None,  # Will be set later
            population_size=2,
            max_generations=2,
            mate_probability=0.9,
            mutate_probability=0.9,
            gene_swap_probability=0.2,
            gene_mutation_probability=0.2,
            tournament_size=2)
        vc_optimization_config = ga_configs.CombinedVCConfig(
            currents=['i_k1'],
            step_range=range(5, 8, 1),
            adequate_fitness_threshold=0.99,  # Very high threshold.
            ga_config=vc_ga_config)

        random.seed(2)
        optimized_vc_protocol = voltage_clamp_optimization_experiments.\
            find_single_current_optimal_protocol(
                current=vc_optimization_config.currents[0],
                vc_opt_config=vc_optimization_config)

        self.assertEqual(len(optimized_vc_protocol.steps), 8)

    def test_combine_protocols(self):
        optimal_protocols = [
            protocols.VoltageClampProtocol(
                steps=[
                    protocols.VoltageClampStep(duration=0.1, voltage=-0.08),
                    protocols.VoltageClampStep(duration=0.1, voltage=-0.12),
                ]),
            protocols.VoltageClampProtocol(
                steps=[
                    protocols.VoltageClampStep(duration=0.5, voltage=-0.06),
                    protocols.VoltageClampStep(duration=0.05, voltage=-0.04),
                ]),
            protocols.VoltageClampProtocol(
                steps=[
                    protocols.VoltageClampStep(duration=0.15, voltage=0.02),
                    protocols.VoltageClampStep(duration=0.025, voltage=-0.08)
                ]),
        ]

        combined_protocol = voltage_clamp_optimization_experiments.\
            combine_protocols(optimal_protocols=optimal_protocols)

        expected_combined_protocol = protocols.VoltageClampProtocol(
            steps=[
                protocols.VoltageClampStep(duration=0.1, voltage=-0.08),
                protocols.VoltageClampStep(duration=0.1, voltage=-0.12),
                protocols.VoltageClampProtocol.HOLDING_STEP,
                protocols.VoltageClampStep(duration=0.5, voltage=-0.06),
                protocols.VoltageClampStep(duration=0.05, voltage=-0.04),
                protocols.VoltageClampProtocol.HOLDING_STEP,
                protocols.VoltageClampStep(duration=0.15, voltage=0.02),
                protocols.VoltageClampStep(duration=0.025, voltage=-0.08)
            ])

        self.assertEqual(combined_protocol, expected_combined_protocol)


if __name__ == '__main__':
    unittest.main()
