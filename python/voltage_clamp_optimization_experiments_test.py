import unittest

import ga_configs
import genetic_algorithm_results
import protocols
import voltage_clamp_optimization_experiments


class TestVoltageClampOptimizationExperiments(unittest.TestCase):

    def setUp(self):
        vc_ga_config = ga_configs.VoltageOptimizationConfig(
            contribution_step=100,
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
        self.ga_result = genetic_algorithm_results.\
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
        best_individual = voltage_clamp_optimization_experiments.\
            get_highest_fitness_individual_overall(self.ga_result)

        self.assertEqual(best_individual.fitness, 9)


if __name__ == '__main__':
    unittest.main()
