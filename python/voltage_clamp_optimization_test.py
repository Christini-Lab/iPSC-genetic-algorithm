import random
import unittest

import numpy as np

import ga_configs
import genetic_algorithm_results
import protocols
import voltage_clamp_optimization


class VoltageClampOptimizationTest(unittest.TestCase):

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
        self.vc_ga = voltage_clamp_optimization.VCOGeneticAlgorithm(
            config=vc_ga_config)

    def test_mate(self):
        protocol_one = protocols.VoltageClampProtocol(
            steps=[
                protocols.VoltageClampStep(voltage=1.0, duration=0.5),
                protocols.VoltageClampStep(voltage=2.0, duration=0.75),
            ]
        )
        protocol_two = protocols.VoltageClampProtocol(
            steps=[
                protocols.VoltageClampStep(voltage=5.0, duration=1.0),
                protocols.VoltageClampStep(voltage=-2.0, duration=2.75),
            ]
        )

        for i in range(500):
            random.seed(i)
            r_num_one = random.random()
            r_num_two = random.random()
            if r_num_two < self.vc_ga.config.gene_swap_probability < r_num_one:
                random.seed(i)
                break
        else:
            self.fail(msg='Could not find seed to meet required behavior.')

        self.vc_ga._mate(
            i_one=genetic_algorithm_results.VCOptimizationIndividual(
                protocol=protocol_one),
            i_two=genetic_algorithm_results.VCOptimizationIndividual(
                protocol=protocol_two))

        expected_protocol_one = protocols.VoltageClampProtocol(
            steps=[
                protocols.VoltageClampStep(voltage=1.0, duration=0.5),
                protocols.VoltageClampStep(voltage=-2.0, duration=2.75),
            ]
        )
        expected_protocol_two = protocols.VoltageClampProtocol(
            steps=[
                protocols.VoltageClampStep(voltage=5.0, duration=1.0),
                protocols.VoltageClampStep(voltage=2.0, duration=0.75),
            ]
        )
        self.assertEqual(protocol_one, expected_protocol_one)
        self.assertEqual(protocol_two, expected_protocol_two)

    def test_mutate(self):
        protocol = protocols.VoltageClampProtocol(
            steps=[
                protocols.VoltageClampStep(voltage=1.0, duration=0.5),
                protocols.VoltageClampStep(voltage=2.0, duration=0.75),
            ]
        )

        # Finds seed so that only second step is mutated.
        for i in range(500):
            random.seed(i)
            num_one = random.random()
            num_two = random.random()
            if num_two < self.vc_ga.config.gene_mutation_probability < num_one:
                random.seed(i)
                break
        else:
            self.fail(msg='Could not find seed to meet required behavior.')

        np.random.seed(3)
        self.vc_ga._mutate(
            individual=genetic_algorithm_results.VCOptimizationIndividual(
                protocol=protocol))
        expected_protocol = protocols.VoltageClampProtocol(
            steps=[
                protocols.VoltageClampStep(voltage=1.0, duration=0.5),
                protocols.VoltageClampStep(
                    voltage=0.6,
                    duration=1.0264562386575933),
            ]
        )

        self.assertEqual(protocol, expected_protocol)

    def test_mutate_steps_falls_between_bounds(self):
        protocol = protocols.VoltageClampProtocol(
            steps=[
                protocols.VoltageClampStep(voltage=-1.0, duration=0.5),
                protocols.VoltageClampStep(voltage=0.4, duration=0.75),
            ]
        )
        individual = genetic_algorithm_results.VCOptimizationIndividual(
            protocol=protocol)
        for _ in range(5000):
            self.vc_ga._mutate(individual=individual)
            for i in individual.protocol.steps:
                self.assertTrue(
                   self.vc_ga.config.step_duration_bounds[0] <= i.duration
                   <= self.vc_ga.config.step_duration_bounds[1])
                self.assertTrue(
                    self.vc_ga.config.step_voltage_bounds[0] <= i.voltage
                    <= self.vc_ga.config.step_voltage_bounds[1])

    def test_evaluate_returns_zero_error(self):
        protocol = protocols.VoltageClampProtocol(
            steps=[
                # Intentionally set to large voltage to cause null trace.
                protocols.VoltageClampStep(voltage=1000.0, duration=0.5),
                protocols.VoltageClampStep(voltage=3.7886, duration=1.1865),
            ]
        )

        error = self.vc_ga._evaluate(
            individual=genetic_algorithm_results.VCOptimizationIndividual(
                protocol=protocol))

        self.assertEqual(error, 0)

    def test_evaluate_returns_normal(self):
        protocol = protocols.VoltageClampProtocol(
            steps=[
                protocols.VoltageClampStep(voltage=0.02, duration=0.1),
                protocols.VoltageClampStep(voltage=-0.03, duration=0.1865),
            ]
        )

        fitness = self.vc_ga._evaluate(
            individual=genetic_algorithm_results.VCOptimizationIndividual(
                protocol=protocol))
        self.assertGreater(fitness, 0)

    def test_init_parameters(self):
        individuals = [self.vc_ga._init_individual() for _ in range(10)]

        for i in individuals:
            for j in i.protocol.steps:
                self.assertTrue(
                    self.vc_ga.config.step_duration_bounds[0] <= j.duration <=
                    self.vc_ga.config.step_duration_bounds[1])
                self.assertTrue(
                    self.vc_ga.config.step_voltage_bounds[0] <= j.voltage <=
                    self.vc_ga.config.step_voltage_bounds[1])

    def test_select(self):
        protocol = protocols.VoltageClampProtocol(
            steps=[
                protocols.VoltageClampStep(voltage=0.2, duration=0.1),
                protocols.VoltageClampStep(voltage=-0.3, duration=0.1865),
            ]
        )

        population = [
            genetic_algorithm_results.VCOptimizationIndividual(
                protocol=protocol, fitness=0),
            genetic_algorithm_results.VCOptimizationIndividual(
                protocol=protocol, fitness=5),
            genetic_algorithm_results.VCOptimizationIndividual(
                protocol=protocol, fitness=1),
            genetic_algorithm_results.VCOptimizationIndividual(
                protocol=protocol, fitness=3.4),
            genetic_algorithm_results.VCOptimizationIndividual(
                protocol=protocol, fitness=8),
            genetic_algorithm_results.VCOptimizationIndividual(
                protocol=protocol, fitness=10),
            genetic_algorithm_results.VCOptimizationIndividual(
                protocol=protocol, fitness=2.2),
            genetic_algorithm_results.VCOptimizationIndividual(
                protocol=protocol, fitness=1.2),
        ]

        random.seed(10)
        new_population = self.vc_ga._select(population=population)

        # Expected population was discovered through printing out
        # random.sample() with seed set to 10.
        expected_population = [
            genetic_algorithm_results.VCOptimizationIndividual(
                protocol=protocol, fitness=3.4),
            genetic_algorithm_results.VCOptimizationIndividual(
                protocol=protocol, fitness=5),
            genetic_algorithm_results.VCOptimizationIndividual(
                protocol=protocol, fitness=2.2),
            genetic_algorithm_results.VCOptimizationIndividual(
                protocol=protocol, fitness=8),
            genetic_algorithm_results.VCOptimizationIndividual(
                protocol=protocol, fitness=1.2),
            genetic_algorithm_results.VCOptimizationIndividual(
                protocol=protocol, fitness=10),
            genetic_algorithm_results.VCOptimizationIndividual(
                protocol=protocol, fitness=5),
            genetic_algorithm_results.VCOptimizationIndividual(
                protocol=protocol, fitness=10),
        ]
        self.assertListEqual(new_population, expected_population)


if __name__ == '__main__':
    unittest.main()
