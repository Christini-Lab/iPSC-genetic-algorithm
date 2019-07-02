import random
import unittest

import numpy as np
import pandas as pd

import protocols
import voltage_clamp_optimization


class VoltageClampOptimizationTest(unittest.TestCase):

    def test_mate(self):
        vc_ga = voltage_clamp_optimization.VCOGeneticAlgorithm()
        individual_one = protocols.VoltageClampProtocol(
            steps=[
                protocols.VoltageClampStep(voltage=1.0, duration=0.5),
                protocols.VoltageClampStep(voltage=2.0, duration=0.75),
            ]
        )
        individual_two = protocols.VoltageClampProtocol(
            steps=[
                protocols.VoltageClampStep(voltage=5.0, duration=1.0),
                protocols.VoltageClampStep(voltage=-2.0, duration=2.75),
            ]
        )

        # With random seed set to 3, first value of random.random() is 0.237,
        # and second value is 0.544. Therefore we should expect only the second
        # step to switch with MATE_PROBABILITY set to 0.5.
        random.seed(3)

        vc_ga._mate(
            individual_one=individual_one,
            individual_two=individual_two)

        expected_individual_one = protocols.VoltageClampProtocol(
            steps=[
                protocols.VoltageClampStep(voltage=1.0, duration=0.5),
                protocols.VoltageClampStep(voltage=-2.0, duration=2.75),
            ]
        )
        expected_individual_two = protocols.VoltageClampProtocol(
            steps=[
                protocols.VoltageClampStep(voltage=5.0, duration=1.0),
                protocols.VoltageClampStep(voltage=2.0, duration=0.75),
            ]
        )
        self.assertEqual(individual_one, expected_individual_one)
        self.assertEqual(individual_two, expected_individual_two)

    def test_mutate(self):
        vc_ga = voltage_clamp_optimization.VCOGeneticAlgorithm()
        individual = protocols.VoltageClampProtocol(
            steps=[
                protocols.VoltageClampStep(voltage=1.0, duration=0.5),
                protocols.VoltageClampStep(voltage=2.0, duration=0.75),
            ]
        )

        # With random seed set to 3, first value of random.random() is 0.237,
        # and second value is 0.544. Therefore we should expect only the second
        # step to mutate with MUTATION_PROBABILITY set to 0.5. With np.random
        # seed set, the voltage will be mutated to 3.7886 and the duration to
        # 1.1865.
        random.seed(3)
        np.random.seed(3)

        vc_ga._mutate(individual=individual)
        expected_individual = protocols.VoltageClampProtocol(
            steps=[
                protocols.VoltageClampStep(voltage=1.0, duration=0.5),
                protocols.VoltageClampStep(voltage=3.7886, duration=1.1865),
            ]
        )

        self.assertEqual(individual, expected_individual)

    def test_evaluate_returns_zero_error(self):
        vc_ga = voltage_clamp_optimization.VCOGeneticAlgorithm()
        individual = protocols.VoltageClampProtocol(
            steps=[
                # Intentionally set to large voltage.
                protocols.VoltageClampStep(voltage=1000.0, duration=0.5),
                protocols.VoltageClampStep(voltage=3.7886, duration=1.1865),
            ]
        )

        error = vc_ga._evaluate(individual=individual)

        self.assertEqual(error, 0)

    def test_evaluate_returns_normal(self):
        vc_ga = voltage_clamp_optimization.VCOGeneticAlgorithm()
        individual = protocols.VoltageClampProtocol(
            steps=[
                # Intentionally set to large voltage.
                protocols.VoltageClampStep(voltage=0.2, duration=0.1),
                protocols.VoltageClampStep(voltage=-0.3, duration=0.1865),
            ]
        )

        error = vc_ga._evaluate(individual=individual)
        self.assertGreater(error, 0)

    def test_calculate_fitness_score_from_contributions(self):
        df_one = pd.DataFrame(
            {'Parameter': ['i_k1', 'i_ks'], 'Percent Contribution': [0.6, 0.4]})
        df_two = pd.DataFrame(
            {'Parameter': ['i_k1', 'i_ks'], 'Percent Contribution': [0.2, 0.8]})

        fitness_score = voltage_clamp_optimization._calc_fitness_score(
            contributions=[df_one, df_two])

        expected_fitness_score = 1.4

        self.assertAlmostEqual(fitness_score, expected_fitness_score)


if __name__ == '__main__':
    unittest.main()
