import random
import unittest

import numpy as np
import pandas as pd

import protocols
import voltage_clamp_optimization


class VoltageClampOptimizationTest(unittest.TestCase):

    # def test_mate(self):
    #     vc_ga = voltage_clamp_optimization.VCOGeneticAlgorithm()
    #     protocol_one = protocols.VoltageClampProtocol(
    #         steps=[
    #             protocols.VoltageClampStep(voltage=1.0, duration=0.5),
    #             protocols.VoltageClampStep(voltage=2.0, duration=0.75),
    #         ]
    #     )
    #     protocol_two = protocols.VoltageClampProtocol(
    #         steps=[
    #             protocols.VoltageClampStep(voltage=5.0, duration=1.0),
    #             protocols.VoltageClampStep(voltage=-2.0, duration=2.75),
    #         ]
    #     )
    #
    #     # With random seed set to 3, first value of random.random() is 0.237,
    #     # and second value is 0.544. Therefore we should expect only the second
    #     # step to switch with CROSSOVER_PROBABILITY set to 0.5.
    #     random.seed(3)
    #
    #     vc_ga._mate(
    #         i_one=voltage_clamp_optimization.Individual(protocol=protocol_one),
    #         i_two=voltage_clamp_optimization.Individual(protocol=protocol_two))
    #
    #     expected_protocol_one = protocols.VoltageClampProtocol(
    #         steps=[
    #             protocols.VoltageClampStep(voltage=1.0, duration=0.5),
    #             protocols.VoltageClampStep(voltage=-2.0, duration=2.75),
    #         ]
    #     )
    #     expected_protocol_two = protocols.VoltageClampProtocol(
    #         steps=[
    #             protocols.VoltageClampStep(voltage=5.0, duration=1.0),
    #             protocols.VoltageClampStep(voltage=2.0, duration=0.75),
    #         ]
    #     )
    #     self.assertEqual(protocol_one, expected_protocol_one)
    #     self.assertEqual(protocol_two, expected_protocol_two)

    # def test_mutate(self):
    #     vc_ga = voltage_clamp_optimization.VCOGeneticAlgorithm()
    #     protocol = protocols.VoltageClampProtocol(
    #         steps=[
    #             protocols.VoltageClampStep(voltage=1.0, duration=0.5),
    #             protocols.VoltageClampStep(voltage=2.0, duration=0.75),
    #         ]
    #     )
    #
    #     # With random seed set to 3, first value of random.random() is 0.237,
    #     # and second value is 0.544. Therefore we should expect only the second
    #     # step to mutate with GENE_SWAP_PROBABILITY set to 0.5. With np.random
    #     # seed set, the voltage will be mutated to 3.7886 and the duration to
    #     # 1.1865.
    #     random.seed(3)
    #     np.random.seed(3)
    #
    #     vc_ga._mutate(individual=voltage_clamp_optimization.Individual(
    #         protocol=protocol))
    #     expected_protocol = protocols.VoltageClampProtocol(
    #         steps=[
    #             protocols.VoltageClampStep(voltage=1.0, duration=0.5),
    #             protocols.VoltageClampStep(voltage=3.7886, duration=1.1865),
    #         ]
    #     )
    #
    #     self.assertEqual(protocol, expected_protocol)

    def test_evaluate_returns_zero_error(self):
        vc_ga = voltage_clamp_optimization.VCOGeneticAlgorithm()
        protocol = protocols.VoltageClampProtocol(
            steps=[
                # Intentionally set to large voltage.
                protocols.VoltageClampStep(voltage=1000.0, duration=0.5),
                protocols.VoltageClampStep(voltage=3.7886, duration=1.1865),
            ]
        )

        error = vc_ga._evaluate(
            individual=voltage_clamp_optimization.Individual(protocol=protocol))

        self.assertEqual(error, 0)

    def test_evaluate_returns_normal(self):
        vc_ga = voltage_clamp_optimization.VCOGeneticAlgorithm()
        protocol = protocols.VoltageClampProtocol(
            steps=[
                # Intentionally set to large voltage.
                protocols.VoltageClampStep(voltage=0.2, duration=0.1),
                protocols.VoltageClampStep(voltage=-0.3, duration=0.1865),
            ]
        )

        error = vc_ga._evaluate(
            individual=voltage_clamp_optimization.Individual(protocol=protocol))
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

    def test_init_parameters(self):
        vc_ga = voltage_clamp_optimization.VCOGeneticAlgorithm()
        individuals = [vc_ga._init_individual() for _ in range(10)]

        for i in individuals:
            for j in i.protocol.steps:
                self.assertTrue(0. <= j.duration <= 2.)
                self.assertTrue(-1.2 <= j.voltage <= .6)


if __name__ == '__main__':
    unittest.main()
