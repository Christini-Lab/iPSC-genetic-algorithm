import unittest

import experiments
import ga_config
import protocols


class TestGeneticAlgorithmResult(unittest.TestCase):

    def test_make_parameter_scaling_examples(self):
        params = [[1, 2], [1.5, 3]]
        protocol_type = 'SAP'
        default_params = [
            ga_config.Parameter(name='g_na', default_value=15),
            ga_config.Parameter(name='g_ca', default_value=1),
        ]

        examples = experiments._make_parameter_scaling_examples(
            params=params,
            protocol_type=protocol_type,
            default_params=default_params)

        expected_examples = [
            [1, 'g_na', 'SAP'],
            [2, 'g_ca', 'SAP'],
            [1.5, 'g_na', 'SAP'],
            [3, 'g_ca', 'SAP']]

        self.assertListEqual(examples, expected_examples)

    def test_run_comparison_experiment_raises_value_error_unequal_hyparam(self):
        # Hyper parameter changed from 0.5 to 0.7
        configs = generate_config_list()
        configs[0].params_lower_bound = 0.7

        with self.assertRaises(ValueError) as ve:
            experiments.run_comparison_experiment(
                configs=configs,
                iterations=2)

        self.assertEqual(
            'Configs do not have the same hyper parameters.',
            str(ve.exception))

    def test_run_comparing_experiment_raises_value_error_double_protocol(self):
        parameters = [
            ga_config.Parameter(name='g_na', default_value=3671.2302),
            ga_config.Parameter(name='g_f_s', default_value=30.10312),
        ]
        sap_protocol = protocols.SingleActionPotentialProtocol()
        sap_config = ga_config.GeneticAlgorithmConfig(
            population_size=2,
            max_generations=2,
            protocol=sap_protocol,
            tunable_parameters=parameters,
            params_lower_bound=0.5,
            params_upper_bound=1.5,
            crossover_probability=0.9,
            parameter_swap_probability=0.5,
            gene_mutation_probability=0.1,
            tournament_size=2)
        configs = generate_config_list()
        configs.append(sap_config)

        with self.assertRaises(ValueError) as ve:
            experiments.run_comparison_experiment(
                configs=configs,
                iterations=2)

        self.assertEqual(
            'Configs do not have unique protocols.',
            str(ve.exception))

    def test_run_comparison_experiment(self):
        configs = generate_config_list()
        results = experiments.run_comparison_experiment(
            configs=configs,
            iterations=2)
        self.assertIn('Single Action Potential', results)
        self.assertIn('Irregular Pacing', results)
        self.assertIn('Voltage Clamp', results)
        self.assertIn('Combined Protocol', results)


def generate_config_list():
    parameters = [
        ga_config.Parameter(name='g_na', default_value=3671.2302),
        ga_config.Parameter(name='g_f_s', default_value=30.10312),
    ]
    sap_protocol = protocols.SingleActionPotentialProtocol()
    ip_protocol = protocols.IrregularPacingProtocol(
        duration=3,
        stimulation_offsets=[0.1])
    vc_protocol = protocols.VoltageClampProtocol(
        steps=[
            protocols.VoltageClampStep(duration=0.1, voltage=-0.08),
            protocols.VoltageClampStep(duration=0.1, voltage=-0.12),
            protocols.VoltageClampStep(duration=0.5, voltage=-0.06),
            protocols.VoltageClampStep(duration=0.05, voltage=-0.04),
            protocols.VoltageClampStep(duration=0.15, voltage=0.02),
            protocols.VoltageClampStep(duration=0.025, voltage=-0.08),
            protocols.VoltageClampStep(duration=0.3, voltage=0.04)])
    sap_config = ga_config.GeneticAlgorithmConfig(
        population_size=2,
        max_generations=2,
        protocol=sap_protocol,
        tunable_parameters=parameters,
        params_lower_bound=0.5,
        params_upper_bound=1.5,
        crossover_probability=0.9,
        parameter_swap_probability=0.5,
        gene_mutation_probability=0.1,
        tournament_size=2)
    ip_config = ga_config.GeneticAlgorithmConfig(
        population_size=2,
        max_generations=2,
        protocol=ip_protocol,
        tunable_parameters=parameters,
        params_lower_bound=0.5,
        params_upper_bound=1.5,
        crossover_probability=0.9,
        parameter_swap_probability=0.5,
        gene_mutation_probability=0.1,
        tournament_size=2)
    vc_config = ga_config.GeneticAlgorithmConfig(
        population_size=2,
        max_generations=2,
        protocol=vc_protocol,
        tunable_parameters=parameters,
        params_lower_bound=0.5,
        params_upper_bound=1.5,
        crossover_probability=0.9,
        parameter_swap_probability=0.5,
        gene_mutation_probability=0.1,
        tournament_size=2)
    combined_config = ga_config.GeneticAlgorithmConfig(
        population_size=2,
        max_generations=2,
        protocol=ip_protocol,
        tunable_parameters=parameters,
        params_lower_bound=0.5,
        params_upper_bound=1.5,
        crossover_probability=0.9,
        parameter_swap_probability=0.5,
        gene_mutation_probability=0.1,
        tournament_size=2,
        secondary_protocol=vc_protocol)

    return [sap_config, ip_config, vc_config, combined_config]


if __name__ == '__main__':
    unittest.main()
