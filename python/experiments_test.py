import unittest

import experiments
import ga_config


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


if __name__ == '__main__':
    unittest.main()
