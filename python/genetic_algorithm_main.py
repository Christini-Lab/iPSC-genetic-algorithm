import copy

import seaborn as sns
from matplotlib import pyplot as plt
import pandas as pd
import numpy as np

import ga_config
import genetic_algorithm
from single_action_potential import SingleActionPotentialProtocol
from irregular_pacing import IrregularPacingProtocol


def run_experiment(config):
    ga_result = genetic_algorithm.GeneticAlgorithm(config=config).run()

    # ga_result.generate_heatmap()
    # ga_result.graph_error_over_generation()
    #
    # random_0 = ga_result.get_random_individual(generation=0)
    # worst_0 = ga_result.get_worst_individual(generation=0)
    # best_0 = ga_result.get_best_individual(generation=0)
    #
    # print('Getting best individual from generation: {}'.format(
    #     config.max_generations // 2))
    # best_middle = ga_result.get_best_individual(
    #     generation=config.max_generations // 2)
    # best_end = ga_result.get_best_individual(
    #     generation=config.max_generations - 1)
    #
    # ga_result.graph_individual_with_param_set(individual=random_0)
    # ga_result.graph_individual_with_param_set(individual=worst_0)
    # ga_result.graph_individual_with_param_set(individual=best_0)
    # ga_result.graph_individual_with_param_set(individual=best_middle)
    # ga_result.graph_individual_with_param_set(individual=best_end)
    return ga_result


def generate_parameter_scaling_figure(sap_results, ip_results):
    if len(sap_results) != len(ip_results):
        raise ValueError('Please provide the same count of sap and ip results.')

    results_count = len(sap_results)
    sap_scalings = []
    ip_scalings = []
    for i in range(results_count):
        sap_scalings.append(_get_best_individuals_param_scaling(sap_results[i]))
        ip_scalings.append(_get_best_individuals_param_scaling(ip_results[i]))

    tunable_params = sap_results[0].config.tunable_parameters
    sap_examples = _make_parameter_scaling_examples(
        params=sap_scalings,
        protocol_type='SAP',
        default_params=tunable_params)
    ip_examples = _make_parameter_scaling_examples(
        params=ip_scalings,
        protocol_type='IP',
        default_params=tunable_params)

    param_example_df = pd.DataFrame(
        np.array(sap_examples + ip_examples),
        columns=['Parameter Value', 'Parameter Type', 'Protocol Type'])
    # Convert parameter value column, which is defaulted to object, to numeric
    # type.
    param_example_df['Parameter Value'] = pd.to_numeric(
        param_example_df['Parameter Value'])

    ax = sns.stripplot(
        x='Parameter Type',
        y='Parameter Value',
        hue='Protocol Type',
        data=param_example_df,
        palette="Set2",
        dodge=True)
    for i in range(0, len(tunable_params), 2):
        ax.axvspan(i + 0.5, i + 1.5, facecolor='lightgrey', alpha=0.3)


def _make_parameter_scaling_examples(params, protocol_type, default_params):
    examples = []
    for i in params:
        for j in range(len(i)):
            examples.append([i[j], default_params[j].name, protocol_type])
    return examples


def _get_best_individuals_param_scaling(ga_result):
    best_individual = ga_result.get_best_individual(
        generation=ga_result.config.max_generations - 1)
    return ga_result.get_parameter_scales(individual=best_individual)


def main():
    parameters = [
        ga_config.Parameter(name='g_na', default_value=3671.2302),
        ga_config.Parameter(name='g_f_s', default_value=30.10312),
        ga_config.Parameter(name='g_ks_s', default_value=2.041),
        ga_config.Parameter(name='g_kr_s', default_value=29.8667),
        ga_config.Parameter(name='g_k1_s', default_value=28.1492),
        ga_config.Parameter(name='g_b_na', default_value=0.95),
        ga_config.Parameter(name='g_na_lmax', default_value=17.25),
        ga_config.Parameter(name='g_ca_l', default_value=8.635702e-5),
        ga_config.Parameter(name='g_p_ca', default_value=0.4125),
        ga_config.Parameter(name='g_b_ca', default_value=0.727272),
    ]
    # Parameters are sorted alphabetically to maintain order during each
    # generation of the genetic algorithm.
    parameters.sort(key=lambda x: x.name)

    sap_config = ga_config.GeneticAlgorithmConfig(
        population_size=2,
        max_generations=1,
        protocol=SingleActionPotentialProtocol(),
        tunable_parameters=parameters,
        params_lower_bound=0.9,
        params_upper_bound=1.1,
        crossover_probability=0.9,
        parameter_swap_probability=0.5,
        gene_mutation_probability=0.1,
        tournament_size=2)

    single_ap_result = run_experiment(config=sap_config)

    ip_config = ga_config.GeneticAlgorithmConfig(
        population_size=2,
        max_generations=1,
        protocol=IrregularPacingProtocol(
            duration=10,
            stimulation_offsets=[0.6, 0.4, 1., 0.1, 0.2, 0.0, 0.8, 0.9]),
        tunable_parameters=parameters,
        params_lower_bound=0.9,
        params_upper_bound=1.1,
        crossover_probability=0.9,
        parameter_swap_probability=0.5,
        gene_mutation_probability=0.1,
        tournament_size=2)

    irregular_pacing_result = run_experiment(config=ip_config)

    generate_parameter_scaling_figure(
        [single_ap_result],
        [irregular_pacing_result])
    plt.show()


if __name__ == '__main__':
    main()
