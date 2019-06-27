"""Contains functions to run genetic algorithm experiments and plot results.

Use the functions in this module in the main.py module.
"""
from typing import List, Union

from matplotlib import pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

import ga_config
import genetic_algorithm
import genetic_algorithm_result
import paci_2018
import protocols


def generate_parameter_scaling_figure(
        sap_results: List[genetic_algorithm_result.GeneticAlgorithmResult],
        ip_results: List[genetic_algorithm_result.GeneticAlgorithmResult]
) -> None:
    if len(sap_results) != len(ip_results):
        raise ValueError('Please provide the same count of sap and ip results.')

    results_count = len(sap_results)
    sap_scalings = []
    ip_scalings = []
    for i in range(results_count):
        best_ind_sap = sap_results[i].get_best_individual(
            sap_results[i].config.max_generations - 1)
        sap_scalings.append(sap_results[i].get_parameter_scales(best_ind_sap))
        best_ind_ip = ip_results[i].get_best_individual(
            ip_results[i].config.max_generations - 1)
        ip_scalings.append(ip_results[i].get_parameter_scales(best_ind_ip))

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
        columns=['Parameter Scaling', 'Parameter', 'Protocol Type'])
    # Convert parameter value column, which is defaulted to object, to numeric
    # type.
    param_example_df['Parameter Scaling'] = pd.to_numeric(
        param_example_df['Parameter Scaling'])

    plt.figure()
    ax = sns.stripplot(
        x='Parameter',
        y='Parameter Scaling',
        hue='Protocol Type',
        data=param_example_df,
        palette='Set2',
        dodge=True)
    for i in range(0, len(tunable_params), 2):
        ax.axvspan(i + 0.5, i + 1.5, facecolor='lightgrey', alpha=0.3)
    plt.show()


def generate_error_strip_plot(
        sap_results: List[genetic_algorithm_result.GeneticAlgorithmResult],
        ip_results: List[genetic_algorithm_result.GeneticAlgorithmResult]
) -> None:
    df = _generate_error_strip_plot_data_frame(sap_results, ip_results)
    plt.figure()
    sns.stripplot(
        x='Protocol Type',
        y='Error',
        data=df,
        palette='Set2')
    plt.show()


def _generate_error_strip_plot_data_frame(sap_results, ip_results):
    errors_sap = []
    for i in sap_results:
        best_ind = i.get_best_individual(
            generation=i.config.max_generations - 1)
        errors_sap.append(best_ind.error)

    errors_ip = []
    for i in ip_results:
        best_ind = i.get_best_individual(
            generation=i.config.max_generations - 1)
        errors_ip.append(best_ind.error)

    df_data_dict = dict()
    df_data_dict['Error'] = errors_sap + errors_ip
    df_data_dict['Protocol Type'] = [
        'Single AP' for _ in range(len(errors_sap))
    ] + [
        'Irregular Pacing' for _ in range(len(errors_ip))
    ]
    return pd.DataFrame(df_data_dict)


def _make_parameter_scaling_examples(
        params: List[List[float]],
        protocol_type: str,
        default_params: List[ga_config.Parameter]
) -> List[List[Union[float, str, str]]]:
    examples = []
    for i in params:
        for j in range(len(i)):
            examples.append([i[j], default_params[j].name, protocol_type])
    return examples


def run_sap_ip_comparison_experiment(
        sap_config: ga_config.GeneticAlgorithmConfig,
        ip_config: ga_config.GeneticAlgorithmConfig,
        iterations: int) -> None:
    if not sap_config.has_equal_hyperparameters(other_config=ip_config):
        raise ValueError('Configs passed in have different hyperparameters.')

    sap_results = []
    for i in range(iterations):
        print('Running SAP GA iteration: {}'.format(i))
        sap_results.append(run_experiment(sap_config))

    ip_results = []
    for i in range(iterations):
        print('Running IP GA iteration: {}'.format(i))
        ip_results.append(run_experiment(ip_config))

    generate_parameter_scaling_figure(
        sap_results=sap_results,
        ip_results=ip_results)
    generate_error_strip_plot(
        sap_results=sap_results,
        ip_results=ip_results)


def run_experiment(
        config: ga_config.GeneticAlgorithmConfig,
        full_output: bool=False
) -> genetic_algorithm_result.GeneticAlgorithmResult:
    ga = genetic_algorithm.GeneticAlgorithm(config=config)
    ga_result = ga.run()

    if full_output:
        ga_result.generate_heatmap()
        ga_result.graph_error_over_generation(with_scatter=True)

        # TODO DEV
        return

        random_0 = ga_result.get_random_individual(generation=0)
        worst_0 = ga_result.get_worst_individual(generation=0)
        best_0 = ga_result.get_best_individual(generation=0)
        best_middle = ga_result.get_best_individual(
            generation=config.max_generations // 2)
        best_end = ga_result.get_best_individual(
            generation=config.max_generations - 1)

        ga_result.graph_individual_with_param_set(
            individual=random_0,
            title='Random individual, generation 0')

        ga_result.graph_individual_with_param_set(
            individual=worst_0,
            title='Worst individual, generation 0')
        ga_result.graph_individual_with_param_set(
            individual=best_0,
            title='Best individual, generation 0')
        ga_result.graph_individual_with_param_set(
            individual=best_middle,
            title='Best individual, generation {}'.format(
                config.max_generations // 2))
        ga_result.graph_individual_with_param_set(
            individual=best_end,
            title='Best individual, generation {}'.format(
                config.max_generations - 1))
    return ga_result


def plot_baseline_single_action_potential_trace():
    model = paci_2018.PaciModel()
    trace = model.generate_response(
        protocol=protocols.SingleActionPotentialProtocol())
    trace.plot()
    plt.show()


def plot_baseline_irregular_pacing_trace():
    model = paci_2018.PaciModel()
    trace = model.generate_response(
        protocol=protocols.IrregularPacingProtocol(
            duration=10,
            stimulation_offsets=[0.6, 0.4, 1., 0.1, 0.2, 0.0, 0.8, 0.9]))
    trace.plot()
    trace.pacing_info.plot_peaks_and_apd_ends(trace=trace)
    plt.show()


def plot_baseline_voltage_clamp_trace():
    model = paci_2018.PaciModel()
    steps = [
        protocols.VoltageClampStep(duration=0.1, voltage=-0.08),
        protocols.VoltageClampStep(duration=0.1, voltage=-0.12),
        protocols.VoltageClampStep(duration=0.5, voltage=-0.06),
        protocols.VoltageClampStep(duration=0.05, voltage=-0.04),
        protocols.VoltageClampStep(duration=0.15, voltage=0.02),
        protocols.VoltageClampStep(duration=0.3, voltage=0.04),
    ]
    trace = model.generate_response(
        protocol=protocols.VoltageClampProtocol(steps=steps))
    trace.plot_with_currents()
    plt.show()
