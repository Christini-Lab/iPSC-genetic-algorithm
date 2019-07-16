"""Contains functions to run parameter tuning GA experiments and plots results.

Use the functions in this module in the main.py module.
"""
import collections
import random
from typing import Dict, List, Union

from matplotlib import pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

import ga_configs
import parameter_tuning_genetic_algorithm
import genetic_algorithm_results
import paci_2018
import protocols

COLORS = {
    'Single Action Potential': 'green',
    'Irregular Pacing': 'blue',
    'Voltage Clamp': 'red',
    'Combined Protocol': 'black',
}

PROTOCOL_TITLES = {
    protocols.SingleActionPotentialProtocol: 'Single Action Potential',
    protocols.IrregularPacingProtocol: 'Irregular Pacing',
    protocols.VoltageClampProtocol: 'Voltage Clamp',
}
COMBINED_TITLE = 'Combined Protocol'


def _graph_error_over_generation(result, color, label):
    """Graphs the change in error over generations."""
    best_individual_errors = []

    for i in range(len(result.generations)):
        best_individual_errors.append(result.get_high_fitness_individual(i).fitness)

    best_individual_error_line, = plt.plot(
        range(len(result.generations)),
        best_individual_errors,
        label='{}: Best Individual'.format(label),
        color=color)
    return best_individual_error_line


def _plot_error_scatter(
        result: genetic_algorithm_results.GeneticAlgorithmResult,
        color: str) -> None:
    # TODO refactor to make use of class method and pass in color attribute
    x_data = []
    y_data = []
    for i in range(result.config.max_generations):
        for j in range(result.config.population_size):
            x_data.append(j)
            y_data.append(
                result.get_individual(generation=i, index=j).fitness)
    plt.scatter(x_data, y_data, alpha=0.3, color=color)


def generate_error_over_generation_graph(
        results: Dict[str, genetic_algorithm_results.GeneticAlgorithmResult]
) -> None:
    # Check to ensure all config hyper parameters are the same.
    random_result = results[random.choice(list(results.keys()))]
    for i in results.values():
        if not random_result.config != i.config:
            raise ValueError('Results given do not have the same config.')

    legend_handles = []
    for key, val in results.items():
        if key not in COLORS:
            raise ValueError('Please specify a color pairing for the protocol.')
        _plot_error_scatter(result=val, color=COLORS[key])

        best_ind_line = _graph_error_over_generation(
            result=val,
            color=COLORS[key],
            label=key)
        legend_handles.append(best_ind_line)

    plt.legend(handles=legend_handles, loc='upper right')
    hfont = {'fontname': 'Helvetica'}
    plt.xlabel('Generation', **hfont)
    plt.ylabel('Individual', **hfont)
    plt.savefig('figures/error_over_generation.png')


def generate_parameter_scaling_figure(
        results: Dict[str,
                      List[genetic_algorithm_results.GAResultParameterTuning]]
) -> None:
    examples = []
    for key, val in results.items():
        temp_examples = []
        tunable_params = val[0].config.tunable_parameters
        for result in val:
            best_individual = result.get_high_fitness_individual(
                result.config.max_generations - 1)
            temp_examples.append(result.get_parameter_scales(best_individual))
        examples.extend(_make_parameter_scaling_examples(
            params=temp_examples,
            protocol_type=key,
            default_params=tunable_params))

    param_example_df = pd.DataFrame(
        np.array(examples),
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
    plt.savefig('figures/parameter_scaling.png')


def generate_error_strip_plot(
        results: Dict[str,
                      List[genetic_algorithm_results.GeneticAlgorithmResult]]
) -> None:
    df = _generate_error_strip_plot_data_frame(results=results)
    plt.figure()
    sns.stripplot(
        x='Protocol Type',
        y='Error',
        data=df,
        palette='Set2')
    plt.savefig('figures/error_strip_plot.png')


def _generate_error_strip_plot_data_frame(
        results: Dict[str,
                      List[genetic_algorithm_results.GeneticAlgorithmResult]]
) -> pd.DataFrame:
    errors = []
    protocol_types = []
    for key, val in results.items():
        for result in val:
            best_individual = result.get_high_fitness_individual(
                generation=result.config.max_generations - 1)
            errors.append(best_individual.fitness)
            protocol_types.append(key)
    return pd.DataFrame(data={'Error': errors, 'Protocol Type': protocol_types})


def _make_parameter_scaling_examples(
        params: List[List[float]],
        protocol_type: str,
        default_params: List[ga_configs.Parameter]
) -> List[List[Union[float, str, str]]]:
    examples = []
    for i in params:
        for j in range(len(i)):
            examples.append([i[j], default_params[j].name, protocol_type])
    return examples


def run_comparison_experiment(
        configs: List[ga_configs.ParameterTuningConfig],
        iterations: int
) -> Dict[str, List[genetic_algorithm_results.GeneticAlgorithmResult]]:
    """Runs a comparison between all the configs that were passed in."""
    if not _has_equal_hyperparameters(configs=configs):
        raise ValueError('Configs do not have the same hyper parameters.')

    if not _has_unique_protocols(configs=configs):
        raise ValueError('Configs do not have unique protocols.')

    results = collections.defaultdict(list)
    for config in configs:
        protocol_title = PROTOCOL_TITLES[type(config.protocol)]
        if config.secondary_protocol:
            protocol_title = COMBINED_TITLE
        for i in range(iterations):
            print('Running {} GA iteration: {}'.format(protocol_title, i))
            results[protocol_title].append(run_param_tuning_experiment(config))

    generate_parameter_scaling_figure(results=results)
    generate_error_strip_plot(results=results)
    return results


def _has_equal_hyperparameters(
        configs: List[ga_configs.ParameterTuningConfig]) -> bool:
    first_config = configs[0]
    for i in configs:
        if not first_config.has_equal_hyperparameters(i):
            return False
    return True


def _has_unique_protocols(
        configs: List[ga_configs.ParameterTuningConfig]) -> bool:
    protocol_list = []
    for i in configs:
        if i.secondary_protocol:
            protocol_list.append('Combined Protocol')
        else:
            protocol_list.append(type(i.protocol))
    return len(protocol_list) == len(set(protocol_list))


def run_param_tuning_experiment(
        config: ga_configs.ParameterTuningConfig,
        full_output: bool=False
) -> genetic_algorithm_results.GeneticAlgorithmResult:
    ga = parameter_tuning_genetic_algorithm.ParameterTuningGeneticAlgorithm(
        config=config)
    ga_result = ga.run()

    if full_output:
        ga_result.generate_heatmap()
        ga_result.graph_error_over_generation(with_scatter=False)

        random_0 = ga_result.get_random_individual(generation=0)
        worst_0 = ga_result.get_high_fitness_individual(generation=0)
        best_0 = ga_result.get_low_fitness_individual(generation=0)
        best_middle = ga_result.get_low_fitness_individual(
            generation=config.max_generations // 2)
        best_end = ga_result.get_low_fitness_individual(
            generation=config.max_generations - 1)

        ga_result.graph_individual_with_param_set(
            individual=random_0,
            title='Random individual, generation 0')
        plt.savefig('figures/Random individual, generation 0.png')

        ga_result.graph_individual_with_param_set(
            individual=worst_0,
            title='Worst individual, generation 0')
        plt.savefig('figures/Worst individual, generation 0')

        ga_result.graph_individual_with_param_set(
            individual=best_0,
            title='Best individual, generation 0')
        plt.savefig('figures/Best individual, generation 0')

        ga_result.graph_individual_with_param_set(
            individual=best_middle,
            title='Best individual, generation {}'.format(
                config.max_generations // 2))
        plt.savefig('figures/Best individual, generation {}'.format(
                config.max_generations // 2))

        ga_result.graph_individual_with_param_set(
            individual=best_end,
            title='Best individual, generation {}'.format(
                config.max_generations - 1))
        plt.savefig('figures/Best individual, generation {}'.format(
                config.max_generations - 1))
    return ga_result


def plot_baseline_single_action_potential_trace():
    trace = paci_2018.generate_trace(
        protocol=protocols.SingleActionPotentialProtocol(duration=30))
    trace.plot()
    plt.savefig('figures/baseline_single_action_potential_trace.png')


def plot_all_in_system_of_equations(duration=10):
    model = paci_2018.PaciModel()
    model.generate_response(
        protocol=protocols.SingleActionPotentialProtocol(duration=duration))

    for i in range(len(model.y_names)):
        plt.figure()
        plt.plot([timestamp[i] for timestamp in model.full_y])
        plt.savefig('figures/{}.png'.format(model.y_names[i]))


def plot_baseline_irregular_pacing_trace():
    model = paci_2018.PaciModel()
    trace = model.generate_response(
        protocol=protocols.IrregularPacingProtocol(
            duration=10,
            stimulation_offsets=[0.6, 0.4, 1., 0.1, 0.2, 0.0, 0.8, 0.9]))
    trace.plot()
    trace.pacing_info.plot_peaks_and_apd_ends(trace=trace)
    plt.savefig('figures/baseline_irregular_pacing_trace.png')


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
    plt.savefig('figures/baseline_voltage_clamp_trace.png')

