from matplotlib import pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

import ga_config
import genetic_algorithm
import paci_2018
import protocols


def run_experiment(config, full_output=False):
    ga = genetic_algorithm.GeneticAlgorithm(config=config)
    ga_result = ga.run()

    if full_output:
        ga_result.generate_heatmap()
        ga_result.graph_error_over_generation()

        random_0 = ga_result.get_random_individual(generation=0)
        worst_0 = ga_result.get_worst_individual(generation=0)
        best_0 = ga_result.get_best_individual(generation=0)

        print('Getting best individual from generation: {}'.format(
            config.max_generations // 2))
        best_middle = ga_result.get_best_individual(
            generation=config.max_generations // 2)
        best_end = ga_result.get_best_individual(
            generation=config.max_generations - 1)
        ga_result.graph_individual_with_param_set(individual=random_0)
        ga_result.graph_individual_with_param_set(individual=worst_0)
        ga_result.graph_individual_with_param_set(individual=best_0)
        ga_result.graph_individual_with_param_set(individual=best_middle)
        ga_result.graph_individual_with_param_set(individual=best_end)
    return ga_result


def generate_parameter_scaling_figure(sap_results, ip_results):
    if len(sap_results) != len(ip_results):
        raise ValueError('Please provide the same count of sap and ip results.')

    results_count = len(sap_results)
    sap_scalings = []
    ip_scalings = []
    for i in range(results_count):
        best_ind_sap = _get_best_individual(sap_results[i])
        sap_scalings.append(sap_results[i].get_parameter_scales(best_ind_sap))
        best_ind_ip = _get_best_individual(ip_results[i])
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


def generate_error_strip_plot(dataframe):
    plt.figure()
    sns.stripplot(
        x='Protocol Type',
        y='Error',
        data=dataframe,
        palette='Set2')


def _generate_error_strip_plot_data_frame(sap_results, ip_results):
    errors_sap = []
    for i in sap_results:
        errors_sap.append(_get_best_individual(i).error)

    errors_ip = []
    for i in ip_results:
        errors_ip.append(_get_best_individual(i).error)

    df_data_dict = dict()
    df_data_dict['Error'] = errors_sap + errors_ip
    df_data_dict['Protocol Type'] = [
        'Single AP' for _ in range(len(errors_sap))
    ] + [
        'Irregular Pacing' for _ in range(len(errors_ip))
    ]
    return pd.DataFrame(df_data_dict)


def _make_parameter_scaling_examples(params, protocol_type, default_params):
    examples = []
    for i in params:
        for j in range(len(i)):
            examples.append([i[j], default_params[j].name, protocol_type])
    return examples


def _get_best_individual(ga_result):
    return ga_result.get_best_individual(
        generation=ga_result.config.max_generations - 1)


def run_comparison_experiment(sap_config, ip_config, iterations):
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
    plt.show()
    generate_error_strip_plot(
        _generate_error_strip_plot_data_frame(
            sap_results=sap_results,
            ip_results=ip_results))
    plt.show()


def plot_baseline_irregular_pacing_trace():
    test_model = paci_2018.PaciModel()
    test_trace = test_model.generate_response(
        protocol=IrregularPacingProtocol(
            duration=10,
            stimulation_offsets=[0.6, 0.4, 1., 0.1, 0.2, 0.0, 0.8, 0.9]))
    test_trace.plot()
    test_trace.plot_apd_ends()
    test_trace.plot_peaks()
    plt.show()


def plot_baseline_voltage_clamp_trace():
    test_model = paci_2018.PaciModel()

    steps = [
        VoltageClampSteps(duration=0.1, voltage=-0.08),
        VoltageClampSteps(duration=0.1, voltage=-0.12),
        VoltageClampSteps(duration=0.5, voltage=-0.06),
        VoltageClampSteps(duration=0.05, voltage=-0.04),
        VoltageClampSteps(duration=0.15, voltage=0.02),
        VoltageClampSteps(duration=0.3, voltage=0.04),
    ]

    test_trace = test_model.generate_response(
        protocol=VoltageClampProtocol(steps=steps))
    test_trace.current_response_info.plot_current_contributions()


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
    ip_protocol = protocols.IrregularPacingProtocol(
            duration=10,
            stimulation_offsets=[0.6, 0.4, 1., 0.1, 0.2, 0.0, 0.8, 0.9])
    steps = [
        protocols.VoltageClampStep(duration=0.1, voltage=-0.08),
        protocols.VoltageClampStep(duration=0.1, voltage=-0.12),
        protocols.VoltageClampStep(duration=0.5, voltage=-0.06),
        protocols.VoltageClampStep(duration=0.05, voltage=-0.04),
        protocols.VoltageClampStep(duration=0.15, voltage=0.02),
        protocols.VoltageClampStep(duration=0.025, voltage=-0.08),
        protocols.VoltageClampStep(duration=0.3, voltage=0.04),
    ]
    vc_protocol = protocols.VoltageClampProtocol(steps=steps)

    ip_config = ga_config.GeneticAlgorithmConfig(
        population_size=2,
        max_generations=2,
        protocol=ip_protocol,
        tunable_parameters=parameters,
        params_lower_bound=0.9,
        params_upper_bound=1.1,
        crossover_probability=0.9,
        parameter_swap_probability=0.5,
        gene_mutation_probability=0.1,
        tournament_size=2)

    test_model = paci_2018.PaciModel()
    test_trace = test_model.generate_response(protocol=vc_protocol)
    plt.plot(test_trace.t, test_trace.y)
    plt.plot(test_trace.t, test_trace.current_response_info.get_current_summed())
    plt.show()


if __name__ == '__main__':
    main()
