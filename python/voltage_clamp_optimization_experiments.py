"""Contains functions to run VC optimization genetic algorithm experiments."""

from matplotlib import pyplot as plt

import ga_configs
import genetic_algorithm_results
import paci_2018
import protocols
import voltage_clamp_optimization


def run_voltage_clamp_experiment(
        config: ga_configs.VoltageOptimizationConfig,
        full_output: bool=False) -> None:
    result = voltage_clamp_optimization.VCOGeneticAlgorithm(config=config).run()

    if full_output:
        result.generate_heatmap()
        result.graph_fitness_over_generation(with_scatter=False)

        random_0 = result.get_random_individual(generation=0)
        worst_0 = result.get_low_fitness_individual(generation=0)
        best_0 = result.get_high_fitness_individual(generation=0)
        best_middle = result.get_high_fitness_individual(
            generation=config.max_generations // 2)
        best_end = result.get_high_fitness_individual(
            generation=config.max_generations - 1)

        genetic_algorithm_results.graph_vc_individual(
            individual=random_0,
            title='Random individual, generation 0')

        genetic_algorithm_results.graph_vc_individual(
            individual=worst_0,
            title='Worst individual, generation 0')

        genetic_algorithm_results.graph_vc_individual(
            individual=best_0,
            title='Best individual, generation 0')

        genetic_algorithm_results.graph_vc_individual(
            individual=best_middle,
            title='Best individual, generation {}'.format(
                config.max_generations // 2))

        genetic_algorithm_results.graph_vc_individual(
            individual=best_end,
            title='Best individual, generation {}'.format(
                config.max_generations - 1))
