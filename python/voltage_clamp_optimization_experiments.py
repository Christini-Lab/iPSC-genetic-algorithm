"""Contains functions to run VC optimization genetic algorithm experiments."""

import ga_configs
import genetic_algorithm_results
import voltage_clamp_optimization


def get_highest_fitness_individual_overall(
        result: genetic_algorithm_results.GAResultVoltageClampOptimization
) -> genetic_algorithm_results.VCOptimizationIndividual:
    """Gets the highest fitness individual across all generations."""
    highest_fitness = 0
    highest_fitness_individual = None
    for i in range(len(result.generations)):
        curr_individual = result.get_high_fitness_individual(generation=i)
        if curr_individual.fitness > highest_fitness:
            highest_fitness = curr_individual.fitness
            highest_fitness_individual = curr_individual
    return highest_fitness_individual


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
        best_all_around = get_highest_fitness_individual_overall(result=result)
        print('Best protocol: {}'.format(best_all_around.protocol))
        print('Best protocol\'s fitness: {}'.format(best_all_around.fitness))

        result.graph_current_contributions(
            individual=best_end,
            title='Best individual currents, generation {}'.format(
                config.max_generations - 1))

        result.graph_current_contributions(
            individual=best_all_around,
            title='Best individual, all generations')

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

        genetic_algorithm_results.graph_vc_individual(
            individual=best_all_around,
            title='Best individual, all generations')
