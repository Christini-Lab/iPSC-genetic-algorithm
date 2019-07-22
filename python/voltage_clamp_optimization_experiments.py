"""Contains functions to run VC optimization genetic algorithm experiments."""

import copy
from typing import List

import ga_configs
import genetic_algorithm_results
import protocols
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
        full_output: bool=False
) -> genetic_algorithm_results.GAResultVoltageClampOptimization:
    """Runs a voltage clamp experiment with output if specified."""
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
            title='Best individual currents, all generations')

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
    return result


def construct_optimal_protocol(
        vc_protocol_optimization_config: ga_configs.VCProtocolOptimizationConfig
) -> protocols.VoltageClampProtocol:
    """Constructs the optimal VC protocol to isolate the provided currents.

    Attempts to optimize voltage clamp protocols for a single current and then
    combines them together with a holding current in between.
    """
    optimal_protocols = []
    for i in vc_protocol_optimization_config.currents:
        print('Optimizing current: {}'.format(i))
        optimal_protocols.append(find_single_current_optimal_protocol(
            current=i,
            vc_opt_config=vc_protocol_optimization_config))
    return combine_protocols(optimal_protocols)


def find_single_current_optimal_protocol(
        current: str,
        vc_opt_config: ga_configs.VCProtocolOptimizationConfig,
) -> protocols.VoltageClampProtocol:
    """Runs genetic algorithm to find optimal VC protocol for a single current.

    Protocols of varying step sizes will be generated. The first protocol to
    meet the adequate fitness threshold set in the config parameter will be
    returned. If no such protocol exists, the highest fitness protocol will be
    returned.
    """
    best_individuals = []
    for i in vc_opt_config.step_range:
        print('Trying to optimize with {} steps.'.format(i))
        new_ga_config = copy.deepcopy(vc_opt_config.ga_config)
        new_ga_config.steps_in_protocol = i
        new_ga_config.target_currents = [current]
        result = run_voltage_clamp_experiment(config=new_ga_config)
        best_individual = get_highest_fitness_individual_overall(result=result)

        if best_individual.fitness > vc_opt_config.adequate_fitness_threshold:
            return best_individual.protocol
        best_individuals.append(best_individual)

    best_individuals.sort()
    return best_individuals[-1].protocol


def combine_protocols(
        optimal_protocols: List[protocols.VoltageClampProtocol]
) -> protocols.VoltageClampProtocol:
    """Combines protocols together."""
    combined_protocol = protocols.VoltageClampProtocol()
    for i in optimal_protocols:
        combined_protocol.steps.extend(i.steps)
    return combined_protocol
