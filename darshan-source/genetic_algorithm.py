"""Runs a genetic algorithm on the specified target objective.

Example usage:
    config = <GENERATE CONFIG OBJECT>
    genetic_algorithm_instance = GeneticAlgorithm(config)
    genetic_algorithm_instance.run()
"""
import math
import random
import statistics

from absl import logging
from deap import base
from deap import creator
from deap import tools
import numpy as np
import seaborn as sns
from matplotlib import pyplot as plt
from matplotlib.colors import LogNorm

import paci_2018
import configs


class GeneticAlgorithm:
    """Encapsulates state and behavior of a genetic algorithm.

    Attributes:
        config: A configs.GeneticAlgorithmConfig object specifying genetic
            algorithm hyperparameters.
        baseline_trace: The baseline trace which will serve as the algorithm's
            target objective.
    """

    def __init__(self, config):
        self.config = config
        self.baseline_trace = _generate_trace(config)

    def run(self):
        """Runs the instance of the genetic algorithm."""
        toolbox = _configure_toolbox(self.config)
        population = toolbox.population(self.config.population_size)
        ga_result = GeneticAlgorithmResult(
            config=self.config,
            baseline_trace=self.baseline_trace)

        logging.info('Evaluating initial population.')
        for individual in population:
            individual.fitness.values = [toolbox.evaluate(self, individual[0])]

        initial_population = []
        for i in population:
            initial_population.append(
                Individual(param_set=i[0], error=i.fitness.values[0]))
        ga_result.generations.append(initial_population)

        for generation in range(self.config.max_generations):
            logging.info('Generation %d', generation + 1)
            # Offspring are chosen through tournament selection. They are then
            # cloned, because they will be modified in-place later on.
            selected_offspring = toolbox.select(population, len(population))
            offspring = [toolbox.clone(i) for i in selected_offspring]

            logging.info('Crossing over.')
            for i_one, i_two in zip(offspring[::2], offspring[1::2]):
                if random.random() < self.config.crossover_probability:
                    toolbox.mate(self, i_one, i_two)
                    del i_one.fitness.values
                    del i_two.fitness.values

            logging.info('Mutating.')
            for i in offspring:
                toolbox.mutate(self, i)
                del i.fitness.values

            logging.info('Updating fitness.')
            # All individuals who were updated, either through crossover or
            # mutation, will be re-evaluated.
            updated_individuals = [i for i in offspring if not i.fitness.values]
            for individual in updated_individuals:
                individual.fitness.values = [
                    toolbox.evaluate(self, individual[0])
                ]

            population = offspring
            intermediate_population = []
            for i in population:
                intermediate_population.append(
                    Individual(param_set=i[0], error=i.fitness.values[0]))
            ga_result.generations.append(intermediate_population)
            _generate_statistics(population)
        return ga_result

    def evaluate_performance(self, new_parameters):
        """Evaluates performance of an individual compared to the target obj.

        Args:
            new_parameters: Parameters to be updated.

        Returns:
            The error between the trace generated by the individual's parameter
            set and the baseline target objective.
        """
        return _calculate_error(
            self.baseline_trace,
            _generate_trace(self.config, new_parameters),
            index=self.config.protocol.y_index)

    def mate(self, i_one, i_two):
        """Performs crossover between two individuals.

        There may be a possibility no parameters are swapped. This probability
        is controlled by `self.config.parameter_swap_probability`. Modifies
        both individuals in-place.

        Args:
            i_one: An individual in a population.
            i_two: Another individual in the population.
        """
        for i in range(len(i_one[0])):
            if random.random() < self.config.parameter_swap_probability:
                i_one[0][i], i_two[0][i] = i_two[0][i], i_one[0][i]

    def mutate(self, individual):
        """Performs a mutation on an individual in the population.

        Chooses random parameter values from the normal distribution centered
        around each of the original parameter values. Modifies individual
        in-place.

        Args:
            individual: An individual in the population.
        """
        for i in range(len(individual[0])):
            if random.random() < self.config.gene_mutation_probability:
                individual[0][i] = np.random.normal(individual[0][i])


def _configure_toolbox(config):
    """Configures toolbox functions."""
    creator.create('FitnessMin', base.Fitness, weights=(-1.0,))
    creator.create('Individual', list, fitness=creator.FitnessMin)

    toolbox = base.Toolbox()
    toolbox.register('init_param', lambda: _initialize_parameters(config))
    toolbox.register(
        'individual',
        tools.initRepeat,
        creator.Individual,
        toolbox.init_param,
        n=1)
    toolbox.register('population', tools.initRepeat, list, toolbox.individual)
    toolbox.register('evaluate', GeneticAlgorithm.evaluate_performance)
    toolbox.register('select', tools.selTournament, tournsize=3)
    toolbox.register('mate', GeneticAlgorithm.mate)
    toolbox.register('mutate', GeneticAlgorithm.mutate)
    return toolbox


def _initialize_parameters(config):
    """Initializes random values within constraints of all tunable params.

    Args:
        config: A configs.GeneticAlgorithmConfig object containing all
            parameters to be tuned, as well as the upper and lower boundaries
            of parameter values.

    Returns:
        A new set of randomly generated parameter values.
    """
    # Builds a list of parameters using random upper and lower bounds.
    randomized_parameters = []
    for param in config.tunable_parameters:
        random_param = random.uniform(
            param.default_value * config.params_lower_bound,
            param.default_value * config.params_upper_bound)
        randomized_parameters.append(random_param)
    return randomized_parameters


def _generate_trace(config, params=None):
    """Generates a trace given a set of parameters and config object.

    Leave `params` argument empty if generating baseline trace with
    default parameter values.

    Args:
        config: A configs.GeneticAlgorithmConfig object.
        params: A set of parameter values (where order must match with ordered
            labels in `config.tunable_parameters`).

    Returns:
        A Trace object.
    """
    new_params = dict()
    if params:
        for i in range(len(params)):
            new_params[config.tunable_parameters[i].name] = params[i]
    return paci_2018.PaciModel(
        updated_parameters=new_params).generate_response(
        config.protocol)


def _calculate_error(baseline_trace, other_trace, index):
    # TODO Implement interpolation method.
    error = 0
    for i in range(0, min(len(baseline_trace.t), len(other_trace.t))):
        error += abs(baseline_trace.y[index][i] - other_trace.y[index][i]) ** 2
    return error


def _generate_statistics(population):
    fitness_values = [i.fitness.values[0] for i in population]
    print('  Min error: {}'.format(min(fitness_values)))
    print('  Max error: {}'.format(max(fitness_values)))
    print('  Average error: {}'.format(statistics.mean(fitness_values)))
    print('  Standard deviation: {}'.format(statistics.stdev(fitness_values)))


class GeneticAlgorithmResult:
    """Contains information about a run of a genetic algorithm.

    Attributes:
        config: The config object used with the genetic algorithm run.
        baseline_trace: A Trace object which was the protocol of the
            genetic algorithm run.
    """

    generations = []

    def __init__(self, config, baseline_trace):
        self.config = config
        self.baseline_trace = baseline_trace

    def get_best_individual(self, generation):
        """Given a generation, returns the individual with the least error."""
        min_error = self.generations[generation][0].error
        min_error_individual = self.get_individual(generation, 0)

        for i in range(len(self.generations[generation])):
            individual = self.get_individual(generation, i)
            error = self.generations[generation][i].error
            if error < min_error:
                min_error = error
                min_error_individual = individual
        return min_error_individual

    def random_individual(self, generation):
        """Returns a random individual from the specified generation."""
        if len(self.generations) <= generation < 0:
            raise ValueError('Please enter a valid generation.')
        return self.get_individual(
            generation=generation,
            index=random.randint(0, len(self.generations[generation]) - 1))

    def get_individual(self, generation, index):
        if len(self.generations) <= generation < 0:
            raise ValueError('Please enter a valid generation.')

        if len(self.generations[generation]) <= index < 0:
            raise ValueError('Please enter a valid index.')

        return self.generations[generation][index]

    def generate_heatmap(self):
        data = []
        for j in range(len(self.generations[0])):
            row = []
            for i in range(len(self.generations)):
                row.append(self.generations[i][j].error)
            data.append(row)

        data = np.array(data)
        tick_range = range(
            math.floor(math.log10(data.min().min())),
            1 + math.ceil(math.log10(data.max().max())))
        cbar_ticks = [math.pow(10, i) for i in tick_range]
        log_norm = LogNorm(vmin=data.min().min(), vmax=data.max().max())
        ax = sns.heatmap(
            data,
            cmap='RdBu',
            xticklabels=2,
            yticklabels=2,
            norm=log_norm,
            cbar_kws={'ticks': cbar_ticks})
        ax.set(xlabel='Generation', ylabel='Individual')
        ax.collections[0].colorbar.set_label('Error')

    def graph_individual(self, individual):
        fig = plt.figure()
        plt.subplot(1, 2, 1)
        plt.plot(
            self.baseline_trace.t,
            self.baseline_trace.y[self.config.protocol.y_index],
            color='black')
        trace = _generate_trace(config=self.config, params=individual.param_set)
        plt.plot(trace.t, trace.y[self.config.protocol.y_index], color='green')
        if isinstance(self.config.protocol, configs.VoltageClampProtocol):
            plt.plot(trace.t, trace.y[0], color='blue')

        plt.subplot(1, 2, 2)
        parameter_scaling = []
        for i in range(len(self.config.tunable_parameters)):
            parameter_scaling.append(
                individual.param_set[i] /
                self.config.tunable_parameters[i].default_value)
        parameter_indices = [i for i in range(len(individual.param_set))]
        plt.barh(
            parameter_indices,
            parameter_scaling,
            height=0.2,
            align='center')
        plt.xlabel('Parameter scaling')
        plt.ylabel('Parameters')
        plt.yticks(parameter_indices, parameter_indices)
        plt.xticks([i for i in range(4)], [i for i in range(4)])
        plt.show()


class Individual:
    """Represents an individual in the population.

    Attributes:
        param_set: The parameter set, ordered according to labels found in
            the config object the individual is associated with.
        error: The error compared to the target objective.
    """

    def __init__(self, param_set, error):
        self.param_set = param_set
        self.error = error
