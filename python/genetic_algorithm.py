"""Runs a genetic algorithm on the specified target objective.

Example usage:
    config = <GENERATE CONFIG OBJECT>
    genetic_algorithm_instance = GeneticAlgorithm(config)
    genetic_algorithm_instance.run()
"""
import math
import random

from deap import base
from deap import creator
from deap import tools
import numpy as np
from matplotlib import pyplot as plt
from matplotlib.colors import LogNorm
import scipy.interpolate
import seaborn as sns

import paci_2018


class GeneticAlgorithm:
    """Encapsulates state and behavior of a genetic algorithm.

    Attributes:
        config: A ga_config.GeneticAlgorithmConfig object specifying genetic
            algorithm hyperparameters.
        baseline_trace: The baseline trace which will serve as the algorithm's
            target objective.
    """

    def __init__(self, config):
        self.config = config
        self.baseline_trace = _generate_trace(config)

    def run(self):
        """Runs an instance of the genetic algorithm."""
        toolbox = self._configure_toolbox()
        population = toolbox.population(self.config.population_size)
        ga_result = GeneticAlgorithmResult(
            config=self.config,
            baseline_trace=self.baseline_trace)

        print('Evaluating initial population.')
        for individual in population:
            individual.fitness.values = [toolbox.evaluate(self, individual[0])]

        # Store initial population details for result processing.
        initial_population = []
        for i in range(len(population)):
            initial_population.append(
                IndividualResult(
                    param_set=population[i][0],
                    error=population[i].fitness.values[0]))
        ga_result.generations.append(initial_population)

        for generation in range(1, self.config.max_generations):
            print('Generation {}'.format(generation))
            # Offspring are chosen through tournament selection. They are then
            # cloned, because they will be modified in-place later on.
            selected_offspring = toolbox.select(population, len(population))
            offspring = [toolbox.clone(i) for i in selected_offspring]

            for i_one, i_two in zip(offspring[::2], offspring[1::2]):
                if random.random() < self.config.crossover_probability:
                    toolbox.mate(self, i_one, i_two)
                    del i_one.fitness.values
                    del i_two.fitness.values

            for i in offspring:
                toolbox.mutate(self, i)
                del i.fitness.values

            # All individuals who were updated, either through crossover or
            # mutation, will be re-evaluated.
            updated_individuals = [i for i in offspring if not i.fitness.values]
            for individual in updated_individuals:
                individual.fitness.values = [
                    toolbox.evaluate(self, individual[0])
                ]

            population = offspring

            # Store intermediate population details for result processing.
            intermediate_population = []
            for i in range(len(population)):
                intermediate_population.append(
                    IndividualResult(
                        param_set=population[i][0],
                        error=population[i].fitness.values[0]))
            ga_result.generations.append(intermediate_population)

            _generate_statistics(population)
        return ga_result

    def _evaluate_performance(self, new_parameters):
        """Evaluates performance of an individual compared to the target obj.

        Args:
            new_parameters: New parameters to use in the model.

        Returns:
            The error between the trace generated by the individual's parameter
            set and the baseline target objective.
        """
        return self._calculate_error(
            _generate_trace(self.config, new_parameters))

    def _calculate_error(self, other_trace):
        """Calculates the error between given trace and baseline trace."""
        # Individual could not produce valid trace
        if not other_trace:
            return self.config.MAX_ERROR

        err = 0
        base_interp = scipy.interpolate.interp1d(
            self.baseline_trace.t,
            self.baseline_trace.y[0])
        for i in range(len(other_trace.t)):
            err += (base_interp(other_trace.t[i]) - other_trace.y[0][i]) ** 2
        return err

    def _mate(self, i_one, i_two):
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

    def _mutate(self, individual):
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

    def _initialize_parameters(self):
        """Initializes random values within constraints of all tunable params.

        Returns:
            A new set of randomly generated parameter values.
        """
        # Builds a list of parameters using random upper and lower bounds.
        randomized_parameters = []
        for param in self.config.tunable_parameters:
            random_param = random.uniform(
                param.default_value * self.config.params_lower_bound,
                param.default_value * self.config.params_upper_bound)
            randomized_parameters.append(random_param)
        return randomized_parameters

    def _configure_toolbox(self):
        """Configures toolbox functions."""
        creator.create('FitnessMin', base.Fitness, weights=(-1.0,))
        creator.create('Individual', list, fitness=creator.FitnessMin)

        toolbox = base.Toolbox()
        toolbox.register('init_param', self._initialize_parameters)
        toolbox.register(
            'individual',
            tools.initRepeat,
            creator.Individual,
            toolbox.init_param,
            n=1)
        toolbox.register('population', tools.initRepeat, list, toolbox.individual)
        toolbox.register('evaluate', GeneticAlgorithm._evaluate_performance)
        toolbox.register('select', tools.selTournament,
                         tournsize=self.config.tournament_size)
        toolbox.register('mate', GeneticAlgorithm._mate)
        toolbox.register('mutate', GeneticAlgorithm._mutate)
        return toolbox


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


def _generate_statistics(population):
    fitness_values = [i.fitness.values[0] for i in population]
    print('  Min error: {}'.format(min(fitness_values)))
    print('  Max error: {}'.format(max(fitness_values)))
    print('  Average error: {}'.format(np.mean(fitness_values)))
    print('  Standard deviation: {}'.format(np.std(fitness_values)))


class GeneticAlgorithmResult:
    """Contains information about a run of a genetic algorithm.

    Attributes:
        config: The config object used in the genetic algorithm run.
        baseline_trace: The baseline trace of the genetic algorithm run.
    """

    def __init__(self, config, baseline_trace):
        self.config = config
        self.baseline_trace = baseline_trace
        self.generations = []

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

    def get_worst_individual(self, generation):
        """Given a generation, returns the individual with the most error."""
        max_error = self.generations[generation][0].error
        max_error_individual = self.get_individual(generation, 0)

        for i in range(len(self.generations[generation])):
            individual = self.get_individual(generation, i)
            error = self.generations[generation][i].error
            if error > max_error:
                max_error = error
                max_error_individual = individual
        return max_error_individual

    def get_random_individual(self, generation):
        """Returns a random individual from the specified generation."""
        if len(self.generations) <= generation < 0:
            raise ValueError('Please enter a valid generation.')
        return self.get_individual(
            generation=generation,
            index=random.randint(0, len(self.generations[generation]) - 1))

    def get_individual(self, generation, index):
        """Returns the individual at generation and index specified."""
        if len(self.generations) <= generation < 0:
            raise ValueError('Please enter a valid generation.')

        if len(self.generations[generation]) <= index < 0:
            raise ValueError('Please enter a valid index.')

        return self.generations[generation][index]

    def generate_heatmap(self):
        """Generates a heatmap showing error of individuals."""
        data = []
        for j in range(len(self.generations[0])):
            row = []
            for i in range(len(self.generations)):
                row.append(self.generations[i][j].error)
            data.append(row)
        data = np.array(data)

        # Display log error in colorbar.
        tick_range = range(
            math.floor(math.log10(data.min().min())),
            1 + math.ceil(math.log10(data.max().max())))
        cbar_ticks = [math.pow(10, i) for i in tick_range]
        log_norm = LogNorm(vmin=data.min().min(), vmax=data.max().max())

        plt.figure()
        ax = sns.heatmap(
            data,
            cmap='RdBu',
            xticklabels=2,
            yticklabels=2,
            norm=log_norm,
            cbar_kws={'ticks': cbar_ticks, 'aspect': 15})

        hfont = {'fontname': 'Helvetica'}
        plt.xlabel('Generation', **hfont)
        plt.ylabel('Individual', **hfont)
        ax.invert_yaxis()
        ax.axhline(linewidth=4, color='black')
        ax.axvline(linewidth=4, color='black')
        ax.collections[0].colorbar.set_label('Error')
        plt.show()

    def graph_individual_with_param_set(self, individual):
        """Graphs an individual and its parameters.

        Graphs an individual's trace on the backdrop of the baseline trace. Also
        displays the scaling of the individual's parameters compared to the
        default parameters which produced the baseline trace.

        Args:
            individual: An individual represented by a list of parameters.

        Returns:
            None.
        """
        plt.figure()
        plt.subplot(1, 2, 1)
        trace = self.graph_individual(individual)
        if trace.pacing_info:
            trace.plot_stimulation_times()

        plt.subplot(1, 2, 2)
        parameter_scaling = self.get_parameter_scales(individual=individual)
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

    def get_parameter_scales(self, individual):
        parameter_scaling = []
        for i in range(len(self.config.tunable_parameters)):
            parameter_scaling.append(
                individual.param_set[i] /
                self.config.tunable_parameters[i].default_value)
        return parameter_scaling

    def graph_individual(self, individual):
        """Graphs an individual's trace."""
        plt.plot(
            self.baseline_trace.t,
            self.baseline_trace.y[0],
            color='black')
        trace = _generate_trace(config=self.config, params=individual.param_set)
        plt.plot(trace.t, trace.y[0], 'b--')
        return trace

    def graph_error_over_generation(self):
        """Graphs the change in error over generations."""
        mean_errors = []
        best_individual_errors = []

        for i in range(len(self.generations)):
            best_individual_errors.append(self.get_best_individual(i).error)
            mean_errors.append(np.mean([j.error for j in self.generations[i]]))

        plt.figure()
        mean_error_line, = plt.plot(
            range(len(self.generations)),
            mean_errors,
            label='Mean Error')
        best_individual_error_line, = plt.plot(
            range(len(self.generations)),
            best_individual_errors,
            label='Best Individual')
        plt.xticks(range(0, len(self.generations), 2))
        hfont = {'fontname': 'Helvetica'}
        plt.xlabel('Generation', **hfont)
        plt.ylabel('Individual', **hfont)
        plt.legend(handles=[mean_error_line, best_individual_error_line])
        plt.show()


class IndividualResult:
    """Represents an individual in the population.

    Attributes:
        param_set: The parameter set, ordered according to labels found in
            the config object the individual is associated with.
        error: The error compared to the target objective.
    """

    def __init__(self, param_set, error):
        self.param_set = param_set
        self.error = error

    def __str__(self):
        return ', '.join([str(i) for i in self.param_set])

    def __repr__(self):
        return ', '.join([str(i) for i in self.param_set])
