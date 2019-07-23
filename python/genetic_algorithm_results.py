"""Contains classes to store the result of a genetic algorithm run.

Additionally, the classes in this module allow for figure generation.
"""

from abc import ABC
import enum
import math
import random
from typing import List

from matplotlib import pyplot as plt
from matplotlib.colors import LogNorm
import numpy as np
import pandas as pd
import seaborn as sns

import ga_configs
import paci_2018
import protocols
import trace


class ExtremeType(enum.Enum):
    LOW = 1
    HIGH = 2


class GeneticAlgorithmResult(ABC):
    """Contains information about a run of a genetic algorithm.

    Attributes:
        config: The config object used in the genetic algorithm run.
        baseline_trace: The baseline trace of the genetic algorithm run.
        generations: A 2D list of every individual in the genetic algorithm.
    """

    def __init__(self):
        self.config = None
        self.baseline_trace = None
        self.generations = []

    def get_individual(self, generation, index):
        """Returns the individual at generation and index specified."""
        if generation < 0 or generation >= len(self.generations):
            raise ValueError('Please enter a valid generation.')

        if index < 0 or index >= len(self.generations[generation]):
            raise ValueError('Please enter a valid index.')

        return self.generations[generation][index]

    def get_random_individual(self, generation):
        """Returns a random individual from the specified generation."""
        if len(self.generations) <= generation < 0:
            raise ValueError('Please enter a valid generation.')
        return self.get_individual(
            generation=generation,
            index=random.randint(0, len(self.generations[generation]) - 1))

    def get_high_fitness_individual(self, generation):
        """Given a generation, returns the individual with the least error."""
        return self._get_individual_at_extreme(generation, ExtremeType.HIGH)

    def get_low_fitness_individual(self, generation):
        """Given a generation, returns the individual with the most error."""
        return self._get_individual_at_extreme(generation, ExtremeType.LOW)

    def _get_individual_at_extreme(self,
                                   generation: int,
                                   extreme_type: ExtremeType) -> 'Individual':
        """Retrieves either the best or worst individual given a generation."""
        top_error_individual = self.get_individual(generation, 0)
        for i in range(len(self.generations[generation])):
            individual = self.get_individual(generation, i)
            if (extreme_type == ExtremeType.LOW and
                    individual.fitness < top_error_individual.fitness):
                top_error_individual = individual
            elif (extreme_type == ExtremeType.HIGH and
                    individual.fitness > top_error_individual.fitness):
                top_error_individual = individual
        return top_error_individual

    def generate_heatmap(self):
        """Generates a heatmap showing error of individuals."""
        data = []
        for j in range(len(self.generations[0])):
            row = []
            for i in range(len(self.generations)):
                row.append(self.generations[i][j].fitness)
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
        plt.savefig('figures/heatmap.png')

    def plot_error_scatter(self):
        x_data = []
        y_data = []
        for i in range(self.config.max_generations):
            for j in range(self.config.population_size):
                x_data.append(j)
                y_data.append(
                    self.get_individual(generation=i, index=j).fitness)
        plt.scatter(x_data, y_data, alpha=0.3, color='red')


class GAResultParameterTuning(GeneticAlgorithmResult):
    """Contains information about a run of a parameter tuning genetic algorithm.

    Attributes:
        config: The config object used in the genetic algorithm run.
        baseline_trace: The baseline trace of the genetic algorithm run.
    """

    def __init__(self, config: ga_configs.ParameterTuningConfig) -> None:
        super().__init__()
        self.config = config
        self.baseline_trace = paci_2018.generate_trace(
            tunable_parameters=config.tunable_parameters,
            protocol=config.protocol)

    def get_parameter_scales(self, individual):
        parameter_scaling = []
        for i in range(len(self.config.tunable_parameters)):
            parameter_scaling.append(
                individual.parameters[i] /
                self.config.tunable_parameters[i].default_value)
        return parameter_scaling

    def graph_individual(self, individual):
        """Graphs an individual's trace."""
        if isinstance(self.config.protocol, protocols.VoltageClampProtocol):
            self.baseline_trace.plot_only_currents(color='black')
        else:
            plt.plot(
                self.baseline_trace.t,
                self.baseline_trace.y,
                color='black')
        trace = paci_2018.generate_trace(
            tunable_parameters=self.config.tunable_parameters,
            protocol=self.config.protocol,
            params=individual.parameters)
        if trace:
            if isinstance(self.config.protocol, protocols.VoltageClampProtocol):
                trace.plot_only_currents(color='b--')
            else:
                plt.plot(trace.t, trace.y, 'b--')
        return trace

    def graph_individual_with_param_set(self, individual, title=''):
        """Graphs an individual and its parameters.

        Graphs an individual's trace on the backdrop of the baseline trace. Also
        displays the scaling of the individual's parameters compared to the
        default parameters which produced the baseline trace.

        Args:
            individual: An individual represented by a list of parameters.
            title: Title of the graph.

        Returns:
            None.
        """
        plt.figure()
        plt.subplot(1, 2, 1)
        self.graph_individual(individual)

        plt.subplot(1, 2, 2)
        parameter_scaling = self.get_parameter_scales(individual=individual)
        parameter_indices = [i for i in range(len(individual.parameters))]

        plt.barh(
            parameter_indices,
            parameter_scaling,
            height=0.2,
            align='center')
        plt.xlabel('Parameter scaling')
        plt.ylabel('Parameters')
        plt.title(title)
        plt.yticks(parameter_indices, parameter_indices)
        plt.xticks([i for i in range(4)], [i for i in range(4)])

    def graph_error_over_generation(self, with_scatter=False):
        """Graphs the change in error over generations."""
        mean_errors = []
        best_individual_errors = []

        for i in range(len(self.generations)):
            best_individual_errors.append(
                self.get_high_fitness_individual(i).fitness)
            mean_errors.append(
                np.mean([j.fitness for j in self.generations[i]]))

        plt.figure()
        if with_scatter:
            self.plot_error_scatter()
        mean_error_line, = plt.plot(
            range(len(self.generations)),
            mean_errors,
            label='Mean Error')
        best_individual_error_line, = plt.plot(
            range(len(self.generations)),
            best_individual_errors,
            label='Best Individual')
        plt.xticks(range(len(self.generations)))
        hfont = {'fontname': 'Helvetica'}
        plt.xlabel('Generation', **hfont)
        plt.ylabel('Individual', **hfont)
        plt.legend(handles=[mean_error_line, best_individual_error_line])
        plt.savefig('figures/error_over_generation.png')


class GAResultVoltageClampOptimization(GeneticAlgorithmResult):
    """Contains information about a run of a parameter tuning genetic algorithm.

    Attributes:
        config: The config object used in the genetic algorithm run.
    """

    def __init__(self, config: ga_configs.VoltageOptimizationConfig) -> None:
        super().__init__()
        self.config = config

    def generate_heatmap(self):
        """Generates a heatmap showing error of individuals."""
        data = []
        for j in range(len(self.generations[0])):
            row = []
            for i in range(len(self.generations)):
                row.append(self.generations[i][j].fitness)
            data.append(row)
        data = np.array(data)

        plt.figure()
        ax = sns.heatmap(
            data,
            cmap='RdBu',
            xticklabels=2,
            yticklabels=2)

        hfont = {'fontname': 'Helvetica'}
        plt.xlabel('Generation', **hfont)
        plt.ylabel('Individual', **hfont)
        ax.invert_yaxis()
        ax.axhline(linewidth=4, color='black')
        ax.axvline(linewidth=4, color='black')
        ax.collections[0].colorbar.set_label('Fitness')
        plt.savefig('figures/heatmap.png')

    def graph_fitness_over_generation(self, with_scatter=False):
        """Graphs the change in error over generations."""
        mean_fitnesses = []
        best_individual_fitnesses = []

        for i in range(len(self.generations)):
            best_individual_fitnesses.append(
                self.get_high_fitness_individual(i).fitness)
            mean_fitnesses.append(
                np.mean([j.fitness for j in self.generations[i]]))

        plt.figure()
        if with_scatter:
            self.plot_error_scatter()
        mean_fitness_line, = plt.plot(
            range(len(self.generations)),
            mean_fitnesses,
            label='Mean Fitness')
        best_individual_fitness_line, = plt.plot(
            range(len(self.generations)),
            best_individual_fitnesses,
            label='Best Individual Fitness')
        plt.xticks(range(len(self.generations)))
        hfont = {'fontname': 'Helvetica'}
        plt.xlabel('Generation', **hfont)
        plt.ylabel('Individual', **hfont)
        plt.legend(handles=[mean_fitness_line, best_individual_fitness_line])
        plt.savefig('figures/fitness_over_generation.png')


def graph_vc_protocol(protocol: protocols.VoltageClampProtocol,
                      title: str) -> None:
    """Graphs a voltage clamp optimization individual."""
    plt.figure()
    i_trace = paci_2018.generate_trace(protocol=protocol)
    if i_trace:
        i_trace.plot_with_currents()
        plt.savefig('figures/{}.png'.format(title))
    else:
        print('Could not generate individual trace for individual: {}.'.format(
            protocol))


class Individual:
    """Represents an individual in a genetic algorithm population.

    Attributes:
        fitness: The fitness of the individual. This value can either be
            maximized or minimized.
    """

    def __init__(self, fitness):
        self.fitness = fitness


class ParameterTuningIndividual(Individual):
    """Represents an individual in a parameter tuning genetic algorithm.

    Attributes:
        parameters: An individuals parameters, ordered according to labels
            found in the config object the individual is associated with.
    """

    def __init__(self, parameters: List[float], fitness: float) -> None:
        super().__init__(fitness=fitness)
        self.parameters = parameters

    def __str__(self):
        return ', '.join([str(i) for i in self.parameters])

    def __repr__(self):
        return ', '.join([str(i) for i in self.parameters])

    def __eq__(self, other):
        if isinstance(other, self.__class__):
            return (self.fitness == other.fitness and
                    self.parameters == other.parameters)
        else:
            return False


class VCOptimizationIndividual(Individual):
    """Represents an individual in voltage clamp optimization genetic algorithm.

    Attributes:
        protocol: The protocol associated with an individual.
    """

    def __init__(self,
                 protocol: protocols.VoltageClampProtocol,
                 fitness: float=0.0) -> None:
        super().__init__(fitness=fitness)
        self.protocol = protocol

    def __str__(self):
        return str(self.fitness)

    def __repr__(self):
        return str(self.fitness)

    def __eq__(self, other):
        if isinstance(other, self.__class__):
            return (self.protocol == other.protocol and
                    self.fitness == other.fitness)
        else:
            return False

    def __lt__(self, other):
        return self.fitness < other.fitness

    def evaluate(self, config: ga_configs.VoltageOptimizationConfig) -> int:
        """Evaluates the fitness of the individual."""
        i_trace = paci_2018.PaciModel().generate_response(
            protocol=self.protocol)
        if not i_trace:
            return 0

        return _calc_fitness_score(
            contributions=get_contributions(i_trace=i_trace, config=config))


def graph_current_contributions(protocol: protocols.VoltageClampProtocol,
                                config: ga_configs.VoltageOptimizationConfig,
                                title: str) -> None:
    """Graphs the max current contributions at any time for all currents."""
    i_trace = paci_2018.generate_trace(protocol=protocol)
    if not i_trace:
        raise ValueError('Individual could not produce a valid trace.')

    max_contributions = _get_max_contributions(
        get_contributions(i_trace=i_trace, config=config))

    ax = max_contributions.plot.bar()
    ax.set_ylim(0, 1.0)
    plt.savefig('figures/{}.png'.format(title))


def get_contributions(
        i_trace: trace.Trace,
        config: ga_configs.VoltageOptimizationConfig) -> List[pd.DataFrame]:
    """Gets current contributions over windows of time."""
    contributions = []
    i = 0
    while i + config.contribution_step < len(i_trace.t):
        contributions.append(
            i_trace.current_response_info.calculate_current_contribution(
                timings=i_trace.t,
                start_t=i_trace.t[i],
                end_t=i_trace.t[i + config.contribution_step],
                target_currents=config.target_currents))
        i += config.contribution_step
    return contributions


def _get_max_contributions(contributions: List[pd.DataFrame]) -> pd.DataFrame:
    """Gets the max contributions for each current."""
    combined_current_contributions = pd.concat(contributions)
    max_contributions = combined_current_contributions.groupby(
        ['Parameter']).max()
    return max_contributions


def _calc_fitness_score(contributions: List[pd.DataFrame]) -> int:
    """Calculates the fitness score based on contribution time steps.

    Sums the max contributions for each parameter.
    """
    return _get_max_contributions(
        contributions=contributions)['Max Percent Contribution'].sum()
