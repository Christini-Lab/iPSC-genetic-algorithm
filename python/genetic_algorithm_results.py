"""Contains classes to store the result of a genetic algorithm run.

Additionally, the classes in this module allow for figure generation.
"""

from abc import ABC
import enum
import math
import random
from typing import List

import numpy as np
from matplotlib import pyplot as plt
from matplotlib.colors import LogNorm
import seaborn as sns

import paci_2018
import protocols


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
                                   extreme_type: ExtremeType):
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


class GAResultParameterTuning(GeneticAlgorithmResult):

    def __init__(self, config):
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
        self.parameters = parameters
        super().__init__(fitness=fitness)

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
                 fitness: float) -> None:
        self.protocol = protocol
        super().__init__(fitness=fitness)
