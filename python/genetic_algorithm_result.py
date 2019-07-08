"""Contains classes to store the result of a genetic algorithm run.

Additionally, the classes in this module allow for figure generation.
"""

import enum
import math
import random


import numpy as np
from matplotlib import pyplot as plt
from matplotlib.colors import LogNorm
import seaborn as sns

import paci_2018
import protocols


class ExtremeType(enum.Enum):
    BEST = 1
    WORST = 2


class GeneticAlgorithmResult:
    """Contains information about a run of a genetic algorithm.

    Attributes:
        config: The config object used in the genetic algorithm run.
        baseline_trace: The baseline trace of the genetic algorithm run.
        generations: A 2D list of every individual in the genetic algorithm.
    """

    def __init__(self, config, baseline_trace):
        self.config = config
        self.baseline_trace = baseline_trace
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

    def get_best_individual(self, generation):
        """Given a generation, returns the individual with the least error."""
        return self._get_individual_at_extreme(generation, ExtremeType.BEST)

    def get_worst_individual(self, generation):
        """Given a generation, returns the individual with the most error."""
        return self._get_individual_at_extreme(generation, ExtremeType.WORST)

    def _get_individual_at_extreme(self,
                                   generation: int,
                                   extreme_type: ExtremeType):
        """Retrieves either the best or worst individual given a generation."""
        top_error_individual = self.get_individual(generation, 0)
        for i in range(len(self.generations[generation])):
            individual = self.get_individual(generation, i)
            if (extreme_type == ExtremeType.BEST and
                    individual.error < top_error_individual.error):
                top_error_individual = individual
            elif (extreme_type == ExtremeType.WORST and
                    individual.error > top_error_individual.error):
                top_error_individual = individual
        return top_error_individual

    def get_parameter_scales(self, individual):
        parameter_scaling = []
        for i in range(len(self.config.tunable_parameters)):
            parameter_scaling.append(
                individual.param_set[i] /
                self.config.tunable_parameters[i].default_value)
        return parameter_scaling

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
        plt.savefig('figures/heatmap.png')

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
        parameter_indices = [i for i in range(len(individual.param_set))]

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

    def graph_individual(self, individual):
        """Graphs an individual's trace."""
        plt.figure()
        if isinstance(self.config.protocol, protocols.VoltageClampProtocol):
            self.baseline_trace.plot_only_currents(color='black')
        else:
            plt.plot(
                self.baseline_trace.t,
                self.baseline_trace.y,
                color='black')
        trace = paci_2018.generate_trace(
            tunable_parameters=self.config.tunable_parameters,
            protocol=self.config.tunable_parameters,
            params=individual.param_set)
        if trace:
            if isinstance(self.config.protocol, protocols.VoltageClampProtocol):
                trace.plot_only_currents(color='b--')
            else:
                plt.plot(trace.t, trace.y, 'b--')

        return trace

    def plot_error_scatter(self):
        x_data = []
        y_data = []
        for i in range(self.config.max_generations):
            for j in range(self.config.population_size):
                x_data.append(j)
                y_data.append(self.get_individual(generation=i, index=j).error)
        plt.scatter(x_data, y_data, alpha=0.3, color='red')

    def graph_error_over_generation(self, with_scatter=False):
        """Graphs the change in error over generations."""
        mean_errors = []
        best_individual_errors = []

        for i in range(len(self.generations)):
            best_individual_errors.append(self.get_best_individual(i).error)
            mean_errors.append(np.mean([j.error for j in self.generations[i]]))

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

    def __eq__(self, other):
        if isinstance(other, self.__class__):
            return self.__dict__ == other.__dict__
