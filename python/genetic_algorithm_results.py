"""Contains classes to store the result of a genetic algorithm run.

Additionally, the classes in this module allow for figure generation.
"""

from abc import ABC
import copy
import enum
import math
import random
from typing import Dict, List, Union

from matplotlib import pyplot as plt
from matplotlib.colors import LogNorm
import numpy as np
import pandas as pd
import seaborn as sns

import ga_configs
import paci_2018
import kernik
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

        plt.figure(figsize=(10, 5))
        ax = sns.heatmap(
            data,
            cmap='viridis',
            xticklabels=2,
            yticklabels=2,
            norm=log_norm,
            cbar_kws={'ticks': cbar_ticks, 'aspect': 15})

        hfont = {'fontname': 'Helvetica'}
        plt.xlabel('Generation', **hfont)
        plt.ylabel('Individual', **hfont)
        plt.xticks(
            [i for i in range(0, self.config.max_generations, 5)],
            [i for i in range(0, self.config.max_generations, 5)])
        plt.yticks(
            [i for i in range(0, self.config.population_size, 5)],
            [i for i in range(0, self.config.population_size, 5)])

        ax.invert_yaxis()
        ax.collections[0].colorbar.set_label('Error')
        ax.spines['right'].set_visible(False)
        ax.spines['top'].set_visible(False)
        plt.savefig('figures/Parameter Tuning Figure/heatmap.svg')

    def plot_error_scatter(self):
        plt.figure(figsize=(10, 5))
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
        #self.baseline_trace = paci_2018.generate_trace(
        #    tunable_parameters=config.tunable_parameters,
        #    protocol=config.protocol)
        self.baseline_trace = kernik.generate_trace(
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
                [i * 1000 for i in self.baseline_trace.t],
                [i * 1000 for i in self.baseline_trace.y],
                color='black')

        trace = kernik.generate_trace(
            tunable_parameters=self.config.tunable_parameters,
            protocol=self.config.protocol,
            params=individual.parameters)
        #trace = paci_2018.generate_trace(
        #    tunable_parameters=self.config.tunable_parameters,
        #    protocol=self.config.protocol,
        #    params=individual.parameters)
        if trace:
            if isinstance(self.config.protocol, protocols.VoltageClampProtocol):
                trace.plot_only_currents(color='b--')
            else:
                plt.plot(
                    [i * 1000 for i in trace.t],
                    [i * 1000 for i in trace.y],
                    'b--')

        plt.xlabel('Time (ms)')
        plt.ylabel(r'$V_m$ (mV)')
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
        fig = plt.figure(figsize=(10, 5))
        fig.suptitle(title)

        ax_1 = plt.subplot(1, 2, 1)
        plt.plot(
            [i * 1000 for i in self.baseline_trace.t],
            [i * 1000 for i in self.baseline_trace.y],
            color='black')
        i_trace = paci_2018.generate_trace(
            tunable_parameters=self.config.tunable_parameters,
            protocol=self.config.protocol,
            params=individual.parameters)
        plt.plot(
            [i * 1000 for i in i_trace.t],
            [i * 1000 for i in i_trace.y],
            'b--')
        ax_1.spines['right'].set_visible(False)
        ax_1.spines['top'].set_visible(False)
        plt.xlabel('Time (ms)')
        plt.ylabel(r'$V_m$ (mV)')

        ax_2 = plt.subplot(1, 2, 2)
        parameter_scaling = self.get_parameter_scales(individual=individual)
        parameter_indices = [i for i in range(len(individual.parameters))]

        x = parameter_indices
        y = np.array(parameter_scaling)
        color = np.where(y >= 1, 'green', 'red')
        plt.vlines(x=x, ymin=1, ymax=y, color=color, alpha=0.75, linewidth=5)
        plt.scatter(x, y, color=color, s=20, alpha=1)
        plt.axhline(1, linewidth=0.5, linestyle='--', color='gray')
        plt.xlabel('Parameters')
        plt.ylabel('Scaling')
        plt.xticks(
            parameter_indices,
            ['$G_{{{}}}$'.format(i.name[2:])
             for i in self.config.tunable_parameters])
        plt.yticks([i for i in range(0, 4)], [i for i in range(0, 4)])
        ax_2.spines['right'].set_visible(False)
        ax_2.spines['top'].set_visible(False)

        fig.subplots_adjust(wspace=.35)
        plt.savefig('figures/Parameter Tuning Figure/{}.svg'.format(title))

    def graph_error_over_generation(self, with_scatter=False):
        """Graphs the change in error over generations."""
        mean_errors = []
        best_individual_errors = []

        for i in range(len(self.generations)):
            best_individual_errors.append(
                self.get_low_fitness_individual(i).fitness)
            mean_errors.append(
                np.mean([j.fitness for j in self.generations[i]]))

        plt.figure(figsize=(10, 5))
        ax = plt.subplot()
        if with_scatter:
            self.plot_error_scatter()
        mean_error_line, = plt.plot(
            range(len(self.generations)),
            mean_errors,
            label='Mean Error of Individuals',
            color='b')
        best_individual_error_line, = plt.plot(
            range(len(self.generations)),
            best_individual_errors,
            label='Lowest Error of an Individual',
            color='green')
        plt.xticks(
            [i for i in range(0, self.config.max_generations, 5)],
            [i for i in range(0, self.config.max_generations, 5)])
        hfont = {'fontname': 'Helvetica'}
        plt.xlabel('Generation', **hfont)
        plt.ylabel('Error', **hfont)
        ax.spines['right'].set_visible(False)
        ax.spines['top'].set_visible(False)

        plt.legend(
            handles=[mean_error_line, best_individual_error_line],
            loc='upper right',
            bbox_to_anchor=(1, 1.1))
        plt.savefig('figures/Parameter Tuning Figure/error_over_generation.svg')


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
        plt.savefig('figures/Voltage Clamp Figure/Single VC Optimization/'
                    'heatmap.svg')

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
        plt.savefig('figures/Voltage Clamp Figure/Single VC Optimization/'
                    'fitness_over_generation.svg')


def graph_vc_protocol(protocol: protocols.VoltageClampProtocol,
                      title: str) -> None:
    """Graphs a voltage clamp optimization individual."""
    plt.figure()
    i_trace = paci_2018.generate_trace(protocol=protocol)
    if i_trace:
        i_trace.plot_with_currents()
        plt.savefig('figures/Voltage Clamp Figure/Single VC Optimization/'
                    '{}.svg'.format(title))
    else:
        print('Could not generate individual trace for individual: {}.'.format(
            protocol))


def graph_optimized_vc_protocol_full_figure(
        single_current_protocols: Dict[str, protocols.VoltageClampProtocol],
        combined_protocol: protocols.VoltageClampProtocol,
        config: ga_configs.VoltageOptimizationConfig) -> None:
    """Graphs a full figure for a optimized voltage protocol."""
    plt.figure(figsize=(20, 10))
    #i_trace = paci_2018.generate_trace(protocol=combined_protocol)
    i_trace = kernik.generate_trace(protocol=combined_protocol)
    i_trace.plot_with_currents(title='')
    plt.savefig('figures/Voltage Clamp Figure/Full VC Optimization/Combined '
                'trace.svg')

    # Plot single current traces.
    i = 1
    for key in sorted(single_current_protocols.keys()):
        plt.figure(figsize=(10, 5))
        #i_trace = paci_2018.generate_trace(
        #    protocol=single_current_protocols[key])
        i_trace = kernik.generate_trace(
            protocol=single_current_protocols[key])
        i_trace.plot_with_currents(title=r'$I_{{{}}}$'.format(key[2:]))
        i += 1
        plt.savefig(
            'figures/Voltage Clamp Figure/Full VC Optimization/'
            '{} single current trace.svg'.format(key))

    # Plot current contributions for combined trace.
    graph_combined_current_contributions(
        protocol=combined_protocol,
        config=config,
        title='Full VC Optimization/Combined current contributions'
    )

    # Plot single current max contributions.
    graph_single_current_contributions(
        single_current_protocols=single_current_protocols,
        config=config,
        title='Full VC Optimization/Single current contributions')


def graph_single_current_contributions(
        single_current_protocols: Dict[str, protocols.VoltageClampProtocol],
        config: ga_configs.VoltageOptimizationConfig,
        title: str) -> None:
    """Graphs the max current contributions for single currents together."""
    single_current_max_contributions = {}
    for key, value in single_current_protocols.items():
        #i_trace = paci_2018.generate_trace(protocol=value)
        i_trace = kernik.generate_trace(protocol=value)

        max_contributions = i_trace.current_response_info.\
            get_max_current_contributions(
                time=i_trace.t,
                window=config.window,
                step_size=config.step_size)
        single_current_max_contributions[key] = max_contributions[
            max_contributions['Current'] == key]['Contribution'].values[0]

    graph_current_contributions_helper(
        currents=single_current_max_contributions.keys(),
        contributions=single_current_max_contributions.values(),
        target_currents=config.target_currents,
        title=title)


def graph_combined_current_contributions(
        protocol: protocols.VoltageClampProtocol,
        config: ga_configs.VoltageOptimizationConfig,
        title: str) -> None:
    """Graphs the max current contributions for a single protocol."""
    #i_trace = paci_2018.generate_trace(protocol=protocol)
    i_trace = kernik.generate_trace(protocol=protocol)
    max_contributions = i_trace.current_response_info.\
        get_max_current_contributions(
            time=i_trace.t,
            window=config.window,
            step_size=config.step_size)

    graph_current_contributions_helper(
        currents=list(max_contributions['Current']),
        contributions=list(max_contributions['Contribution']),
        target_currents=config.target_currents,
        title=title)


def graph_current_contributions_helper(currents,
                                       contributions,
                                       target_currents,
                                       title):
    plt.figure()
    sns.set(style="white")

    # Sort currents according to alphabetic order.
    zipped_list = sorted(zip(currents, contributions))
    contributions = [
        contrib for curr, contrib in zipped_list if curr in target_currents
    ]
    currents = [curr for curr, _ in zipped_list if curr in target_currents]

    currents = ['$I_{{{}}}$'.format(i[2:]) for i in currents]

    ax = sns.barplot(
        x=currents,
        y=[i * 100 for i in contributions],
        color='gray',
        linewidth=0.75)
    ax.set_ylabel('Percent Contribution')
    ax.set_yticks([i for i in range(0, 120, 20)])
    ax.set_ybound(lower=0, upper=100)
    ax.set_xticklabels(ax.get_xticklabels(), rotation=-30)
    ax.spines['right'].set_visible(False)
    ax.spines['top'].set_visible(False)
    plt.savefig('figures/Voltage Clamp Figure/{}.svg'.format(title))

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
        #i_trace = paci_2018.PaciModel().generate_response(
        #    protocol=self.protocol)
        i_trace = kernik.KernikModel().generate_response(
            protocol=self.protocol)

        if not i_trace:
            return 0

        max_contributions = i_trace.current_response_info.\
            get_max_current_contributions(
                time=i_trace.t,
                window=config.window,
                step_size=config.step_size)
        return max_contributions['Contribution'].sum()

