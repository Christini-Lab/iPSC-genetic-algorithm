"""Module level comment."""
from __future__ import annotations

import random
from typing import List

import numpy as np
import pandas as pd

import ga_configs
import paci_2018
import protocols


class VCOGeneticAlgorithm:

    def __init__(self, config: ga_configs.VoltageOptimizationConfig):
        self.config = config

    def run(self):
        print('Running GA.')
        population = self._init_population()

        print('Evaluating initial population.')
        for individual in population:
            individual.fitness = self._evaluate(individual=individual)

        for generation in range(1, self.config.max_generations):
            print('Generation {}'.format(generation))

            # TODO call selection method here.

            for i_one, i_two in zip(population[::2], population[1::2]):
                if random.random() < self.config.mate_probability:
                    self._mate(i_one=i_one, i_two=i_two)

            for individual in population:
                if random.random() < self.config.mutate_probability:
                    self._mutate(individual=individual)

            # Update fitness of all individuals in population.
            for individual in population:
                individual.fitness = self._evaluate(individual=individual)

            generate_statistics(population)

    def _evaluate(self,
                  individual: Individual) -> int:
        """Evaluates the fitness of an individual.

        Fitness is determined by how well the voltage clamp protocol isolates
        individual ionic currents.
        """
        trace = paci_2018.PaciModel().generate_response(
            protocol=individual.protocol)
        if not trace:
            return 0

        contributions = []
        i = 0
        while i + self.config.contribution_step < len(trace.t):
            contributions.append(
                trace.current_response_info.calculate_current_contribution(
                    timings=trace.t,
                    start_t=trace.t[i],
                    end_t=trace.t[i + self.config.contribution_step]))
            i += self.config.contribution_step
        return _calc_fitness_score(contributions=contributions)

    def _mate(self,
              i_one: Individual,
              i_two: Individual) -> None:
        """Mates two individuals, modifies them in-place."""
        if len(i_one.protocol.steps) != len(i_two.protocol.steps):
            raise ValueError('Individuals do not have the same num of steps.')

        for i in range(len(i_one.protocol.steps)):
            if random.random() < self.config.gene_swap_probability:
                i_one.protocol.steps[i], i_two.protocol.steps[i] = (
                    i_two.protocol.steps[i], i_one.protocol.steps[i])

    def _mutate(self,
                individual: Individual) -> None:
        """Mutates an individual by choosing a number for norm. distribution."""
        for i in range(len(individual.protocol.steps)):
            if random.random() < self.config.gene_mutation_probability:
                individual.protocol.steps[i].voltage = np.random.normal(
                    individual.protocol.steps[i].voltage)
                individual.protocol.steps[i].duration = np.random.normal(
                    individual.protocol.steps[i].duration)

    def _select(self, population: List[Individual]) -> None:
        """Selects a list of individuals using tournament selection."""
        pass

    def _init_individual(self):
        """Initializes a individual with a randomized protocol."""
        steps = []
        for i in range(self.config.steps_in_protocol):
            random_step = protocols.VoltageClampStep(
                voltage=random.uniform(*self.config.step_voltage_bounds),
                duration=random.uniform(*self.config.step_duration_bounds))
            steps.append(random_step)
        return Individual(
            protocol=protocols.VoltageClampProtocol(steps=steps),
            fitness=0)

    def _init_population(self):
        return [
            self._init_individual() for _ in range(self.config.population_size)
        ]


def _calc_fitness_score(contributions: List[pd.DataFrame]) -> int:
    """Calculates the fitness score based on contribution time steps.

    Sums the max contributions for each parameter.
    """
    combined_current_contributions = pd.concat(contributions)
    max_contributions = combined_current_contributions.groupby(
        ['Parameter']).max()
    return max_contributions['Percent Contribution'].sum()


def generate_statistics(population):
    fitness_values = [i.fitness for i in population]
    print('  Min fitness: {}'.format(min(fitness_values)))
    print('  Max fitness: {}'.format(max(fitness_values)))
    print('  Average fitness: {}'.format(np.mean(fitness_values)))
    print('  Standard deviation: {}'.format(np.std(fitness_values)))


class Individual:

    def __init__(self, protocol, fitness=0):
        self.protocol = protocol
        self.fitness = fitness
