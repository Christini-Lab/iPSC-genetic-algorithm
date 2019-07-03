"""Module level comment."""
from __future__ import annotations

import random
from typing import List

import numpy as np
import pandas as pd

import paci_2018
import protocols


class VCOGeneticAlgorithm:

    # Genetic algorithm hyperparameters.
    MUTATE_PROBABILITY = 0.5
    GENE_SWAP_PROBABILITY = 0.5
    MATE_PROBABILITY = 0.5
    CROSSOVER_PROBABILITY = 0.5
    CONTRIBUTION_STEP = 100
    STEP_COUNT = 6
    DURATION_BOUNDS = (0., 2.)
    VOLTAGE_BOUNDS = (-1.2, .6)

    def __init__(self, population_size=2, generations=2):
        self.population_size = population_size
        self.generations = generations

    def run(self):
        print('Running GA.')
        population = self._init_population()

        print('Evaluating initial population.')
        for individual in population:
            individual.fitness = self._evaluate(individual=individual)

        for generation in range(1, self.generations):
            print('Generation {}'.format(generation))

            for i_one, i_two in zip(population[::2], population[1::2]):
                if random.random() < self.MATE_PROBABILITY:
                    self._mate(i_one=i_one, i_two=i_two)

            for individual in population:
                if random.random() < self.MUTATE_PROBABILITY:
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
        while i + self.CONTRIBUTION_STEP < len(trace.t):
            contributions.append(
                trace.current_response_info.calculate_current_contribution(
                    timings=trace.t,
                    start_t=trace.t[i],
                    end_t=trace.t[i + self.CONTRIBUTION_STEP]))
            i += self.CONTRIBUTION_STEP
        return _calc_fitness_score(contributions=contributions)

    def _mate(self,
              i_one: Individual,
              i_two: Individual) -> None:
        """Mates two individuals, modifies them in-place."""
        if len(i_one.protocol.steps) != len(i_two.protocol.steps):
            raise ValueError('Individuals do not have the same num of steps.')

        for i in range(len(i_one.protocol.steps)):
            if random.random() < self.CROSSOVER_PROBABILITY:
                i_one.protocol.steps[i], i_two.protocol.steps[i] = (
                    i_two.protocol.steps[i], i_one.protocol.steps[i])

    def _mutate(self,
                individual: Individual) -> None:
        """Mutates an individual by choosing a number for norm. distribution."""
        for i in range(len(individual.protocol.steps)):
            if random.random() < self.GENE_SWAP_PROBABILITY:
                individual.protocol.steps[i].voltage = np.random.normal(
                    individual.protocol.steps[i].voltage)
                individual.protocol.steps[i].duration = np.random.normal(
                    individual.protocol.steps[i].duration)

    def _init_individual(self):
        """Initializes a individual with a randomized protocol."""
        steps = []
        for i in range(self.STEP_COUNT):
            random_step = protocols.VoltageClampStep(
                voltage=random.uniform(*self.VOLTAGE_BOUNDS),
                duration=random.uniform(*self.DURATION_BOUNDS))
            steps.append(random_step)
        return Individual(
            protocol=protocols.VoltageClampProtocol(steps=steps),
            fitness=0)

    def _init_population(self):
        return [self._init_individual() for _ in range(self.population_size)]


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
