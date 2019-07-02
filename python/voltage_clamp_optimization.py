"""Module level comment."""

import random
from typing import List

import numpy as np
import pandas as pd

import paci_2018
import protocols


class VCOGeneticAlgorithm:

    # Genetic algorithm hyperparameters.
    MUTATION_PROBABILITY = 0.5
    MATE_PROBABILITY = 0.5
    CONTRIBUTION_STEP = 100

    def __init__(self):
        pass

    def run(self):
        print('Running GA.')

    def _evaluate(self, individual):
        trace = paci_2018.PaciModel().generate_response(protocol=individual)
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
              individual_one: protocols.VoltageClampProtocol,
              individual_two: protocols.VoltageClampProtocol) -> None:
        """Mates two individuals, modifies them in-place."""
        if len(individual_one.steps) != len(individual_two.steps):
            raise ValueError('Individuals do not have the same num of steps.')

        for i in range(len(individual_one.steps)):
            if random.random() > self.MATE_PROBABILITY:
                individual_one.steps[i], individual_two.steps[i] = (
                    individual_two.steps[i], individual_one.steps[i])

    def _mutate(self,
                individual: protocols.VoltageClampProtocol) -> None:
        """Mutates an individual by choosing a number for norm. distribution."""
        for i in range(len(individual.steps)):
            if random.random() > self.MUTATION_PROBABILITY:
                individual.steps[i].voltage = np.random.normal(
                    individual.steps[i].voltage)
                individual.steps[i].duration = np.random.normal(
                    individual.steps[i].duration)

    def _init_params(self):
        pass


def _calc_fitness_score(contributions: List[pd.DataFrame]) -> int:
    combined_current_contributions = pd.concat(contributions)
    max_contributions = combined_current_contributions.groupby(
        ['Parameter']).max()
    return max_contributions['Percent Contribution'].sum()
