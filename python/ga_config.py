"""Contains classes used to configure the genetic algorithm."""

from __future__ import annotations

from typing import List, Union
import protocols

PROTOCOL_TYPE = Union[
    protocols.SingleActionPotentialProtocol,
    protocols.IrregularPacingProtocol,
    protocols.VoltageClampProtocol
]


class Parameter:
    """Represents a parameter in the model.

    Attributes:
        name: Name of parameter.
        default_value: Default value of parameter.
    """

    def __init__(self, name: str, default_value: float) -> None:
        self.name = name
        # Do not change default value once set during init.
        self.default_value = default_value

    def __str__(self) -> str:
        return self.name

    def __repr__(self) -> str:
        return self.name

    def __eq__(self, other):
        if isinstance(other, self.__class__):
            return self.__dict__ == other.__dict__


class GeneticAlgorithmConfig:
    """Contains hyperparameters for configuring a genetic algorithm.

    Attributes:
        population_size: Size of the population in each generation.
        max_generations: Max number of generations to run the algorithm for.
        protocol: Object representing the specific target objective of the
            genetic algorithm.
        tunable_parameters: List of strings representing the names of parameters
            that will be tuned.
        params_lower_bound: A float representing the lower bound a randomized
            parameter value can be for any individual in the population. For
            example, if the default parameter value is 100, and
            params_lower_bound is 0.1, than 10 is smallest value that parameter
            can be set to.
        params_upper_bound: A float representing the upper bound a randomized
            parameter value can be for any individual in the population. See
            the description of `params_lower_bound` for more info.
        crossover_probability: The probability two individuals will `mate`.
        parameter_swap_probability: The probability a parameter, or `gene`, will
            be swapped between a pair of `mated` individuals.
        gene_mutation_probability: Probability a certain gene will be mutated:
            replaced with a random number from a normal distribution centered
            around the value of the gene.
        tournament_size: Number of individuals chosen during each round of
            tournament selection.
        secondary_protocol: A secondary protocol used for a combined protocol.
    """

    # If a model with an individual's parameter set fails to generate a trace,
    # the individual will have it's fitness set to one of the following,
    # according to the protocol.
    SAP_MAX_ERROR = 100
    IP_MAX_ERROR = 130
    VC_MAX_ERROR = 130

    def __init__(self,
                 population_size: int,
                 max_generations: int,
                 protocol: PROTOCOL_TYPE,
                 tunable_parameters: List[Parameter],
                 params_lower_bound: float,
                 params_upper_bound: float,
                 crossover_probability: float,
                 parameter_swap_probability: float,
                 gene_mutation_probability: float,
                 tournament_size: int,
                 secondary_protocol: PROTOCOL_TYPE=None) -> None:
        self.population_size = population_size
        self.max_generations = max_generations
        self.protocol = protocol
        self.tunable_parameters = tunable_parameters
        self.params_lower_bound = params_lower_bound
        self.params_upper_bound = params_upper_bound
        self.crossover_probability = crossover_probability
        self.parameter_swap_probability = parameter_swap_probability
        self.gene_mutation_probability = gene_mutation_probability
        self.tournament_size = tournament_size
        self.secondary_protocol = secondary_protocol

    def has_equal_hyperparameters(self,
                                  other_config: GeneticAlgorithmConfig) -> bool:
        """Checks if another config object has the same hyperparameters.

        This is used when running comparisons between SAP and IP genetic
        algorithms. Both configs should have the same hyperparameters, but will
        differ in their protocol.
        """
        return (self.population_size == other_config.population_size and
                self.max_generations == other_config.max_generations and
                self.tunable_parameters == other_config.tunable_parameters and
                self.params_lower_bound == other_config.params_lower_bound and
                self.params_upper_bound == other_config.params_upper_bound and
                self.crossover_probability ==
                other_config.crossover_probability and
                self.parameter_swap_probability ==
                other_config.parameter_swap_probability and
                self.gene_mutation_probability ==
                other_config.gene_mutation_probability and
                self.tournament_size == other_config.tournament_size)


def get_appropriate_max_error(protocol):
    if isinstance(protocol, protocols.SingleActionPotentialProtocol):
        return GeneticAlgorithmConfig.SAP_MAX_ERROR
    elif isinstance(protocol, protocols.IrregularPacingProtocol):
        return GeneticAlgorithmConfig.IP_MAX_ERROR
    elif isinstance(protocol, protocols.VoltageClampProtocol):
        return GeneticAlgorithmConfig.VC_MAX_ERROR
