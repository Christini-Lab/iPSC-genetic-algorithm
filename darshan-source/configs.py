"""Contains config classes used to set genetic algorithm hyperparameters."""

from enum import Enum


class TargetObjective(Enum):
    """Represents the target objective of the genetic algorithm."""

    # A single action potential maps the change in membrane potential over the
    # course of a single heart beat.
    SINGLE_ACTION_POTENTIAL = 1

    # Stochastic pacing refers to the trace derived from several random
    # stimulations over a period of time.
    STOCHASTIC_PACING = 2

    # Refers to the current trace resulting from a voltage clamp experiment.
    VOLTAGE_CLAMP = 3


class Parameter:
    """Represents a single, tunable parameter in a cardiac myocyte model.

    Attributes:
        name: A string representing name of the parameter.
        default_value: The default value of the parameter.
    """

    def __init__(self, name, default_value):
        self.name = name
        self.default_value = default_value


class GeneticAlgorithmConfig:
    """Encapsulates hyperparameters for running a genetic algorithm.

    Contains relevant hyperparameters for parameter estimation on a cardiac
    model.

    Attributes:
        population_size: The size of the population in each generation.
        max_generations: The max number of generations to run the algorithm for.
        target_objective: A TargetObjective enum object representing the
                target objective of the genetic algorithm.
        tunable_parameters: A list of strings, representing the names of which
            parameters will be tuned.
        params_lower_bound: A float representing the lower bound a randomized
            parameter value can be for any individual in the population. For
            example, if the default parameter value is 100, and
            params_lower_bound is 0.1, than 10 is smallest value that parameter
            can be during the genetic algorithm.
        params_upper_bound: A float representing the upper bound a randomized
            parameter value can be for any individual in the population. See
            the description of `params_lower_bound` for more info.
        crossover_probability: The probability two individuals will `mate`.
        parameter_swap_probability: The probability a parameter, or `gene`, will
            be swapped between a pair of `mated` individuals.
        gene_mutation_probability: Probability a certain gene will be mutated:
            replaced with a random number from a normal distribution centered
            around the value of the gene.
    """

    def __init__(self, population_size, max_generations, target_objective,
                 tunable_parameters, params_lower_bound,
                 params_upper_bound, crossover_probability,
                 parameter_swap_probability, gene_mutation_probability):
        self.population_size = population_size
        self.max_generations = max_generations
        self.target_objective = target_objective
        self.tunable_parameters = tunable_parameters
        self.params_lower_bound = params_lower_bound
        self.params_upper_bound = params_upper_bound
        self.crossover_probability = crossover_probability
        self.parameter_swap_probability = parameter_swap_probability
        self.gene_mutation_probability = gene_mutation_probability
