"""Contains classes used to configure the genetic algorithm."""


class Parameter:
    """Represents a parameter in the model.

    Attributes:
        name: Name of parameter.
        default_value: Default value of parameter.
    """

    def __init__(self, name, default_value):
        self.name = name
        self.default_value = default_value


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
    """

    MAX_ERROR = 4

    def __init__(self, population_size, max_generations, protocol,
                 tunable_parameters, params_lower_bound,
                 params_upper_bound, crossover_probability,
                 parameter_swap_probability, gene_mutation_probability,
                 tournament_size):
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
