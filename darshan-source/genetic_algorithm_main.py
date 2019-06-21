from absl import app
from matplotlib import pyplot as plt
from numpy.polynomial import Polynomial

import numpy as np

import ga_config
import genetic_algorithm
import irregular_pacing
import paci_2018
import single_action_potential


def main(unused_argv):
    parameters = [
        ga_config.Parameter(name='g_na', default_value=3671.2302),
        ga_config.Parameter(name='g_f_s_per_f', default_value=30.10312),
        ga_config.Parameter(name='g_ks_s_per_f', default_value=2.041),
        ga_config.Parameter(name='g_kr_s_per_f', default_value=29.8667),
        ga_config.Parameter(name='g_k1_s_per_f', default_value=28.1492),
        ga_config.Parameter(name='g_b_na', default_value=0.95),
        ga_config.Parameter(name='g_na_lmax', default_value=17.25),
    ]
    # Parameters are sorted alphabetically to maintain order during each
    # generation of the genetic algorithm.
    parameters.sort(key=lambda x: x.name)

    config = ga_config.GeneticAlgorithmConfig(
        population_size=2,
        max_generations=2,
        protocol=single_action_potential.SingleActionPotentialProtocol(),
        tunable_parameters=parameters,
        params_lower_bound=0.5,
        params_upper_bound=1.5,
        crossover_probability=0.9,
        parameter_swap_probability=0.5,
        gene_mutation_probability=0.1,
        tournament_size=2)

    ga_result = genetic_algorithm.GeneticAlgorithm(config=config).run()


if __name__ == '__main__':
    app.run(main)
