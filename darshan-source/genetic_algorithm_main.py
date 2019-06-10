from absl import app

import configs
import genetic_algorithm


def main(unused_argv):
    parameters = [
        configs.Parameter(name='g_na', default_value=3671.2302),
        configs.Parameter(name='g_f_s_per_f', default_value=30.10312),
    ]
    # Parameters are sorted alphabetically to maintain order during each
    # generation of the genetic algorithm.
    parameters.sort(key=lambda x: x.name)

    config = configs.GeneticAlgorithmConfig(
        population_size=10,
        max_generations=20,
        target_objective=configs.TargetObjective.SINGLE_ACTION_POTENTIAL,
        tunable_parameters=parameters,
        params_lower_bound=0.8,
        params_upper_bound=1.2,
        crossover_probability=0.9,
        parameter_swap_probability=0.5,
        gene_mutation_probability=0.1)

    genetic_algorithm_instance = genetic_algorithm.GeneticAlgorithm(config)
    genetic_algorithm_instance.run()


if __name__ == '__main__':
    app.run(main)
