from absl import app

import configs
import genetic_algorithm


def main(unused_argv):
    parameters = [
        configs.Parameter(name='g_na', default_value=3671.2302),
        configs.Parameter(name='g_f_s_per_f', default_value=30.10312),
        configs.Parameter(name='g_ks_s_per_f', default_value=2.041),
        configs.Parameter(name='g_kr_s_per_f', default_value=29.8667),
        configs.Parameter(name='g_k1_s_per_f', default_value=28.1492),
        configs.Parameter(name='g_b_na', default_value=0.95),
        configs.Parameter(name='g_na_lmax', default_value=17.25),
    ]
    # Parameters are sorted alphabetically to maintain order during each
    # generation of the genetic algorithm.
    parameters.sort(key=lambda x: x.name)

    config = configs.GeneticAlgorithmConfig(
        population_size=5,
        max_generations=5,
        target_objective=configs.TargetObjective.SINGLE_ACTION_POTENTIAL,
        tunable_parameters=parameters,
        params_lower_bound=0.5,
        params_upper_bound=1.5,
        crossover_probability=0.9,
        parameter_swap_probability=0.5,
        gene_mutation_probability=0.1)

    genetic_algorithm_instance = genetic_algorithm.GeneticAlgorithm(config)
    ga_result = genetic_algorithm_instance.run()
    ga_result.generate_heatmap()

    best_ind = ga_result.get_best_individual(generation=0)
    ga_result.graph_individual(best_ind)

    best_ind = ga_result.get_best_individual(generation=config.max_generations)
    ga_result.graph_individual(best_ind)


if __name__ == '__main__':
    app.run(main)
