"""Main driver for program. Use functions in experiments.py to run GAs."""

import experiments
import protocols
import ga_config


PARAMETERS = [
    ga_config.Parameter(name='g_na', default_value=3671.2302),
    ga_config.Parameter(name='g_f_s', default_value=30.10312),
    ga_config.Parameter(name='g_ks_s', default_value=2.041),
    ga_config.Parameter(name='g_kr_s', default_value=29.8667),
    ga_config.Parameter(name='g_k1_s', default_value=28.1492),
    ga_config.Parameter(name='g_b_na', default_value=0.95),
    ga_config.Parameter(name='g_na_lmax', default_value=17.25),
    ga_config.Parameter(name='g_ca_l', default_value=8.635702e-5),
    ga_config.Parameter(name='g_p_ca', default_value=0.4125),
    ga_config.Parameter(name='g_b_ca', default_value=0.727272),
]
# Parameters are sorted alphabetically to maintain order during each
# generation of the genetic algorithm.
PARAMETERS.sort(key=lambda x: x.name)

SAP_PROTOCOL = protocols.SingleActionPotentialProtocol()
IP_PROTOCOL = protocols.IrregularPacingProtocol(
        duration=3,
        stimulation_offsets=[0.1])
VC_PROTOCOL = protocols.VoltageClampProtocol(
    steps=[
        protocols.VoltageClampStep(duration=0.1, voltage=-0.08),
        protocols.VoltageClampStep(duration=0.1, voltage=-0.12),
        protocols.VoltageClampStep(duration=0.5, voltage=-0.06),
        protocols.VoltageClampStep(duration=0.05, voltage=-0.04),
        protocols.VoltageClampStep(duration=0.15, voltage=0.02),
        protocols.VoltageClampStep(duration=0.025, voltage=-0.08),
        protocols.VoltageClampStep(duration=0.3, voltage=0.04),
    ]
)

SAP_CONFIG = ga_config.GeneticAlgorithmConfig(
    population_size=10,
    max_generations=10,
    protocol=SAP_PROTOCOL,
    tunable_parameters=PARAMETERS,
    params_lower_bound=0.5,
    params_upper_bound=1.5,
    crossover_probability=0.9,
    parameter_swap_probability=0.5,
    gene_mutation_probability=0.1,
    tournament_size=2)

IP_CONFIG = ga_config.GeneticAlgorithmConfig(
        population_size=10,
        max_generations=10,
        protocol=IP_PROTOCOL,
        tunable_parameters=PARAMETERS,
        params_lower_bound=0.5,
        params_upper_bound=1.5,
        crossover_probability=0.9,
        parameter_swap_probability=0.5,
        gene_mutation_probability=0.1,
        tournament_size=2)


def main():
    sap_result = experiments.run_experiment(config=SAP_CONFIG)
    ip_result = experiments.run_experiment(config=IP_CONFIG)
    experiments.generate_error_over_generation_graph(
        results={'Single Action Potential': sap_result,
                 'Irregular Pacing': ip_result})


if __name__ == '__main__':
    main()
