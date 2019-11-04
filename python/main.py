"""Main driver for program. Before running, make sure to have a directory
called `figures` for matplotlib pictures to be stored in."""

import copy
import time

import parameter_tuning_experiments
import protocols
import ga_configs
import genetic_algorithm_results
import voltage_clamp_optimization_experiments

PARAMETERS = [
    ga_configs.Parameter(name='G_Na', default_value=1),
    ga_configs.Parameter(name='G_F', default_value=1),
    ga_configs.Parameter(name='G_Ks', default_value=1),
    ga_configs.Parameter(name='G_Kr', default_value=1),
    ga_configs.Parameter(name='G_K1', default_value=1),
    ga_configs.Parameter(name='G_bNa', default_value=1),
    ga_configs.Parameter(name='G_NaL', default_value=1),
    ga_configs.Parameter(name='G_CaL', default_value=1),
    ga_configs.Parameter(name='G_pCa', default_value=1),
    ga_configs.Parameter(name='G_bCa', default_value=1),
    ga_configs.Parameter(name='K_NaCa', default_value=1)
]
# Parameters are sorted alphabetically to maintain order during each
# generation of the genetic algorithm.
PARAMETERS.sort(key=lambda x: x.name)

SAP_PROTOCOL_KERNIK = protocols.SingleActionPotentialProtocol(1800)

VC_PROTOCOL = protocols.VoltageClampProtocol(
    steps=[
        protocols.VoltageClampStep(duration=0.050, voltage=-0.08),
        protocols.VoltageClampStep(duration=0.050, voltage=-0.12),
        protocols.VoltageClampStep(duration=0.500, voltage=-0.057),
        protocols.VoltageClampStep(duration=0.025, voltage=-0.04),
        protocols.VoltageClampStep(duration=0.075, voltage=0.02),
        protocols.VoltageClampStep(duration=0.025, voltage=-0.08),
        protocols.VoltageClampStep(duration=0.250, voltage=0.04),
        protocols.VoltageClampStep(duration=1.900, voltage=-0.03),
        protocols.VoltageClampStep(duration=0.750, voltage=0.04),
        protocols.VoltageClampStep(duration=1.725, voltage=-0.03),
        protocols.VoltageClampStep(duration=0.650, voltage=-0.08),
    ]
)

SAP_CONFIG = ga_configs.ParameterTuningConfig(
    population_size=10,
    max_generations=4,
    protocol=SAP_PROTOCOL_KERNIK,
    tunable_parameters=PARAMETERS,
    params_lower_bound=0.1,
    params_upper_bound=3,
    mate_probability=0.9,
    mutate_probability=0.9,
    gene_swap_probability=0.2,
    gene_mutation_probability=0.2,
    tournament_size=2)

VC_CONFIG = ga_configs.ParameterTuningConfig(
    population_size=4,
    max_generations=10,
    protocol=VC_PROTOCOL,
    tunable_parameters=PARAMETERS,
    params_lower_bound=0.1,
    params_upper_bound=3,
    mate_probability=0.9,
    mutate_probability=0.9,
    gene_swap_probability=0.2,
    gene_mutation_probability=0.2,
    tournament_size=4)

def main():
    """Run parameter tuning or voltage clamp protocol experiments here
    """
    parameter_tuning_experiments.run_param_tuning_experiment(
        config=SAP_CONFIG,
        with_output=True)


if __name__ == '__main__':
    main()
