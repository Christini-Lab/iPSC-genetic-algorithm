"""Main driver for program.

Use functions in parameter_tuning_experiments.py to run GAs."""

import parameter_tuning_experiments
import protocols
import ga_configs
import voltage_clamp_optimization_experiments


PARAMETERS = [
    ga_configs.Parameter(name='g_na', default_value=3671.2302),
    ga_configs.Parameter(name='g_f_s', default_value=30.10312),
    ga_configs.Parameter(name='g_ks_s', default_value=2.041),
    ga_configs.Parameter(name='g_kr_s', default_value=29.8667),
    ga_configs.Parameter(name='g_k1_s', default_value=28.1492),
    ga_configs.Parameter(name='g_b_na', default_value=0.95),
    ga_configs.Parameter(name='g_na_lmax', default_value=17.25),
    ga_configs.Parameter(name='g_ca_l', default_value=8.635702e-5),
    ga_configs.Parameter(name='g_p_ca', default_value=0.4125),
    ga_configs.Parameter(name='g_b_ca', default_value=0.727272),
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

SAP_CONFIG = ga_configs.ParameterTuningConfig(
    population_size=10,
    max_generations=10,
    protocol=SAP_PROTOCOL,
    tunable_parameters=PARAMETERS,
    params_lower_bound=0.5,
    params_upper_bound=1.5,
    mate_probability=0.9,
    mutate_probability=1.0,
    gene_swap_probability=0.5,
    gene_mutation_probability=0.1,
    tournament_size=2)

IP_CONFIG = ga_configs.ParameterTuningConfig(
    population_size=2,
    max_generations=2,
    protocol=IP_PROTOCOL,
    tunable_parameters=PARAMETERS,
    params_lower_bound=0.5,
    params_upper_bound=1.5,
    mate_probability=0.9,
    mutate_probability=1.0,
    gene_swap_probability=0.5,
    gene_mutation_probability=0.1,
    tournament_size=2)

VCO_CONFIG = ga_configs.VoltageOptimizationConfig(
    contribution_step=100,
    steps_in_protocol=8,
    step_duration_bounds=(0.05, 0.6),
    step_voltage_bounds=(-.12, .06),
    target_currents=None,  # Setting None isolates all currents.
    population_size=2,
    max_generations=2,
    mate_probability=0.9,
    mutate_probability=0.9,
    gene_swap_probability=0.2,
    gene_mutation_probability=0.2,
    tournament_size=2)


def main():
    voltage_clamp_optimization_experiments.run_voltage_clamp_experiment(
        config=VCO_CONFIG,
        full_output=True)


if __name__ == '__main__':
    main()
