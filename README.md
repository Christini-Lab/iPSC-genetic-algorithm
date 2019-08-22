# iPSC-genetic-algorithm

This repository contains python code to do two things: parameter estimation via a genetic algorithm and voltage clamp protocol construction via a genetic algorithm.  

## Parameter Estimation

### Purpose
A cardiac cell model has several conductance parameters, which are real-valued numbers. These conductance parameters impact the output of the cell model, namely the action potential. Different cells have different parameter values, so our goal is to estimate what these parameters are from _in vivo_ data.

Before we use _in vivo_ data, however, it is helpful to validate the parameter estimation approach. We do this by trying to configure model parameters to reproduce default model output. See this [paper](https://journals.plos.org/ploscompbiol/article?id=10.1371/journal.pcbi.1004242) for more details.

### Types of model outputs
There are three different model outputs we use for parameter estimation:
1. A single action potential.
2. Model output when a irregular pacing protocol is applied. 
3. Current output when a voltage clamp protocol is applied.

### How to run
To run a genetic algorithm for parameter estimation using any of the three model outputs, make the following changes to `main.py`.

1. Define which parameters you want to estimate and add them to a list. __Make sure the list is sorted alphabetically__.
```
PARAMETERS = [
    ga_configs.Parameter(name='G_Na', default_value=3671.2302),
    ga_configs.Parameter(name='G_F', default_value=30.10312),
    ga_configs.Parameter(name='G_Ks', default_value=2.041),
    ga_configs.Parameter(name='G_Kr', default_value=29.8667),
    ga_configs.Parameter(name='G_K1', default_value=28.1492),
    ga_configs.Parameter(name='G_bNa', default_value=0.95),
    ga_configs.Parameter(name='G_NaL', default_value=17.25),
    ga_configs.Parameter(name='G_CaL', default_value=8.635702e-5),
    ga_configs.Parameter(name='G_pCa', default_value=0.4125),
    ga_configs.Parameter(name='G_bCa', default_value=0.727272),
]
PARAMETERS.sort(key=lambda x: x.name)
```

2. Define the model output you will be using. Here are examples of all three. 
```
SAP_PROTOCOL = protocols.SingleActionPotentialProtocol()
IP_PROTOCOL = protocols.IrregularPacingProtocol(
        duration=3,
        stimulation_offsets=[0.1])
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
```

3. Create a genetic algorithm config object with your protocol. This is where you configure population and generation size, as well as other GA hyperparameters. The example shown is for a single action potential, but all three model outputs work similarly.
```
SAP_CONFIG = ga_configs.ParameterTuningConfig(
    population_size=40,
    max_generations=40,
    protocol=SAP_PROTOCOL,
    tunable_parameters=PARAMETERS,
    params_lower_bound=0.1,
    params_upper_bound=3,
    mate_probability=0.9,
    mutate_probability=0.9,
    gene_swap_probability=0.2,
    gene_mutation_probability=0.2,
    tournament_size=4)
```

4. Run the experiment by calling the `run_param_tuning_experiment` function, which is in `parameter_tuning_experiments.py`.
```
parameter_tuning_experiments.run_param_tuning_experiment(
        config=SAP_CONFIG,
        with_output=True)
```

Together, including imports, your `main.py` should look like this (if using a SAP as model output).
```
import parameter_tuning_experiments
import protocols
import ga_configs


PARAMETERS = [
    ga_configs.Parameter(name='G_Na', default_value=3671.2302),
    ga_configs.Parameter(name='G_F', default_value=30.10312),
    ga_configs.Parameter(name='G_Ks', default_value=2.041),
    ga_configs.Parameter(name='G_Kr', default_value=29.8667),
    ga_configs.Parameter(name='G_K1', default_value=28.1492),
    ga_configs.Parameter(name='G_bNa', default_value=0.95),
    ga_configs.Parameter(name='G_NaL', default_value=17.25),
    ga_configs.Parameter(name='G_CaL', default_value=8.635702e-5),
    ga_configs.Parameter(name='G_pCa', default_value=0.4125),
    ga_configs.Parameter(name='G_bCa', default_value=0.727272),
]
# Parameters are sorted alphabetically to maintain order during each
# generation of the genetic algorithm.
PARAMETERS.sort(key=lambda x: x.name)

SAP_PROTOCOL = protocols.SingleActionPotentialProtocol()

SAP_CONFIG = ga_configs.ParameterTuningConfig(
    population_size=40,
    max_generations=40,
    protocol=SAP_PROTOCOL,
    tunable_parameters=PARAMETERS,
    params_lower_bound=0.1,
    params_upper_bound=3,
    mate_probability=0.9,
    mutate_probability=0.9,
    gene_swap_probability=0.2,
    gene_mutation_probability=0.2,
    tournament_size=4)


def main():
    parameter_tuning_experiments.run_param_tuning_experiment(
        config=SAP_CONFIG,
        with_output=True)


if __name__ == '__main__':
    main()
```

## Voltage Clamp Protocol Construction

### Purpose
One of the three model outputs we use for parameter estimation is model output when a voltage clamp protocol is applied. A voltage clamp protocol defines a series of voltage steps to hold a cell at. A good voltage clamp protocol is able to isolate individual currents over the course of the protocol, because this theoretically creates a better target objective for the genetic algorithm.

Typically, voltage clamp protocols are designed by hand by an expert with knowledge about currents. But, we can automate this process using a genetic algorithm and hopefully create voltage clamp protocols which are better than those created by hand. 

### How to run
To run a genetic algorithm for voltage clamp construction, make the following changes in `main.py`.

1. Define a genetic algorithm config object. This is where population size, generation size, and other GA hyperparameters are set.
```
VCO_CONFIG = ga_configs.VoltageOptimizationConfig(
    window=0.1,
    step_size=0.05,
    steps_in_protocol=8,
    step_duration_bounds=(0.05, 0.6),
    step_voltage_bounds=(-.12, .06),
    target_currents=['I_Na', 'I_K1', 'I_To', 'I_CaL', 'I_Kr', 'I_Ks'],
    population_size=4,
    max_generations=4,
    mate_probability=0.9,
    mutate_probability=0.9,
    gene_swap_probability=0.2,
    gene_mutation_probability=0.2,
    tournament_size=2)
```

2. Define a combined voltage clamp optimization object. The reason we create this object is because we are actually running multiple genetic algorithms and then stitching the results we get together. You can read about each attribute in detail in comments in the code.
```
COMBINED_VC_CONFIG = ga_configs.CombinedVCConfig(
    currents=[
        'I_Na', 'I_K1', 'I_To',
        'I_CaL', 'I_Kr', 'I_Ks',
    ],
    step_range=range(5, 6, 1),
    adequate_fitness_threshold=0.95,
    ga_config=VCO_CONFIG)
```

3. Run the experiment by calling the `construct_optimal_protocol` function, which is in `voltage_clamp_optimization_experiments.py`.
```
voltage_clamp_optimization_experiments.construct_optimal_protocol(
        vc_protocol_optimization_config=COMBINED_VC_CONFIG,
        with_output=True)
```

Together, including imports, your `main.py` should look like this.
```
import ga_configs
import voltage_clamp_optimization_experiments


VCO_CONFIG = ga_configs.VoltageOptimizationConfig(
    window=0.1,
    step_size=0.05,
    steps_in_protocol=8,
    step_duration_bounds=(0.05, 0.6),
    step_voltage_bounds=(-.12, .06),
    target_currents=['I_Na', 'I_K1', 'I_To', 'I_CaL', 'I_Kr', 'I_Ks'],
    population_size=4,
    max_generations=4,
    mate_probability=0.9,
    mutate_probability=0.9,
    gene_swap_probability=0.2,
    gene_mutation_probability=0.2,
    tournament_size=2)

COMBINED_VC_CONFIG = ga_configs.CombinedVCConfig(
    currents=[
        'I_Na', 'I_K1', 'I_To',
        'I_CaL', 'I_Kr', 'I_Ks',
    ],
    step_range=range(5, 6, 1),
    adequate_fitness_threshold=0.95,
    ga_config=VCO_CONFIG)


def main():
    voltage_clamp_optimization_experiments.construct_optimal_protocol(
        vc_protocol_optimization_config=COMBINED_VC_CONFIG,
        with_output=True)


if __name__ == '__main__':
    main()
```
