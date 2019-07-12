"""Contains functions to run VC optimization genetic algorithm experiments."""

import ga_configs
import voltage_clamp_optimization


def run_voltage_clamp_experiment(
        config: ga_configs.VoltageOptimizationConfig,
        full_output: bool=False) -> None:
    result = voltage_clamp_optimization.VCOGeneticAlgorithm(config=config).run()

    if full_output:
        result.generate_heatmap()
