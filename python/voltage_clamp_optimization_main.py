"""Module level comment."""

import voltage_clamp_optimization


def main():
    vc_ga = voltage_clamp_optimization.VCOGeneticAlgorithm(
        population_size=10,
        generations=10)
    vc_ga.run()


if __name__ == '__main__':
    main()
