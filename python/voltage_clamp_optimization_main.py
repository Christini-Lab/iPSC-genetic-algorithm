"""Module level comment."""

import voltage_clamp_optimization


def main():
    vc_ga = voltage_clamp_optimization.VCOGeneticAlgorithm()
    vc_ga.run()


if __name__ == '__main__':
    main()
