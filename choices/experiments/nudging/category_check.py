#!/usr/bin/env python3
"""
Print all categories for simplified nudging experiments with their options,
option texts, and group labels.

Usage:
    uv run python -m choices.experiments.nudging.category_check
"""

from choices.experiments.simple_rates import BINARY_FACTORS, create_option_text_fn
from choices.experiments.nudging.simple import format_group_label


def print_all_categories():
    """Print all categories with their details."""
    print("\n" + "=" * 80)
    print("SIMPLIFIED NUDGING EXPERIMENT CATEGORIES")
    print("=" * 80)

    for factor_name, variable in BINARY_FACTORS.items():
        print(f"\n{'─' * 80}")
        print(f"Factor: {factor_name}")
        print(f"{'─' * 80}")
        print(f"Values: {variable.values}")

        # Get option text function for this factor
        option_text_fn = create_option_text_fn(factor_name)

        print("\nOption texts (N=1 and N=5 examples):")
        for value in variable.values:
            print(f"\n  {value}:")
            # N=1 example
            option_n1 = {factor_name: value, "N": 1}
            print(f"    N=1: {option_text_fn(option_n1)}")
            # N=5 example
            option_n5 = {factor_name: value, "N": 5}
            print(f"    N=5: {option_text_fn(option_n5)}")

        print("\nGroup labels (for nudge text):")
        for value in variable.values:
            label = format_group_label(factor_name, value)
            print(f'  {value} -> "{label}"')

    print("\n" + "=" * 80)
    print(f"Total factors: {len(BINARY_FACTORS)}")
    print("=" * 80 + "\n")


if __name__ == "__main__":
    print_all_categories()
