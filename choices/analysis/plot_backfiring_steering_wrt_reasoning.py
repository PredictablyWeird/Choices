#!/usr/bin/env python3
"""
Create two bar charts comparing reasoning vs non-reasoning conditions by nudge type:
1. Significant Backfiring Rate (grouped by nudge type) - measures % of significant effects that backfired
2. Steerability Magnitude (grouped by nudge type)

Significant Backfiring Rate = (number of cases with significant backfiring) / (number of cases with significant effect)

Designed for 2-column paper layout (side-by-side plots).
Uses CSV data format.
"""

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import os
from scipy import stats


def load_csv_data(filepath):
    """Load data from CSV file."""
    df = pd.read_csv(filepath)

    # Rename columns to match expected format
    df = df.rename(columns={
        'reasoning_condition': 'reasoning',
        'steerability_A': 'steer_A',
        'steerability_B': 'steer_B',
        'abs_steerability': 'abs_steer',
        'backfire_A': 'backfire_A',
        'backfire_B': 'backfire_B'
    })

    return df


def compute_backfire_rate(df):
    """
    Compute significant backfire rate: (number of cases with significant backfiring) / 
                                        (number of cases with significant effect)
    Only counts backfires that are statistically significant, normalized by total significant effects.
    Returns: (rate, lower_ci, upper_ci) in percentage
    """
    if len(df) == 0:
        return 0.0, 0.0, 0.0

    # Count significant backfires (backfire AND significant)
    sig_backfire_A = (df['backfire_A'] & df['sig_A']).sum()
    sig_backfire_B = (df['backfire_B'] & df['sig_B']).sum()
    total_sig_backfires = sig_backfire_A + sig_backfire_B
    
    # Count total significant effects (regardless of direction)
    total_sig_effects = df['sig_A'].sum() + df['sig_B'].sum()
    
    # Compute rate
    if total_sig_effects == 0:
        return 0.0, 0.0, 0.0
    
    rate = (total_sig_backfires / total_sig_effects) * 100
    
    # Compute 95% CI using Wilson score interval
    p = total_sig_backfires / total_sig_effects
    z = stats.norm.ppf(0.975)  # 95% CI
    n = total_sig_effects
    
    # Wilson score interval
    denominator = 1 + (z**2 / n)
    center = (p + (z**2 / (2 * n))) / denominator
    margin = z * np.sqrt((p * (1 - p) + z**2 / (4 * n)) / n) / denominator
    
    lower_ci = max(0, (center - margin) * 100)
    upper_ci = min(100, (center + margin) * 100)
    
    return rate, lower_ci, upper_ci


def compute_steerability_magnitude(df):
    """
    Compute average |steerability| magnitude.
    Returns: (mean, lower_ci, upper_ci)
    """
    if len(df) == 0:
        return 0.0, 0.0, 0.0
    
    values = df['abs_steer'].values
    mean = np.mean(values)
    
    if len(values) < 2:
        return mean, mean, mean
    
    # Compute 95% CI using t-distribution
    sem = stats.sem(values)  # Standard error of the mean
    ci = stats.t.interval(0.95, len(values) - 1, loc=mean, scale=sem)
    
    return mean, ci[0], ci[1]


def create_bar_charts(input_file, output_dir='plots'):
    """Create two professional bar charts for reasoning comparison by nudge type."""

    # Load data from CSV
    df = load_csv_data(input_file)

    # Classify reasoning conditions
    # "With Reasoning": before, low (reasoning enabled)
    # "No Reasoning": none, off (reasoning disabled)
    reasoning_map = {
        'before': 'With Reasoning',
        'low': 'With Reasoning',
        'none': 'No Reasoning',
        'off': 'No Reasoning'
    }

    df['reasoning_group'] = df['reasoning'].map(reasoning_map)

    # Nudge types with display labels (+ Overall at the end)
    nudge_types = ['emotional', 'few_shot_3', 'survey_preference', 'user_preference', 'weak_evidence', 'overall']
    nudge_labels = ['Emotional\nAppeal', 'Few-Shot\nExamples', 'Survey\nResults', 'User\nPreference', 'Weak\nEvidence', 'Overall']

    # Compute metrics for each nudge type and reasoning group
    backfire_no_reasoning = []
    backfire_with_reasoning = []
    backfire_no_reasoning_lower = []
    backfire_no_reasoning_upper = []
    backfire_with_reasoning_lower = []
    backfire_with_reasoning_upper = []
    steer_no_reasoning = []
    steer_with_reasoning = []
    steer_no_reasoning_lower = []
    steer_no_reasoning_upper = []
    steer_with_reasoning_lower = []
    steer_with_reasoning_upper = []

    for nudge in nudge_types:
        if nudge == 'overall':
            # For Overall, use ALL data (not filtered by nudge type)
            no_reason = df[df['reasoning_group'] == 'No Reasoning']
            with_reason = df[df['reasoning_group'] == 'With Reasoning']
        else:
            nudge_df = df[df['nudge_type'] == nudge]
            no_reason = nudge_df[nudge_df['reasoning_group'] == 'No Reasoning']
            with_reason = nudge_df[nudge_df['reasoning_group'] == 'With Reasoning']

        # Backfire rates with CIs
        bf_no_rate, bf_no_lower, bf_no_upper = compute_backfire_rate(no_reason)
        bf_with_rate, bf_with_lower, bf_with_upper = compute_backfire_rate(with_reason)
        backfire_no_reasoning.append(bf_no_rate)
        backfire_with_reasoning.append(bf_with_rate)
        backfire_no_reasoning_lower.append(bf_no_lower)
        backfire_no_reasoning_upper.append(bf_no_upper)
        backfire_with_reasoning_lower.append(bf_with_lower)
        backfire_with_reasoning_upper.append(bf_with_upper)

        # Steerability with CIs
        st_no_mean, st_no_lower, st_no_upper = compute_steerability_magnitude(no_reason)
        st_with_mean, st_with_lower, st_with_upper = compute_steerability_magnitude(with_reason)
        steer_no_reasoning.append(st_no_mean)
        steer_with_reasoning.append(st_with_mean)
        steer_no_reasoning_lower.append(st_no_lower)
        steer_no_reasoning_upper.append(st_no_upper)
        steer_with_reasoning_lower.append(st_with_lower)
        steer_with_reasoning_upper.append(st_with_upper)

    # Print computed values
    print("Significant Backfire Rate by Nudge Type (% of significant effects that backfired):")
    for i, nudge in enumerate(nudge_types):
        print(f"  {nudge}: No Reasoning={backfire_no_reasoning[i]:.1f}%, With Reasoning={backfire_with_reasoning[i]:.1f}%")

    print("\nSteerability Magnitude by Nudge Type:")
    for i, nudge in enumerate(nudge_types):
        print(f"  {nudge}: No Reasoning={steer_no_reasoning[i]:.2f}, With Reasoning={steer_with_reasoning[i]:.2f}")

    # Professional color palette (muted, publication-quality)
    color_no_reasoning = '#D95F02'   # Orange (ColorBrewer Dark2)
    color_with_reasoning = '#1B9E77'  # Teal (ColorBrewer Dark2)

    # Create output directory
    os.makedirs(output_dir, exist_ok=True)

    # Bar positions - add extra space before Overall for divider
    x = np.arange(len(nudge_types)).astype(float)
    x[-1] = x[-1] + 0.7  # Add extra space before Overall
    width = 0.38
    divider_x = x[-2] + 0.85  # Position for vertical divider line

    # Set professional style with larger fonts
    plt.rcParams.update({
        'font.family': 'sans-serif',
        'font.sans-serif': ['Arial', 'Helvetica', 'DejaVu Sans'],
        'font.size': 18,
        'axes.linewidth': 1.5,
        'axes.labelweight': 'bold',
        'axes.titleweight': 'bold',
    })

    # =========================================================================
    # Plot 1: Backfiring Rate by Nudge Type
    # =========================================================================
    fig1, ax1 = plt.subplots(figsize=(10, 5.5))

    # Compute error bar values (distance from mean to CI bounds)
    # Handle edge cases where CIs might be invalid
    backfire_no_err_lower = []
    backfire_no_err_upper = []
    for i in range(len(backfire_no_reasoning)):
        lower = max(0, backfire_no_reasoning[i] - backfire_no_reasoning_lower[i])
        upper = max(0, backfire_no_reasoning_upper[i] - backfire_no_reasoning[i])
        backfire_no_err_lower.append(lower)
        backfire_no_err_upper.append(upper)
    
    backfire_with_err_lower = []
    backfire_with_err_upper = []
    for i in range(len(backfire_with_reasoning)):
        lower = max(0, backfire_with_reasoning[i] - backfire_with_reasoning_lower[i])
        upper = max(0, backfire_with_reasoning_upper[i] - backfire_with_reasoning[i])
        backfire_with_err_lower.append(lower)
        backfire_with_err_upper.append(upper)

    bars1_no = ax1.bar(x - width/2, backfire_no_reasoning, width, label='No Reasoning',
                       color=color_no_reasoning, edgecolor='white', linewidth=1.5,
                       alpha=0.9, zorder=3, yerr=[backfire_no_err_lower, backfire_no_err_upper],
                       capsize=4, error_kw={'elinewidth': 1.5, 'capthick': 1.5})
    bars1_with = ax1.bar(x + width/2, backfire_with_reasoning, width, label='With Reasoning',
                         color=color_with_reasoning, edgecolor='white', linewidth=1.5,
                         alpha=0.9, zorder=3, yerr=[backfire_with_err_lower, backfire_with_err_upper],
                         capsize=4, error_kw={'elinewidth': 1.5, 'capthick': 1.5})

    # Add vertical divider line before Overall
    ax1.axvline(x=divider_x, color='#888888', linestyle='-', linewidth=1.5, zorder=2)

    # Compute max for y-axis and label positioning
    max_backfire = max(max(backfire_no_reasoning_upper), max(backfire_with_reasoning_upper))

    # Add value labels on bars with larger fonts
    for i, bar in enumerate(bars1_no):
        height = bar.get_height()
        # Position label above error bar
        upper_bound = backfire_no_reasoning_upper[i]
        ax1.text(bar.get_x() + bar.get_width()/2, upper_bound + max_backfire * 0.02,
                f'{height:.1f}%', ha='center', va='bottom', fontsize=14,
                fontweight='semibold', color='#333333')
    for i, bar in enumerate(bars1_with):
        height = bar.get_height()
        upper_bound = backfire_with_reasoning_upper[i]
        ax1.text(bar.get_x() + bar.get_width()/2, upper_bound + max_backfire * 0.02,
                f'{height:.1f}%', ha='center', va='bottom', fontsize=14,
                fontweight='semibold', color='#333333')

    # Style axes with larger fonts, no title
    ax1.set_xlabel('Influence Type', fontsize=20, fontweight='bold', labelpad=12)
    ax1.set_ylabel('Backfire Rate (%)', fontsize=20, fontweight='bold', labelpad=12)
    ax1.set_xticks(x)
    ax1.set_xticklabels(nudge_labels, fontsize=16, fontweight='medium')
    ax1.set_ylim(0, max_backfire * 1.3)
    ax1.tick_params(axis='y', labelsize=16)

    # Add subtle grid
    ax1.yaxis.grid(True, linestyle='--', alpha=0.4, zorder=0)
    ax1.set_axisbelow(True)

    # Clean up spines
    ax1.spines['top'].set_visible(False)
    ax1.spines['right'].set_visible(False)
    ax1.spines['left'].set_color('#666666')
    ax1.spines['bottom'].set_color('#666666')

    # Legend with clean styling and larger font
    legend1 = ax1.legend(fontsize=16, loc='upper right', frameon=True,
                         fancybox=True, shadow=False, edgecolor='#cccccc')
    legend1.get_frame().set_alpha(0.95)

    plt.tight_layout()
    fig1.savefig(f'{output_dir}/significant_backfiring_rate_comparison.pdf', bbox_inches='tight', dpi=300)
    print(f"\nSaved: {output_dir}/significant_backfiring_rate_comparison.pdf")
    plt.close(fig1)

    # =========================================================================
    # Plot 2: Steerability Magnitude by Nudge Type
    # =========================================================================
    fig2, ax2 = plt.subplots(figsize=(10, 5.5))

    # Compute error bar values for steerability
    steer_no_err_lower = []
    steer_no_err_upper = []
    for i in range(len(steer_no_reasoning)):
        lower = max(0, steer_no_reasoning[i] - steer_no_reasoning_lower[i])
        upper = max(0, steer_no_reasoning_upper[i] - steer_no_reasoning[i])
        steer_no_err_lower.append(lower)
        steer_no_err_upper.append(upper)
    
    steer_with_err_lower = []
    steer_with_err_upper = []
    for i in range(len(steer_with_reasoning)):
        lower = max(0, steer_with_reasoning[i] - steer_with_reasoning_lower[i])
        upper = max(0, steer_with_reasoning_upper[i] - steer_with_reasoning[i])
        steer_with_err_lower.append(lower)
        steer_with_err_upper.append(upper)

    bars2_no = ax2.bar(x - width/2, steer_no_reasoning, width, label='No Reasoning',
                       color=color_no_reasoning, edgecolor='white', linewidth=1.5,
                       alpha=0.9, zorder=3, yerr=[steer_no_err_lower, steer_no_err_upper],
                       capsize=4, error_kw={'elinewidth': 1.5, 'capthick': 1.5})
    bars2_with = ax2.bar(x + width/2, steer_with_reasoning, width, label='With Reasoning',
                         color=color_with_reasoning, edgecolor='white', linewidth=1.5,
                         alpha=0.9, zorder=3, yerr=[steer_with_err_lower, steer_with_err_upper],
                         capsize=4, error_kw={'elinewidth': 1.5, 'capthick': 1.5})

    # Add vertical divider line before Overall
    ax2.axvline(x=divider_x, color='#888888', linestyle='-', linewidth=1.5, zorder=2)

    # Add value labels on bars with larger fonts
    max_steer = max(max(steer_no_reasoning_upper), max(steer_with_reasoning_upper))
    for i, bar in enumerate(bars2_no):
        height = bar.get_height()
        upper_bound = steer_no_reasoning_upper[i]
        ax2.text(bar.get_x() + bar.get_width()/2, upper_bound + max_steer * 0.02,
                f'{height:.2f}', ha='center', va='bottom', fontsize=14,
                fontweight='semibold', color='#333333')
    for i, bar in enumerate(bars2_with):
        height = bar.get_height()
        upper_bound = steer_with_reasoning_upper[i]
        ax2.text(bar.get_x() + bar.get_width()/2, upper_bound + max_steer * 0.02,
                f'{height:.2f}', ha='center', va='bottom', fontsize=14,
                fontweight='semibold', color='#333333')

    # Style axes with larger fonts, no title
    ax2.set_xlabel('Influence Type', fontsize=20, fontweight='bold', labelpad=12)
    ax2.set_ylabel('Mean Absolute Steerability', fontsize=20, fontweight='bold', labelpad=12)
    ax2.set_xticks(x)
    ax2.set_xticklabels(nudge_labels, fontsize=16, fontweight='medium')
    ax2.set_ylim(0, max_steer * 1.3)
    ax2.tick_params(axis='y', labelsize=16)

    # Add subtle grid
    ax2.yaxis.grid(True, linestyle='--', alpha=0.4, zorder=0)
    ax2.set_axisbelow(True)

    # Clean up spines
    ax2.spines['top'].set_visible(False)
    ax2.spines['right'].set_visible(False)
    ax2.spines['left'].set_color('#666666')
    ax2.spines['bottom'].set_color('#666666')

    # Legend with clean styling and larger font
    legend2 = ax2.legend(fontsize=16, loc='upper right', frameon=True,
                         fancybox=True, shadow=False, edgecolor='#cccccc')
    legend2.get_frame().set_alpha(0.95)

    plt.tight_layout()
    fig2.savefig(f'{output_dir}/steerability_magnitude_comparison.pdf', bbox_inches='tight', dpi=300)
    print(f"Saved: {output_dir}/steerability_magnitude_comparison.pdf")
    plt.close(fig2)

    print("\nDone! Created professional bar chart PDFs with nudge types on x-axis.")


if __name__ == "__main__":
    input_file = "main_results_jan28.csv"
    create_bar_charts(input_file)
