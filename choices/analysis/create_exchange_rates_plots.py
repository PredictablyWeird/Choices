#!/usr/bin/env python3
"""
Generate exchange rate plots for experiments.

This script creates exchange rate plots from experiment results using
the new choices framework results format.
"""

import argparse
import math
import os
from pathlib import Path
from typing import Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import statsmodels.api as sm
from matplotlib.colors import LogNorm
from matplotlib.ticker import FuncFormatter, LogFormatterSciNotation, ScalarFormatter

from choices.results import ExperimentResults
from choices.utils import find_result_files
from choices.variable import AnalysisType

###############################################################################
# Measure Titles: to display on the x-axis label
###############################################################################
MEASURE_TITLES = {
    "terminal_illness": "Terminal Illness Saved",
    "death": "Deaths",
    "happiness": "Minutes of Happiness",
    "wealth": "Percent Wealthier",
    "qaly": "Quality-Adjusted Life Years",
    "saved": "Number Saved",
    # Add more if needed
}


###############################################################################
#                           HELPER FUNCTIONS
###############################################################################


def plot_single_model_bar_chart(
    ax,
    ratio_dict,
    title="",
    plot_scale="log",
    min_ratio=1e-2,
    max_ratio=1e2,
    yaxis_format="plain",
    bar_label_format="scientific",
    bar_label_offset=0.05,
    bar_font_size=9,
    bar_font_color="black",
    bar_color_above="teal",
    bar_color_below="orange",
):
    """
    Draws a single pivot bar chart on a given Axes.

    ratio_dict: {display_name -> ratio (float >= 0, inf, -inf, or None)}

    For +/- ∞:
      - Negative infinity => draw a bar from min_ratio up to 1.0 (fully within the plot),
        colored bar_color_below, label "-∞" at the bottom.
      - Positive infinity => draw a bar from 1.0 up to max_ratio,
        colored bar_color_above, label "∞" at the top.
      - No black outline, so it matches the non-dummy bars.

    All other logic remains unchanged.
    """

    def format_bar_value(val, mode="scientific"):
        # Always use decimal format
        if val < 1.0:
            return f"{val:.3f}"
        else:
            return f"{val:.3f}"

    # Separate into groups in the order we want to display them:
    pos_inf_entries = []
    neg_inf_entries = []
    above_one = []
    pivot_vals = []
    below_one = []
    zero_entries = []

    for x_name, val in ratio_dict.items():
        if val is None:
            continue
        if math.isinf(val):
            if val > 0:
                pos_inf_entries.append((x_name, val))
            else:  # val < 0 => negative infinity
                neg_inf_entries.append((x_name, val))
        elif val == 0:
            zero_entries.append((x_name, val))
        elif 0 < val < 1:
            below_one.append((x_name, val))
        elif val > 1:
            above_one.append((x_name, val))
        else:
            # we treat val ~ 1.0
            pivot_vals.append((x_name, val))

    # Sort each group
    pos_inf_entries.sort(key=lambda x: x[0])
    above_one.sort(key=lambda x: x[1], reverse=True)
    pivot_vals.sort(key=lambda x: x[0])
    below_one.sort(key=lambda x: x[1], reverse=True)
    zero_entries.sort(key=lambda x: x[0])
    neg_inf_entries.sort(key=lambda x: x[0])

    # Final order: +inf, above_one, pivot_vals, below_one, zero, -inf
    combined = (
        pos_inf_entries
        + above_one
        + pivot_vals
        + below_one
        + zero_entries
        + neg_inf_entries
    )

    x_labels = [t[0] for t in combined]
    x_ratios = [t[1] for t in combined]
    x_positions = np.arange(len(x_labels))

    # For labeling bars that go off the top or bottom
    if plot_scale == "log":
        top_label_y = max_ratio / (10**bar_label_offset)
        bottom_label_y = min_ratio * (10**bar_label_offset)
    else:
        y_range = max_ratio - min_ratio
        top_label_y = max_ratio - bar_label_offset * y_range
        bottom_label_y = min_ratio + bar_label_offset * y_range

    for i, val in enumerate(x_ratios):
        if val is None:
            continue

        # ----- NEGATIVE INFINITY BAR -----
        if math.isinf(val) and val < 0:
            # bar from min_ratio up to 1.0
            bottom_ = min_ratio
            top_ = 1.0
            height_ = abs(top_ - bottom_)
            ax.bar(
                i,
                height_,
                bottom=bottom_,
                width=0.6,
                color=bar_color_below,
                edgecolor="none",
            )
            # Label "-∞" near the bottom, just like we do if val < min_ratio
            ax.text(
                i,
                bottom_label_y,
                r"$-\infty$",
                ha="center",
                va="bottom",
                color=bar_font_color,
                fontsize=bar_font_size,
                rotation=90,
            )
            continue

        # ----- POSITIVE INFINITY BAR -----
        if math.isinf(val) and val > 0:
            # bar from 1.0 up to max_ratio
            bottom_ = 1.0
            top_ = max_ratio
            height_ = abs(top_ - bottom_)
            ax.bar(
                i,
                height_,
                bottom=bottom_,
                width=0.6,
                color=bar_color_above,
                edgecolor="none",
            )
            # Label "∞" near the top, like val > max_ratio
            ax.text(
                i,
                top_label_y,
                r"$\infty$",
                ha="center",
                va="top",
                color=bar_font_color,
                fontsize=bar_font_size + 4,
                rotation=90,
            )
            continue

        # ----- ZERO -----
        if val == 0:
            ax.bar(
                i,
                0,
                bottom=min_ratio,
                width=0.6,
                color=bar_color_below,
                edgecolor="none",
            )
            label_txt = "0"
            ax.text(
                i,
                bottom_label_y,
                label_txt,
                ha="center",
                va="bottom",
                color=bar_font_color,
                fontsize=bar_font_size,
                rotation=90,
            )
            continue

        # ----- FINITE POSITIVE RATIO -----
        if 0 < val < 1:
            bar_col = bar_color_below
            bottom_ = val
            top_ = 1.0
            height_ = abs(top_ - bottom_)
            ax.bar(
                i, height_, bottom=bottom_, width=0.6, color=bar_col, edgecolor="none"
            )
            # Always show the exchange rate value at the top of the bar
            label_txt = format_bar_value(val, bar_label_format)
            ax.text(
                i,
                top_,
                label_txt,
                ha="center",
                va="bottom",
                color=bar_font_color,
                fontsize=bar_font_size,
            )

        elif val >= 1:
            bar_col = bar_color_above
            bottom_ = 1.0
            top_ = val
            height_ = abs(val - 1.0)
            ax.bar(
                i, height_, bottom=bottom_, width=0.6, color=bar_col, edgecolor="none"
            )
            # Always show the exchange rate value at the top of the bar
            label_txt = format_bar_value(val, bar_label_format)
            ax.text(
                i,
                top_,
                label_txt,
                ha="center",
                va="bottom",
                color=bar_font_color,
                fontsize=bar_font_size,
            )

    ax.set_title(title, fontsize=18)
    ax.set_xticks(x_positions)
    ax.set_xticklabels(x_labels, rotation=45, ha="right", fontsize=14)
    ax.axhline(y=1.0, color="gray", linestyle="--")
    ax.set_ylabel("Two-way geometric mean ratio")

    # y-scale
    if plot_scale == "log":
        ax.set_yscale("log")
        ax.set_ylim(min_ratio, max_ratio)
        if yaxis_format == "scientific":
            ax.yaxis.set_major_formatter(LogFormatterSciNotation())
        else:

            def log_plain_formatter(y, pos):
                return f"{y:.3g}"

            ax.yaxis.set_major_formatter(FuncFormatter(log_plain_formatter))
    else:
        ax.set_ylim(min_ratio, max_ratio)
        sfmt = ScalarFormatter()
        if yaxis_format == "scientific":
            sfmt.set_scientific(True)
            sfmt.set_powerlimits((0, 0))
        else:
            sfmt.set_scientific(False)
        ax.yaxis.set_major_formatter(sfmt)


def fit_utility_curves(df, return_mse=False):
    """
    Fits a line of the form:
        utility_mean = b + a * ln(N)
    for each X. Returns (slopes, intercepts) or (slopes, intercepts, mse_scores).

    MSE = model.mse_resid in statsmodels (sum of squared residuals / df_resid).
    """
    slopes, intercepts = {}, {}
    mse_scores = {}

    grouped = df.groupby("X", dropna=True)
    for xval, gdf in grouped:
        # Need at least 2 distinct lnN points
        if len(gdf["lnN"].unique()) < 2:
            continue
        xvals = gdf["lnN"].values
        yvals = gdf["utility_mean"].values

        X_design = sm.add_constant(xvals)  # [1, lnN]
        model = sm.OLS(yvals, X_design).fit()
        b = model.params[0]
        a = model.params[1]
        slopes[xval] = a
        intercepts[xval] = b

        if return_mse:
            mse_scores[xval] = model.mse_resid

    if return_mse:
        return slopes, intercepts, mse_scores
    return slopes, intercepts


def two_way_geometric_exchange_rate(
    Xi,
    Xj,
    measure_values,
    slopes,
    intercepts,
    skip_if_negative_slope=True,
    verbose=False,
    allow_negative_slopes=False,
):
    """
    Returns the two-way geometric mean ratio for Xi vs. Xj.
    If skip_if_negative_slope=False but one slope is negative => float('-inf').

    The logic:
      - For each N_j, solve for M_i giving same utility => ratio_fwd = M_i/N_j
      - For each N_i, solve for M_j => ratio_bwd = N_i/M_j
      - Returns exp(mean(ln(ratio_fwd), ln(ratio_bwd)))  [the "two-way" ratio].
    """
    if Xi not in slopes or Xi not in intercepts:
        return None
    if Xj not in slopes or Xj not in intercepts:
        return None

    a_i, b_i = slopes[Xi], intercepts[Xi]
    a_j, b_j = slopes[Xj], intercepts[Xj]

    if skip_if_negative_slope:
        if a_i < 0 or a_j < 0:
            if verbose:
                print(
                    f"Skipping {Xi}->{Xj} because one slope is negative: a_i={a_i}, a_j={a_j}"
                )
            return None
    elif not allow_negative_slopes:
        if a_i < 0 or a_j < 0:
            if verbose:
                print(
                    f"Forcing ratio to -inf because slope is negative (a_i={a_i}, a_j={a_j})"
                )
            return float("-inf")
    # If allow_negative_slopes=True, we continue with the calculation even with negative slopes

    log_ratios = []

    for N_j in measure_values:
        if N_j <= 0:
            continue
        try:
            # b_i + a_i ln(M_i) = b_j + a_j ln(N_j)
            # ln(M_i) = [b_j - b_i + a_j ln(N_j)] / a_i
            M_i = math.exp((b_j - b_i + a_j * math.log(N_j)) / a_i)
            ratio_fwd = M_i / N_j
            if ratio_fwd > 0 and not math.isinf(ratio_fwd):
                log_ratios.append(math.log(ratio_fwd))
        except (ValueError, OverflowError):
            pass

    for N_i in measure_values:
        if N_i <= 0:
            continue
        try:
            # b_j + a_j ln(M_j) = b_i + a_i ln(N_i)
            # ln(M_j) = [b_i - b_j + a_i ln(N_i)] / a_j
            M_j = math.exp((b_i - b_j + a_i * math.log(N_i)) / a_j)
            ratio_bwd = N_i / M_j
            if ratio_bwd > 0 and not math.isinf(ratio_bwd):
                log_ratios.append(math.log(ratio_bwd))
        except (ValueError, OverflowError):
            pass

    if not log_ratios:
        return None
    mean_log_ratio = sum(log_ratios) / len(log_ratios)
    return math.exp(mean_log_ratio)


def geometric_mean(values):
    filtered = [v for v in values if v is not None and v > 0 and not math.isinf(v)]
    if not filtered:
        return None
    logs = [math.log(v) for v in filtered]
    return math.exp(sum(logs) / len(logs))


def infer_numerical_variable(
    results: ExperimentResults, provided_var: str = None
) -> str:
    """
    Infer the numerical variable name from results.

    Prefers LOG_NUMERICAL over NUMERICAL if both exist.
    Falls back to provided_var if it exists in results.

    Args:
        results: ExperimentResults object
        provided_var: Optional variable name provided by user

    Returns:
        Name of the numerical variable to use
    """
    # Get all numerical variables (both NUMERICAL and LOG_NUMERICAL)
    all_numerical = {}
    # Check analysis config for numerical fields
    for field_name, analysis_type in results.graph.analysis_config.fields.items():
        if analysis_type in (AnalysisType.NUMERICAL, AnalysisType.LOG_NUMERICAL):
            # Find corresponding variable
            for var in results.graph.variables:
                if var.name == field_name:
                    all_numerical[field_name] = var
                    break

    if not all_numerical:
        raise ValueError(
            "No numerical variables found in results. "
            "Exchange rate plots require at least one numerical variable."
        )

    # If user provided a variable, validate it exists
    if provided_var:
        if provided_var not in all_numerical:
            available = list(all_numerical.keys())
            raise ValueError(
                f"Numerical variable '{provided_var}' not found in results. "
                f"Available numerical variables: {available}"
            )
        return provided_var

    # Auto-detect: prefer LOG_NUMERICAL over NUMERICAL
    log_numerical_vars = {}
    for name, var in all_numerical.items():
        atype = results.graph.analysis_config.get_analysis_type(name)
        if atype == AnalysisType.LOG_NUMERICAL:
            log_numerical_vars[name] = var

    if log_numerical_vars:
        # Use first LOG_NUMERICAL variable found
        return list(log_numerical_vars.keys())[0]
    else:
        # Fall back to first NUMERICAL variable
        return list(all_numerical.keys())[0]


def validate_factor(results: ExperimentResults, factor_name: str) -> None:
    """
    Validate that the factor exists in results and is categorical.

    Args:
        results: ExperimentResults object
        factor_name: Name of the factor to validate

    Raises:
        ValueError if factor doesn't exist or isn't categorical
    """
    # Check if factor is in analysis config and is categorical
    atype = results.graph.analysis_config.get_analysis_type(factor_name)

    if atype is None:
        categorical_vars = list(results.graph.get_categorical_variables().keys())
        raise ValueError(
            f"Factor '{factor_name}' not found in analysis config. "
            f"Available categorical variables: {categorical_vars}"
        )

    if atype != AnalysisType.CATEGORICAL:
        raise ValueError(
            f"Factor '{factor_name}' is not categorical (type: {atype.value}). "
            "Exchange rate plots require a categorical factor."
        )


def load_exchange_rates_data(
    results_dir: str, factor_name: str, numerical_var: str = None
) -> Tuple[pd.DataFrame, str]:
    """
    Load exchange rates experiment results and prepare DataFrame for plotting.

    Args:
        results_dir: Directory containing preference_graph and utility_model JSON files
        factor_name: Name of the categorical factor (e.g., "gender", "ethnicity")
        numerical_var: Name of the numerical variable (default: auto-detected from results)

    Returns:
        Tuple of (DataFrame, numerical_var_name) where:
        - DataFrame has columns: option_id, label, utility_mean, utility_variance, {numerical_var}, X, lnN
        - numerical_var_name is the name of the numerical variable used (may be auto-detected)
    """
    # Find result files
    graph_path, model_path, suffix = find_result_files(results_dir)
    if graph_path is None or model_path is None:
        raise FileNotFoundError(
            f"Could not find preference_graph_*.json and utility_model_*.json files in {results_dir}"
        )

    # Load results using the new API
    results = ExperimentResults.load(results_dir, suffix)

    # Validate and infer variables
    validate_factor(results, factor_name)
    numerical_var = infer_numerical_variable(results, numerical_var)

    print(f"Using numerical variable: {numerical_var}")
    print(f"Using categorical factor: {factor_name}")

    # Build DataFrame from options and utilities
    rows = []
    for opt in results.graph.options:
        opt_id = str(opt.id)

        # Check if this option has a utility
        if opt_id not in results.utility_model.utilities:
            continue

        util_data = results.utility_model.utilities[opt_id]
        util_mean = util_data["mean"]
        util_var = util_data["variance"]

        # Extract the factor value (X) and numerical value (N)
        # These can be in _extra_fields or as direct attributes
        factor_value = opt.get(factor_name) or opt.get("factor_value")
        N_val = opt.get(numerical_var)

        if factor_value is None or N_val is None:
            continue

        rows.append(
            {
                "option_id": opt.id,
                "label": opt.label,
                "utility_mean": util_mean,
                "utility_variance": util_var,
                numerical_var: N_val,
                "X": factor_value,  # Use X as the standard name for the factor
            }
        )

    df = pd.DataFrame(rows)

    # Clean and prepare data
    df = df.dropna(subset=[numerical_var, "X"])
    df[numerical_var] = pd.to_numeric(df[numerical_var], errors="coerce")
    df = df[df[numerical_var] > 0]
    df["lnN"] = np.log(df[numerical_var])

    return df, numerical_var


def get_factor_values_and_N_values(df: pd.DataFrame, numerical_var: str = "N") -> tuple:
    """
    Extract unique factor values (X) and N values from the DataFrame.

    Returns:
        (X_values, N_values_list) - sorted lists of unique values
    """
    X_values = sorted(df["X"].unique())
    N_values_list = sorted(df[numerical_var].unique())
    return X_values, N_values_list


def plot_appendix_multi_model_average(
    df,
    N_values_list,
    canonical_X,
    model_name="model",
    measure="",
    include_Xs=None,
    exclude_Xs=None,
    plot_scale="log",
    min_ratio=1e-2,
    max_ratio=1e2,
    yaxis_format="plain",
    bar_label_format="scientific",
    bar_label_offset=0.05,
    bar_font_size=9,
    bar_font_color="black",
    bar_color_above="teal",
    bar_color_below="orange",
    skip_if_negative_slope=True,
    plot_auxiliary_figures=False,
    plot_mse=False,
    MSE_threshold=None,
    aggregator_plot_title=None,
    aggregator_plot_y_label=None,
    x_name_mapping=None,
    numerical_var="N",
):
    """
    Simplified version of plot_appendix_multi_model_average for single model.

    Creates exchange rate plots from a DataFrame with utility data.

    Args:
        numerical_var: Name of the numerical variable (default: "N")
    """

    def make_log10_regression_figure(
        df, slopes, intercepts, model_key="", category="", measure="", numerical_var="N"
    ):
        measure_title = MEASURE_TITLES.get(measure, measure)
        df = df[df["X"].isin(slopes.keys())].copy()
        if df.empty:
            return None

        df["log10N"] = np.log10(df[numerical_var])

        X_list_local = sorted(df["X"].unique())
        if not X_list_local:
            return None

        fig_reg, ax_reg = plt.subplots(figsize=(10, 7))

        palette = sns.color_palette("tab10", n_colors=len(X_list_local))
        color_map = {xval: palette[i] for i, xval in enumerate(X_list_local)}

        for xval in X_list_local:
            sub = df[df["X"] == xval]
            color_ = color_map[xval]

            ax_reg.scatter(
                sub["log10N"], sub["utility_mean"], color=color_, alpha=0.7, label=xval
            )

            sns.regplot(
                data=sub,
                x="log10N",
                y="utility_mean",
                ci=95,
                truncate=False,
                scatter=False,
                line_kws={"color": color_},
                ax=ax_reg,
            )

        ax_reg.set_title(f"{model_key} - {category}/{measure}", fontsize=14)
        ax_reg.set_xlabel(f"log10({measure_title})", fontsize=12)
        ax_reg.set_ylabel("Utility", fontsize=12)

        handles, labels = ax_reg.get_legend_handles_labels()
        by_label = dict(zip(labels, handles))
        ax_reg.legend(
            by_label.values(),
            by_label.keys(),
            loc="upper center",
            bbox_to_anchor=(0.5, -0.1),
            ncol=3,
            fancybox=True,
        )

        fig_reg.subplots_adjust(bottom=0.25)
        return fig_reg

    figures_dict = {}

    # Filter X values
    if include_Xs is not None:
        df = df[df["X"].isin(include_Xs)].copy()
    if exclude_Xs is not None:
        df = df[~df["X"].isin(exclude_Xs)].copy()

    # Fit utility curves
    if plot_mse:
        slopes, intercepts, mse_scores = fit_utility_curves(df, return_mse=True)
    else:
        slopes, intercepts = fit_utility_curves(df, return_mse=False)
        mse_scores = {}

    # MSE filter
    if MSE_threshold is not None and plot_mse:
        for xval in list(slopes.keys()):
            mse_val = mse_scores.get(xval, float("inf"))
            if mse_val > MSE_threshold:
                del slopes[xval]
                del intercepts[xval]
                mse_scores.pop(xval, None)

    X_list = sorted(slopes.keys())
    if not X_list or (canonical_X not in X_list):
        raise ValueError(
            f"canonical_X='{canonical_X}' not in fitted slopes. Available: {X_list}"
        )

    # Check if canonical_X has negative utility (negative intercept)
    canonical_intercept = intercepts.get(canonical_X, 0)
    scale_reversed = canonical_intercept < 0

    # For negative utilities, we need to handle negative slopes differently
    effective_skip_negative_slope = skip_if_negative_slope and not scale_reversed

    # Build ratio dict
    ratio_dict = {}
    for X_other in X_list:
        if X_other == canonical_X:
            continue
        val = two_way_geometric_exchange_rate(
            canonical_X,
            X_other,
            N_values_list,
            slopes,
            intercepts,
            skip_if_negative_slope=effective_skip_negative_slope,
            allow_negative_slopes=scale_reversed,
        )
        ratio_dict[X_other] = val
    ratio_dict[canonical_X] = 1.0

    # Remap the X names if x_name_mapping is provided
    remapped_dict = {}
    for orig_x, ratio_val in ratio_dict.items():
        if x_name_mapping and orig_x in x_name_mapping:
            new_key = x_name_mapping[orig_x]
        else:
            new_key = orig_x
        remapped_dict[new_key] = ratio_val
    ratio_dict = remapped_dict

    # Calculate appropriate min_ratio and max_ratio from the data
    finite_ratios = [
        v for v in ratio_dict.values() if v is not None and not math.isinf(v) and v > 0
    ]
    if finite_ratios:
        min_val = min(finite_ratios)
        max_val = max(finite_ratios)

        min_log10 = math.floor(math.log10(min_val))
        max_log10 = math.ceil(math.log10(max_val))

        candidate_min = 10 ** (min_log10 - 1)
        candidate_max = 10 ** (max_log10 + 1)

        data_range = max_val / min_val
        candidate_range = candidate_max / candidate_min

        if candidate_range > data_range * 10:
            if max_log10 - min_log10 <= 1:
                effective_min_ratio = 10 ** (min_log10 - 1)
                effective_max_ratio = 10 ** (max_log10 + 1)
            else:
                effective_min_ratio = 10**min_log10
                effective_max_ratio = 10**max_log10
        else:
            effective_min_ratio = candidate_min
            effective_max_ratio = candidate_max

        if min_ratio == 1e-2:  # Using default value
            min_ratio = effective_min_ratio
        if max_ratio == 1e2:  # Using default value
            max_ratio = effective_max_ratio

    # Final aggregator bar chart
    fig_agg, ax_agg = plt.subplots(figsize=(12, 6))

    if aggregator_plot_title is None:
        aggregator_plot_title = f"Exchange Rates (pivot={canonical_X})"
    if aggregator_plot_y_label is None:
        aggregator_plot_y_label = f"Exchange Rate Relative to {canonical_X}"

    plot_single_model_bar_chart(
        ax_agg,
        ratio_dict,
        title=aggregator_plot_title,
        plot_scale=plot_scale,
        min_ratio=min_ratio,
        max_ratio=max_ratio,
        yaxis_format=yaxis_format,
        bar_label_format=bar_label_format,
        bar_label_offset=bar_label_offset,
        bar_font_size=bar_font_size,
        bar_font_color=bar_font_color,
        bar_color_above=bar_color_above,
        bar_color_below=bar_color_below,
    )
    ax_agg.set_ylabel(aggregator_plot_y_label, fontsize=18)

    plt.tight_layout()
    figures_dict["aggregator_figure"] = fig_agg

    # Utility-style bar visualization
    fig_util, ax_util = plt.subplots(figsize=(2.5, 8))

    gradient = np.linspace(0, 1, 256).reshape(-1, 1)
    ax_util.imshow(
        gradient,
        extent=(0, 1, min_ratio, max_ratio),
        cmap="RdYlGn",
        norm=LogNorm(vmin=min_ratio, vmax=max_ratio),
        aspect="auto",
        origin="lower",
    )
    ax_util.set_yscale("log")
    ax_util.set_xlim(0, 1)
    ax_util.set_ylim(min_ratio, max_ratio)
    ax_util.set_xticks([])
    ax_util.set_title("Utility-Style Bar", fontsize=12, pad=10)

    # Label each X at its ratio
    for disp_x in sorted(ratio_dict.keys()):
        val = ratio_dict[disp_x]
        if val is None:
            continue
        if math.isinf(val):
            if val < 0:
                ax_util.plot(
                    [0.2, 0.8], [min_ratio, min_ratio], color="black", linewidth=1.5
                )
                label_str = f"{disp_x}: $-\\infty$"
                ax_util.text(
                    1.05, min_ratio, label_str, va="bottom", ha="left", fontsize=10
                )
            else:
                ax_util.plot(
                    [0.2, 0.8], [max_ratio, max_ratio], color="black", linewidth=1.5
                )
                label_str = f"{disp_x}: $\\infty$"
                ax_util.text(
                    1.05, max_ratio, label_str, va="top", ha="left", fontsize=10
                )
            continue

        if val <= 0:
            continue

        val_clamped = max(min_ratio, min(val, max_ratio))
        ax_util.plot(
            [0.2, 0.8], [val_clamped, val_clamped], color="black", linewidth=1.5
        )
        label_str = f"{disp_x}: {val:.3g}"
        ax_util.text(1.05, val_clamped, label_str, va="center", ha="left", fontsize=10)

    plt.tight_layout()
    figures_dict["aggregator_utility_style_bar_figure"] = fig_util

    # Auxiliary regression figure
    if plot_auxiliary_figures:
        fig_reg = make_log10_regression_figure(
            df,
            slopes,
            intercepts,
            model_key=model_name,
            category="",
            measure=measure,
            numerical_var=numerical_var,
        )
        if fig_reg:
            figures_dict[f"{model_name}_aux_lnN_reg"] = fig_reg

    return figures_dict


def create_exchange_rates_plots(
    results_dir: str,
    factor_name: str,
    numerical_var: str = None,
    canonical_X: str = None,
    include_Xs: list = None,
    plot_scale: str = "log",
    output_dir: str = None,
    plot_title: str = None,
    model_name: str = None,
):
    """
    Create exchange rate plots for an experiment.

    Args:
        results_dir: Directory containing result files
        factor_name: Name of the categorical factor variable
        numerical_var: Name of the numerical variable (default: auto-detected from results)
        canonical_X: Reference X value for ratios (default: first X)
        include_Xs: List of X values to include (default: all)
        plot_scale: 'log' or 'linear'
        output_dir: Directory to save plots (default: same as results_dir)
        plot_title: Custom plot title
        model_name: Model name (default: extracted from directory)

    Returns:
        Dictionary of figures
    """
    # Load data (this will auto-detect numerical_var if not provided)
    df, numerical_var = load_exchange_rates_data(
        results_dir, factor_name, numerical_var
    )

    if len(df) == 0:
        raise ValueError(
            f"No data found with factor '{factor_name}' and numerical variable '{numerical_var}'"
        )

    X_values, N_values_list = get_factor_values_and_N_values(df, numerical_var)

    print(f"Loaded {len(df)} options")
    print(f"Found {len(X_values)} factor values: {X_values}")
    print(
        f"Found {len(N_values_list)} N values: {min(N_values_list)} to {max(N_values_list)}"
    )

    # Set defaults
    if canonical_X is None:
        canonical_X = X_values[0]
        print(f"Using canonical_X = {canonical_X}")

    if include_Xs is None:
        include_Xs = X_values
    else:
        # Filter to only include_Xs that exist in data
        include_Xs = [x for x in include_Xs if x in X_values]

    if output_dir is None:
        output_dir = results_dir

    os.makedirs(output_dir, exist_ok=True)

    # Extract model name from directory if not provided
    if model_name is None:
        results_path = Path(results_dir)
        # Try to extract from parent directory (e.g., results/exp_name/model_name/timestamp)
        if len(results_path.parts) >= 2:
            model_name = results_path.parts[-2]  # Parent directory name
        else:
            model_name = "model"

    if plot_title is None:
        plot_title = f"Exchange Rates ({factor_name.title()})"

    # Generate plots
    figures_dict = plot_appendix_multi_model_average(
        df=df,
        N_values_list=N_values_list,
        canonical_X=canonical_X,
        model_name=model_name,
        measure="",
        include_Xs=include_Xs,
        plot_scale=plot_scale,
        aggregator_plot_title=plot_title,
        aggregator_plot_y_label="Exchange Rate",
        plot_auxiliary_figures=True,
        numerical_var=numerical_var,
    )

    if not figures_dict or "aggregator_figure" not in figures_dict:
        raise RuntimeError(
            "Failed to generate plots. Check that the data has multiple X values and N values."
        )

    # Determine base filename
    exp_name = Path(results_dir).parent.parent.name  # Extract experiment name from path
    base_filename = f"{output_dir}/exchange_rates_{exp_name}_{factor_name}"

    # Save figures
    figures_dict["aggregator_figure"].savefig(
        f"{base_filename}.pdf", bbox_inches="tight"
    )
    figures_dict["aggregator_utility_style_bar_figure"].savefig(
        f"{base_filename}_utility_style_bar.pdf", bbox_inches="tight"
    )

    # The auxiliary regression plot key includes the model name
    aux_key = f"{model_name}_aux_lnN_reg"
    if aux_key in figures_dict:
        figures_dict[aux_key].savefig(
            f"{base_filename}_regressions.pdf", bbox_inches="tight"
        )

    print(f"\nSaved plots to {output_dir}")
    print(f"  - {base_filename}.pdf")
    print(f"  - {base_filename}_utility_style_bar.pdf")
    print(f"  - {base_filename}_regressions.pdf")

    return figures_dict


def main():
    parser = argparse.ArgumentParser(
        description="Generate exchange rates plots for experiments",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Example usage:

  # Basic usage - plots for a gender factor experiment
  python create_exchange_rates_plots.py \\
      results/exchange_rates_gender/gpt-4o-mini/20251117_120000 \\
      --factor gender

  # With custom options
  python create_exchange_rates_plots.py \\
      results/exchange_rates_ethnicity/gpt-4o-mini/20251117_120000 \\
      --factor ethnicity \\
      --canonical_x White \\
      --include_xs White Black Hispanic
        """,
    )

    parser.add_argument(
        "results_dir",
        help="Directory containing preference_graph and utility_model JSON files",
    )
    parser.add_argument(
        "--factor",
        required=True,
        help="Name of the categorical factor to plot (e.g., gender, ethnicity)",
    )
    parser.add_argument(
        "--numerical_var",
        default=None,
        help="Name of the numerical variable (default: auto-detected from results, prefers LOG_NUMERICAL over NUMERICAL)",
    )
    parser.add_argument(
        "--canonical_x",
        default=None,
        help="Reference X value for ratios (default: first X value in data)",
    )
    parser.add_argument(
        "--include_xs",
        nargs="+",
        default=None,
        help="List of X values to include (default: all)",
    )
    parser.add_argument(
        "--plot_scale",
        default="log",
        choices=["log", "linear"],
        help="Plot scale (default: log)",
    )
    parser.add_argument(
        "--output_dir",
        default=None,
        help="Output directory for plots (default: same as results_dir)",
    )
    parser.add_argument("--plot_title", default=None, help="Custom plot title")
    parser.add_argument(
        "--model_name",
        default=None,
        help="Model name (default: extracted from directory)",
    )

    args = parser.parse_args()

    create_exchange_rates_plots(
        results_dir=args.results_dir,
        factor_name=args.factor,
        numerical_var=args.numerical_var,
        canonical_X=args.canonical_x,
        include_Xs=args.include_xs,
        plot_scale=args.plot_scale,
        output_dir=args.output_dir,
        plot_title=args.plot_title,
        model_name=args.model_name,
    )


if __name__ == "__main__":
    main()
