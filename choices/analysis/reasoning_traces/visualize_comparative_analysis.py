#!/usr/bin/env python3
"""
Visualize results from comparative trace analysis.

Creates:
1. Bar chart of theme/factor frequencies by condition
2. Word cloud of key differences (from LLM analysis)
3. Word clouds of actual reasoning traces for each condition

Usage:
    uv run python -m choices.analysis.reasoning_traces.visualize_comparative_analysis \
        --input analysis_results.json

    # Custom output directory
    uv run python -m choices.analysis.reasoning_traces.visualize_comparative_analysis \
        --input analysis_results.json \
        --output-dir my_plots/
"""

import argparse
import json
import re
from collections import Counter
from pathlib import Path
from typing import List

import matplotlib.pyplot as plt
import numpy as np

from choices.analysis.utils import PLOTS_OUTPUT_DIR

# Import wordcloud
try:
    from wordcloud import WordCloud, STOPWORDS

    WORDCLOUD_AVAILABLE = True
except ImportError:
    WORDCLOUD_AVAILABLE = False
    STOPWORDS = set()
    print("Warning: wordcloud not installed. Install with: uv add wordcloud")

# Analysis-specific stopwords to add to the standard set
ANALYSIS_STOPWORDS = {
    "condition",
    "conditions",
    "option",
    "options",
    "choice",
    "choices",
    "reasoning",
    "trace",
    "traces",
    "model",
    "shows",
    "appears",
    "user",
    "asking",
}

# Combine standard stopwords with analysis-specific ones
ALL_STOPWORDS = STOPWORDS | ANALYSIS_STOPWORDS


# =============================================================================
# Data Loading
# =============================================================================


def load_analysis(filepath: str) -> dict:
    """Load analysis results from JSON file."""
    with open(filepath) as f:
        return json.load(f)


def extract_themes(cases: List[dict], condition: int = 1) -> List[str]:
    """Extract all themes from a condition across all cases."""
    key = f"condition_{condition}_themes"
    themes = []
    for case in cases:
        analysis = case.get("analysis", {})
        if key in analysis:
            themes.extend(analysis[key])
    return themes


def extract_factors(cases: List[dict], condition: int = 1) -> List[str]:
    """Extract all factors from a condition across all cases."""
    key = f"factors_condition_{condition}"
    factors = []
    for case in cases:
        analysis = case.get("analysis", {})
        if key in analysis:
            factors.extend(analysis[key])
    return factors


def extract_key_differences(cases: List[dict]) -> List[str]:
    """Extract all key differences across all cases."""
    differences = []
    for case in cases:
        analysis = case.get("analysis", {})
        if "key_differences" in analysis:
            differences.extend(analysis["key_differences"])
    return differences


def extract_reasoning_traces(cases: List[dict], condition: str = "a") -> List[str]:
    """
    Extract all reasoning traces from a condition across all cases.

    Args:
        cases: List of analyzed cases
        condition: "a" for condition_a_traces, "b" for condition_b_traces
    """
    key = f"condition_{condition}_traces"
    traces = []
    for case in cases:
        if key in case:
            for trace in case[key]:
                if "reasoning" in trace and trace["reasoning"]:
                    traces.append(trace["reasoning"])
    return traces


# =============================================================================
# Theme/Factor Bar Chart
# =============================================================================


def plot_theme_comparison(
    cases: List[dict],
    output_path: str,
    top_n: int = 15,
    use_factors: bool = False,
):
    """
    Plot bar chart comparing themes or factors between conditions (as percentages).

    Args:
        cases: List of analyzed cases
        output_path: Path to save the plot
        top_n: Number of top items to show
        use_factors: If True, use factors instead of themes
    """
    if use_factors:
        items_1 = extract_factors(cases, condition=1)
        items_2 = extract_factors(cases, condition=2)
        title = "Factors Emphasized by Condition"
        ylabel = "Factor"
    else:
        items_1 = extract_themes(cases, condition=1)
        items_2 = extract_themes(cases, condition=2)
        title = "Reasoning Themes by Condition"
        ylabel = "Theme"

    # Count frequencies
    counts_1 = Counter(items_1)
    counts_2 = Counter(items_2)

    # Get totals for percentage calculation
    total_1 = sum(counts_1.values()) if counts_1 else 1
    total_2 = sum(counts_2.values()) if counts_2 else 1

    # Get top items from combined counts
    combined = Counter(items_1 + items_2)
    top_items = [item for item, _ in combined.most_common(top_n)]

    # Prepare data for plotting (as percentages)
    x = np.arange(len(top_items))
    width = 0.35

    pcts_1 = [counts_1.get(item, 0) / total_1 * 100 for item in top_items]
    pcts_2 = [counts_2.get(item, 0) / total_2 * 100 for item in top_items]

    # Create plot
    fig, ax = plt.subplots(figsize=(12, 8))
    ax.barh(x - width / 2, pcts_1, width, label="Condition 1", color="#3498db")
    ax.barh(x + width / 2, pcts_2, width, label="Condition 2", color="#e74c3c")

    ax.set_xlabel("Percentage (%)")
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    ax.set_yticks(x)
    ax.set_yticklabels(top_items)
    ax.legend()
    ax.invert_yaxis()  # Top items at top

    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close()

    print(f"Saved theme comparison to {output_path}")


# =============================================================================
# Word Cloud Utilities
# =============================================================================


def tokenize(text: str) -> List[str]:
    """Simple tokenization with stopword removal."""
    words = re.findall(r"\b[a-z]{3,}\b", text.lower())
    return [w for w in words if w not in ALL_STOPWORDS]


def compute_tfidf_weights(documents: List[str]) -> dict:
    """
    Compute simple TF-IDF weights for words across documents.

    Args:
        documents: List of text documents

    Returns:
        Dictionary mapping words to TF-IDF scores
    """
    if not documents:
        return {}

    # Compute term frequency per document
    doc_tokens = [tokenize(doc) for doc in documents]

    # Compute document frequency
    doc_freq = Counter()
    for tokens in doc_tokens:
        doc_freq.update(set(tokens))

    n_docs = len(documents)

    # Compute TF-IDF
    tfidf = Counter()
    for tokens in doc_tokens:
        tf = Counter(tokens)
        for word, count in tf.items():
            # TF-IDF = tf * log(N / df)
            idf = np.log(n_docs / (doc_freq[word] + 1)) + 1
            tfidf[word] += count * idf

    return dict(tfidf)


def create_wordcloud(
    weights: dict,
    title: str,
    output_path: str,
    colormap: str = "viridis",
):
    """
    Create and save a word cloud from word weights.

    Args:
        weights: Dictionary mapping words to weights
        title: Title for the plot
        output_path: Path to save the plot
        colormap: Matplotlib colormap name
    """
    if not WORDCLOUD_AVAILABLE:
        print(f"Skipping word cloud ({title}): wordcloud package not installed")
        return

    if not weights:
        print(f"Skipping word cloud ({title}): no words found")
        return

    wordcloud = WordCloud(
        width=1200,
        height=600,
        background_color="white",
        colormap=colormap,
        max_words=100,
    ).generate_from_frequencies(weights)

    fig, ax = plt.subplots(figsize=(14, 7))
    ax.imshow(wordcloud, interpolation="bilinear")
    ax.axis("off")
    ax.set_title(title, fontsize=16, pad=20)

    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close()

    print(f"Saved word cloud to {output_path}")


# =============================================================================
# Word Cloud Plots
# =============================================================================


def plot_differences_wordcloud(
    cases: List[dict],
    output_path: str,
    use_tfidf: bool = True,
):
    """
    Generate word cloud from key differences (LLM analysis output).

    Args:
        cases: List of analyzed cases
        output_path: Path to save the plot
        use_tfidf: If True, weight words by TF-IDF; otherwise use raw frequency
    """
    differences = extract_key_differences(cases)

    if not differences:
        print("No key differences found in analysis")
        return

    if use_tfidf:
        weights = compute_tfidf_weights(differences)
    else:
        # Raw frequency
        all_words = []
        for diff in differences:
            all_words.extend(tokenize(diff))
        weights = dict(Counter(all_words))

    create_wordcloud(
        weights,
        "Key Differences Between Conditions",
        output_path,
        colormap="viridis",
    )


def compute_discriminative_weights(
    traces_target: List[str],
    traces_other: List[str],
    min_count: int = 3,
    smoothing: float = 1.0,
) -> dict[str, float]:
    """
    Compute discriminative word weights using log-ratio scoring.

    Words that appear more frequently in target vs other get higher scores.
    Uses smoothed relative frequencies to handle rare words.

    Args:
        traces_target: Traces from the target condition
        traces_other: Traces from the other condition
        min_count: Minimum total count for a word to be included
        smoothing: Laplace smoothing factor

    Returns:
        Dictionary mapping words to discriminative scores (positive = more in target)
    """
    # Count words in each condition
    words_target = []
    for trace in traces_target:
        words_target.extend(tokenize(trace))
    words_other = []
    for trace in traces_other:
        words_other.extend(tokenize(trace))

    counts_target = Counter(words_target)
    counts_other = Counter(words_other)

    total_target = sum(counts_target.values()) + smoothing
    total_other = sum(counts_other.values()) + smoothing

    # Compute log-ratio scores
    all_words = set(counts_target.keys()) | set(counts_other.keys())
    weights = {}

    for word in all_words:
        count_target = counts_target.get(word, 0)
        count_other = counts_other.get(word, 0)

        # Skip rare words
        if count_target + count_other < min_count:
            continue

        # Smoothed relative frequencies
        freq_target = (count_target + smoothing) / total_target
        freq_other = (count_other + smoothing) / total_other

        # Log ratio (positive means more frequent in target)
        log_ratio = np.log2(freq_target / freq_other)

        # Only keep words that are more frequent in target
        if log_ratio > 0:
            # Scale by sqrt of count to balance frequency and distinctiveness
            weights[word] = log_ratio * np.sqrt(count_target)

    return weights


def plot_traces_wordcloud(
    cases: List[dict],
    output_path: str,
    condition: str = "a",
    discriminative: bool = True,
):
    """
    Generate word cloud from actual reasoning traces.

    Args:
        cases: List of analyzed cases
        output_path: Path to save the plot
        condition: "a" or "b" for which condition's traces to use
        discriminative: If True, show words that distinguish this condition from the other
    """
    traces_target = extract_reasoning_traces(cases, condition=condition)
    other_condition = "b" if condition == "a" else "a"
    traces_other = extract_reasoning_traces(cases, condition=other_condition)

    if not traces_target:
        print(f"No traces found for condition {condition}")
        return

    if discriminative and traces_other:
        weights = compute_discriminative_weights(traces_target, traces_other)
        title = f"Distinctive Words - Condition {condition.upper()}"
    else:
        # Fallback to TF-IDF if no comparison possible
        weights = compute_tfidf_weights(traces_target)
        title = f"Reasoning Traces - Condition {condition.upper()}"

    if not weights:
        print(f"No distinctive words found for condition {condition}")
        return

    # Use different colors for each condition
    colormap = "Blues" if condition == "a" else "Reds"

    create_wordcloud(weights, title, output_path, colormap=colormap)


# =============================================================================
# CLI
# =============================================================================


def main():
    parser = argparse.ArgumentParser(
        description="Visualize comparative trace analysis results"
    )
    parser.add_argument(
        "--input",
        "-i",
        required=True,
        help="Input JSON file with analysis results",
    )
    parser.add_argument(
        "--output-dir",
        "-o",
        default=PLOTS_OUTPUT_DIR,
        help=f"Directory to save plots (default: {PLOTS_OUTPUT_DIR})",
    )
    parser.add_argument(
        "--top-n",
        type=int,
        default=15,
        help="Number of top themes/factors to show (default: 15)",
    )
    parser.add_argument(
        "--use-factors",
        action="store_true",
        help="Plot factors instead of themes",
    )
    parser.add_argument(
        "--no-tfidf",
        action="store_true",
        help="Use raw word frequency instead of TF-IDF for word clouds",
    )
    parser.add_argument(
        "--format",
        "-f",
        choices=["png", "pdf"],
        default="png",
        help="Output format for plots (default: png)",
    )

    args = parser.parse_args()

    # Load data
    print(f"Loading analysis from {args.input}...")
    data = load_analysis(args.input)
    cases = data.get("cases", [])
    print(f"Loaded {len(cases)} analyzed cases")

    # Filter to successful analyses
    cases = [c for c in cases if c.get("analysis_success", False)]
    print(f"Using {len(cases)} successful analyses")

    if not cases:
        print("No successful analyses found!")
        return

    # Create output directory
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Generate input filename stem for output naming
    input_stem = Path(args.input).stem
    fmt = args.format

    # Plot theme/factor comparison
    item_type = "factors" if args.use_factors else "themes"
    theme_path = output_dir / f"{input_stem}_{item_type}.{fmt}"
    plot_theme_comparison(
        cases,
        str(theme_path),
        top_n=args.top_n,
        use_factors=args.use_factors,
    )

    # Plot key differences word cloud
    diff_wordcloud_path = output_dir / f"{input_stem}_differences_wordcloud.{fmt}"
    plot_differences_wordcloud(
        cases,
        str(diff_wordcloud_path),
        use_tfidf=not args.no_tfidf,
    )

    # Plot condition A traces word cloud (discriminative)
    cond_a_wordcloud_path = output_dir / f"{input_stem}_condition_a_wordcloud.{fmt}"
    plot_traces_wordcloud(
        cases,
        str(cond_a_wordcloud_path),
        condition="a",
    )

    # Plot condition B traces word cloud (discriminative)
    cond_b_wordcloud_path = output_dir / f"{input_stem}_condition_b_wordcloud.{fmt}"
    plot_traces_wordcloud(
        cases,
        str(cond_b_wordcloud_path),
        condition="b",
    )

    print(f"\nDone! Plots saved to {output_dir}/")


if __name__ == "__main__":
    main()
