"""Streamlit viewer for real_data processed datasets.

Run with:
    streamlit run choices/real_data/viewer.py
"""

from __future__ import annotations

import difflib
import json
import re
from pathlib import Path

import streamlit as st

CONVERTED_DIR = Path(__file__).parent / "converted"
REWRITTEN_DIR = CONVERTED_DIR / "rewritten"


@st.cache_data
def load_dataset(source: str) -> list[dict]:
    path = CONVERTED_DIR / f"{source}.jsonl"
    records = []
    with open(path) as f:
        for line in f:
            records.append(json.loads(line))
    return records


@st.cache_data
def load_rewritten(filename: str) -> list[dict]:
    path = REWRITTEN_DIR / filename
    records = []
    with open(path) as f:
        for line in f:
            records.append(json.loads(line))
    return records


@st.cache_data
def get_available_datasets() -> list[str]:
    return sorted(p.stem for p in CONVERTED_DIR.glob("*.jsonl"))


@st.cache_data
def get_available_rewritten() -> list[str]:
    if not REWRITTEN_DIR.exists():
        return []
    return sorted(p.name for p in REWRITTEN_DIR.glob("*.jsonl"))


def _display_original(records: list[dict]):
    """Display UI for original field-based datasets."""
    # --- Discover filterable fields ---
    all_mod_keys: dict[str, set[str]] = {}
    for rec in records:
        for k, v in rec.get("modifiable_fields", {}).items():
            if v:
                all_mod_keys.setdefault(k, set()).add(str(v))

    filterable = {k: sorted(v) for k, v in all_mod_keys.items() if 1 < len(v) <= 200}

    # --- Sidebar filters ---
    st.sidebar.markdown("---")
    st.sidebar.subheader("Filters")

    active_filters: dict[str, list[str]] = {}
    for field, options in sorted(filterable.items()):
        chosen = st.sidebar.multiselect(
            field.replace("_", " ").title(),
            options,
            key=f"filter_{field}",
        )
        if chosen:
            active_filters[field] = chosen

    filtered = records
    for field, allowed in active_filters.items():
        filtered = [
            r
            for r in filtered
            if str(r.get("modifiable_fields", {}).get(field, "")) in allowed
        ]

    st.sidebar.markdown(f"**{len(filtered):,}** records after filtering")

    if not filtered:
        st.warning("No records match the current filters.")
        return

    # --- Record navigation ---
    col1, col2 = st.columns([3, 1])
    with col1:
        idx = st.number_input(
            "Record index", min_value=0, max_value=len(filtered) - 1, value=0, step=1
        )
    with col2:
        st.markdown(f"**{idx + 1}** / {len(filtered):,}")

    record = filtered[idx]

    # --- Display ---
    tab_rendered, tab_redacted, tab_fields, tab_text, tab_raw = st.tabs(
        ["Rendered", "Redacted", "Modifiable Fields", "Text Fields", "Raw JSON"]
    )

    with tab_rendered:
        fields = {
            **record.get("modifiable_fields", {}),
            **record.get("text_fields", {}),
        }
        try:
            rendered = record["prompt_template"].format_map(fields)
            st.text(rendered)
        except KeyError as e:
            st.error(f"Missing field in template: {e}")
            st.text(record["prompt_template"])

    with tab_redacted:
        mod_keys = record.get("modifiable_fields", {}).keys()
        redacted_fields = {
            **{k: "MODIFIABLE" for k in mod_keys},
            **record.get("text_fields", {}),
        }
        try:
            rendered = record["prompt_template"].format_map(redacted_fields)
            st.text(rendered)
        except KeyError as e:
            st.error(f"Missing field in template: {e}")
            st.text(record["prompt_template"])

    with tab_fields:
        mod = record.get("modifiable_fields", {})
        if mod:
            for k, v in mod.items():
                st.markdown(f"**{k}**: {v}")
        else:
            st.info("No modifiable fields.")

    with tab_text:
        text = record.get("text_fields", {})
        if text:
            for k, v in text.items():
                with st.expander(k, expanded=True):
                    st.write(v)
        else:
            st.info("No text fields.")

    with tab_raw:
        st.json(record)


def _tokenize(text: str) -> list[str]:
    """Split text into words and whitespace/punctuation tokens."""
    return re.findall(r"\S+|\n+|[ \t]+", text)


def _word_diff_html(text_a: str, text_b: str, label_a: str, label_b: str) -> str:
    """Produce word-level inline diff HTML between two texts.

    Deletions (from A) are shown in red, insertions (from B) in green.
    """
    tokens_a = _tokenize(text_a)
    tokens_b = _tokenize(text_b)
    sm = difflib.SequenceMatcher(None, tokens_a, tokens_b, autojunk=False)

    parts: list[str] = []
    for op, i1, i2, j1, j2 in sm.get_opcodes():
        if op == "equal":
            parts.append(_esc("".join(tokens_a[i1:i2])))
        elif op == "replace":
            parts.append(
                f'<span style="background:#fbb;text-decoration:line-through" '
                f'title="{label_a}">{"".join(_esc(t) for t in tokens_a[i1:i2])}</span>'
            )
            parts.append(
                f'<span style="background:#bfb" '
                f'title="{label_b}">{"".join(_esc(t) for t in tokens_b[j1:j2])}</span>'
            )
        elif op == "delete":
            parts.append(
                f'<span style="background:#fbb;text-decoration:line-through" '
                f'title="{label_a}">{"".join(_esc(t) for t in tokens_a[i1:i2])}</span>'
            )
        elif op == "insert":
            parts.append(
                f'<span style="background:#bfb" '
                f'title="{label_b}">{"".join(_esc(t) for t in tokens_b[j1:j2])}</span>'
            )
    return "".join(parts)


def _esc(text: str) -> str:
    """Escape HTML, preserving newlines as <br>."""
    return (
        text.replace("&", "&amp;")
        .replace("<", "&lt;")
        .replace(">", "&gt;")
        .replace("\n", "<br>")
    )


@st.cache_data
def _build_original_index(source: str) -> dict[str, str]:
    """Build a mapping from record ID to rendered original text."""
    try:
        originals = load_dataset(source)
    except FileNotFoundError:
        return {}
    index = {}
    for rec in originals:
        fields = {**rec.get("modifiable_fields", {}), **rec.get("text_fields", {})}
        try:
            index[rec["id"]] = rec["prompt_template"].format_map(fields)
        except KeyError:
            pass
    return index


def _resolve_variants(record: dict, original_index: dict[str, str]) -> dict[str, str]:
    """Return all variants with nulls filled from the original source text."""
    variants = record.get("variants", {})
    original_text = original_index.get(record.get("id", ""))
    resolved = {}
    for name, text in variants.items():
        if text is not None:
            resolved[name] = text
        elif original_text:
            resolved[name] = original_text
    return resolved


def _display_rewritten(records: list[dict]):
    """Display UI for rewritten variant-based datasets."""
    # Discover all variant names across records
    all_variants: set[str] = set()
    for rec in records:
        all_variants.update(rec.get("variants", {}).keys())
    variant_names = sorted(all_variants)

    # Load originals for filling null variants
    source = records[0].get("source", "") if records else ""
    original_index = _build_original_index(source)

    # --- Sidebar filters ---
    st.sidebar.markdown("---")
    st.sidebar.subheader("Filters")

    # Filter by which variants were rewritten (non-null)
    show_only = st.sidebar.multiselect(
        "Has rewritten variant",
        variant_names,
        key="rewritten_has_variant",
        help="Only show records where these variants were rewritten (not original)",
    )

    filtered = records
    if show_only:
        filtered = [
            r
            for r in filtered
            if all(r.get("variants", {}).get(v) is not None for v in show_only)
        ]

    st.sidebar.markdown(f"**{len(filtered):,}** records after filtering")

    if not filtered:
        st.warning("No records match the current filters.")
        return

    # --- Record navigation ---
    col1, col2 = st.columns([3, 1])
    with col1:
        idx = st.number_input(
            "Record index", min_value=0, max_value=len(filtered) - 1, value=0, step=1
        )
    with col2:
        st.markdown(f"**{idx + 1}** / {len(filtered):,}")

    record = filtered[idx]
    raw_variants = record.get("variants", {})
    resolved = _resolve_variants(record, original_index)

    st.caption(
        f"**ID:** {record.get('id', '?')}  |  "
        f"**Source:** {record.get('source', '?')}  |  "
        f"**Rewrite:** {record.get('rewrite_name', '?')}  |  "
        f"**Model:** {record.get('model', '?')}"
    )

    # --- Variant tabs (all variants, marking which are original) ---
    tab_names = [*variant_names, "Diff", "Side by Side", "Raw JSON"]
    tabs = st.tabs(tab_names)

    for i, variant_name in enumerate(variant_names):
        with tabs[i]:
            is_original = raw_variants.get(variant_name) is None
            if is_original:
                st.caption("*(original — not rewritten)*")
            text = resolved.get(variant_name)
            if text:
                st.text(text)
            else:
                st.warning("No text available (original record not found).")

    # Diff tab
    with tabs[len(variant_names)]:
        texts = [(v, resolved.get(v, "")) for v in variant_names if resolved.get(v)]
        if len(texts) >= 2:
            # Diff first variant against second
            label_a, text_a = texts[0]
            label_b, text_b = texts[1]
            st.markdown(
                f"<span style='background:#fbb;text-decoration:line-through'>"
                f"&nbsp;{_esc(label_a)}&nbsp;</span> → "
                f"<span style='background:#bfb'>&nbsp;{_esc(label_b)}&nbsp;</span>",
                unsafe_allow_html=True,
            )
            diff_html = _word_diff_html(text_a, text_b, label_a, label_b)
            st.markdown(
                f"<pre style='white-space:pre-wrap;font-size:14px;line-height:1.5'>{diff_html}</pre>",
                unsafe_allow_html=True,
            )
        else:
            st.info("Need at least two variants to show a diff.")

    # Side by side tab
    with tabs[len(variant_names) + 1]:
        if resolved:
            cols = st.columns(len(variant_names))
            for col, variant_name in zip(cols, variant_names):
                with col:
                    is_original = raw_variants.get(variant_name) is None
                    label = (
                        f"{variant_name} *(original)*" if is_original else variant_name
                    )
                    st.subheader(label)
                    text = resolved.get(variant_name, "")
                    st.text(text)
        else:
            st.warning("No variants available for this record.")

    # Raw JSON tab
    with tabs[-1]:
        st.json(record)


def main():
    st.set_page_config(page_title="Real Data Viewer", layout="wide")
    st.title("Real Data Viewer")

    original_datasets = get_available_datasets()
    rewritten_datasets = get_available_rewritten()

    if not original_datasets and not rewritten_datasets:
        st.error("No datasets found. Run the conversion scripts first.")
        return

    # --- Top-level mode selector ---
    mode_options = []
    if original_datasets:
        mode_options.append("Original")
    if rewritten_datasets:
        mode_options.append("Rewritten")

    mode = st.sidebar.radio("Dataset type", mode_options)

    if mode == "Original":
        dataset_name = st.sidebar.selectbox("Dataset", original_datasets)
        records = load_dataset(dataset_name)
        st.sidebar.markdown(f"**{len(records):,}** records in `{dataset_name}`")
        _display_original(records)
    else:
        # Show rewritten dataset picker (strip .jsonl for display)
        display_names = [f.replace(".jsonl", "") for f in rewritten_datasets]
        chosen_idx = st.sidebar.selectbox(
            "Dataset",
            range(len(rewritten_datasets)),
            format_func=lambda i: display_names[i],
        )
        filename = rewritten_datasets[chosen_idx]
        records = load_rewritten(filename)
        st.sidebar.markdown(
            f"**{len(records):,}** records in `{display_names[chosen_idx]}`"
        )
        _display_rewritten(records)


if __name__ == "__main__":
    main()
