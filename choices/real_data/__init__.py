"""
real_data — Real-world datasets with swappable demographic fields for bias testing.

Two categories of dataset:

1. FORMAT-STRING DATASETS (okcupid, resumes)
   Demographics live in structured fields, separate from text. Modifications are
   simple format-string substitutions at serve time. No LLM needed.

2. LLM-REWRITE DATASETS (aita)
   Demographics are embedded in narrative text. Producing counterfactual variants
   requires an LLM to rewrite the text with minimal changes.

================================================================================
CATEGORY 1: Format-string datasets
================================================================================

Setup:
    python -m choices.real_data.okcupid.download
    python -m choices.real_data.okcupid.convert

Usage:
    from choices.real_data.base import serve, load_dataset

    records = load_dataset("okcupid")
    print(serve("okcupid", index=42))
    print(serve("okcupid", {"sex": "f", "age": "30"}, index=42))

================================================================================
CATEGORY 2: LLM-rewrite datasets
================================================================================

Setup:
    python -m choices.real_data.aita.download
    python -m choices.real_data.aita.convert

    # Then run the LLM rewrite (costs money — calls OpenAI API):
    python -m choices.real_data.aita.rewrite_gender --model gpt-5.2 --reasoning low --n 100

Usage:
    from choices.real_data.llm_rewrite import load_rewritten
    pairs = load_rewritten("aita", "gender", "gpt-5.2")

================================================================================
Standard record format (both categories)
================================================================================

Each converted record is a dict with:
    source            dataset identifier (e.g. "okcupid", "aita")
    id                unique record id
    profile_type      category (e.g. "dating", "resume", "social_post")
    modifiable_fields dict of swappable demographics (category 1) or metadata (category 2)
    text_fields       dict of free-text content
    prompt_template   renderable string with {field} placeholders

================================================================================
Adding a new dataset
================================================================================

For EITHER category:
1. Create choices/real_data/<name>/__init__.py
2. Create choices/real_data/<name>/download.py   — fetch raw data to data/<name>/
3. Create choices/real_data/<name>/convert.py    — subclass DatasetConverter from base.py
4. Add choices/real_data/<name>/LICENSE.txt       — license or explanation if none

For format-string datasets (category 1):
    - convert.py should produce records with modifiable_fields containing the
      demographic values and a prompt_template with {field} placeholders.
    - See okcupid/convert.py for a dataset with built-in demographics.
    - See resumes/convert.py for a dataset where demographics are injected.

For LLM-rewrite datasets (category 2):
    - convert.py stores raw text + any extractable metadata.
    - convert.py stores raw text + any extractable metadata.
    - Create a preset YAML in presets/ with the instruction, variants, and
      filter_pattern. Run via: python -m choices.real_data.llm_rewrite
    - The llm_rewrite.py module handles all LLM calling, retries, and output.

================================================================================
Available datasets
================================================================================

Format-string:
    okcupid   57k+ dating profiles. Fields: age, sex, ethnicity, orientation,
              religion, education, job, body_type, diet, drinks, drugs, etc.
              + 10 essay fields. (License: attribution required, see LICENSE.txt)

    resumes   2,483 resumes across 24 job categories. No demographics in source —
              gender, age, ethnicity, nationality injected as placeholders.
              (License: CC0 Public Domain)

    peerread  12.3k scientific papers from ACL, NIPS, ICLR, ICML, AAAI, arxiv.
              Authors and conference are modifiable. Full paper sections when
              available. Acceptance decisions and peer reviews stored in metadata.
              (License: not specified, see LICENSE.txt)

LLM-rewrite:
    aita      97k+ Reddit AITA posts with verdicts. Demographics embedded in
              narrative text. Use llm_rewrite.py with presets to produce
              counterfactual pairs. (License: not specified, see LICENSE.txt)
"""
