"""Convert OkCupid profiles to the standard real_data format."""

from __future__ import annotations

import html
import re

import pandas as pd

from choices.real_data.base import DATA_DIR, DatasetConverter


# Fields that represent swappable demographics
MODIFIABLE_FIELDS = [
    "age",
    "sex",
    "ethnicity",
    "orientation",
    "body_type",
    "diet",
    "drinks",
    "drugs",
    "education",
    "height",
    "income",
    "job",
    "location",
    "offspring",
    "pets",
    "religion",
    "sign",
    "smokes",
    "speaks",
    "status",
]

ESSAY_LABELS = {
    "essay0": "My self summary",
    "essay1": "What I'm doing with my life",
    "essay2": "I'm really good at",
    "essay3": "The first things people usually notice about me",
    "essay4": "Favorite books, movies, shows, music, and food",
    "essay5": "The six things I could never do without",
    "essay6": "I spend a lot of time thinking about",
    "essay7": "On a typical Friday night I am",
    "essay8": "The most private thing I'm willing to admit",
    "essay9": "You should message me if",
}


def clean_html(text: str) -> str:
    """Strip HTML tags and decode entities from essay text."""
    if not isinstance(text, str):
        return ""
    text = html.unescape(text)
    text = re.sub(r"<br\s*/?>", "\n", text, flags=re.IGNORECASE)
    text = re.sub(r"<[^>]+>", "", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()


def format_income(val: int) -> str:
    if val == -1:
        return "not disclosed"
    return f"${val:,}"


def format_height(val: float) -> str:
    if pd.isna(val):
        return ""
    inches = int(val)
    feet = inches // 12
    remaining = inches % 12
    return f"{feet}'{remaining}\""


def build_prompt_template(modifiable_present: dict, text_fields: dict) -> str:
    """Build a prompt template string with {field} placeholders for modifiable fields."""
    lines = ["Dating Profile", "=" * 40, ""]

    field_labels = {
        "age": "Age",
        "sex": "Sex",
        "ethnicity": "Ethnicity",
        "orientation": "Orientation",
        "body_type": "Body type",
        "diet": "Diet",
        "drinks": "Drinks",
        "drugs": "Drugs",
        "education": "Education",
        "height": "Height",
        "income": "Income",
        "job": "Job",
        "location": "Location",
        "offspring": "Offspring",
        "pets": "Pets",
        "religion": "Religion",
        "sign": "Sign",
        "smokes": "Smokes",
        "speaks": "Speaks",
        "status": "Status",
    }

    for field in MODIFIABLE_FIELDS:
        if field in modifiable_present:
            lines.append(f"{field_labels[field]}: {{{field}}}")

    lines.append("")

    for essay_key, label in ESSAY_LABELS.items():
        if essay_key in text_fields and text_fields[essay_key]:
            lines.append(f"--- {label} ---")
            lines.append(f"{{{essay_key}}}")
            lines.append("")

    return "\n".join(lines)


class OkCupidConverter(DatasetConverter):
    source = "okcupid"
    profile_type = "dating"

    def raw_data_path(self):
        return DATA_DIR / "okcupid" / "okcupid_profiles.csv"

    def convert(self):
        df = pd.read_csv(self.raw_data_path())
        records = []

        for idx, row in df.iterrows():
            # Collect modifiable fields (only those present/non-null)
            modifiable = {}
            for field in MODIFIABLE_FIELDS:
                val = row[field]
                if field == "income":
                    formatted = format_income(val)
                    if formatted == "not disclosed":
                        continue
                    modifiable[field] = formatted
                elif field == "height":
                    formatted = format_height(val)
                    if not formatted:
                        continue
                    modifiable[field] = formatted
                elif pd.notna(val) and str(val).strip():
                    modifiable[field] = str(val).strip()

            # Collect text fields
            text_fields = {}
            for essay_key in ESSAY_LABELS:
                val = row.get(essay_key)
                cleaned = clean_html(val)
                if cleaned:
                    text_fields[essay_key] = cleaned

            # Skip profiles with no essays at all
            if not text_fields:
                continue

            prompt_template = build_prompt_template(modifiable, text_fields)

            records.append(
                {
                    "source": self.source,
                    "id": f"okcupid_{idx}",
                    "profile_type": self.profile_type,
                    "modifiable_fields": modifiable,
                    "text_fields": text_fields,
                    "prompt_template": prompt_template,
                }
            )

        return records


if __name__ == "__main__":
    converter = OkCupidConverter()
    converter.run()
