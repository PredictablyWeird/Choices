"""Convert the snehaanbhawal resume dataset to the standard real_data format.

The raw resumes have no demographic fields — only resume text and job category.
We add placeholder modifiable fields (gender, age, ethnicity, nationality) that
can be swapped in at serve time to test for hiring bias.
"""

from __future__ import annotations

import re

import pandas as pd

from choices.real_data.base import DATA_DIR, DatasetConverter

# Demographic fields that can be prepended to the resume for bias testing.
# These are not present in the original data — they are injected as modifiable
# placeholders with empty defaults.
MODIFIABLE_FIELDS = [
    "gender",
    "age",
    "ethnicity",
    "nationality",
]


def clean_resume_text(text: str) -> str:
    """Clean up whitespace artifacts from the scraped resume text."""
    if not isinstance(text, str):
        return ""
    # Collapse excessive whitespace
    text = re.sub(r"[ \t]{2,}", "  ", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()


def build_prompt_template(has_demographics: bool = True) -> str:
    """Build a prompt template for a resume.

    Demographics are optional header fields prepended before the resume body.
    """
    lines = []
    if has_demographics:
        lines += [
            "Candidate Profile",
            "=" * 40,
            "Gender: {gender}",
            "Age: {age}",
            "Ethnicity: {ethnicity}",
            "Nationality: {nationality}",
            "Job Category: {category}",
            "",
            "Resume",
            "-" * 40,
        ]
    else:
        lines += [
            "Job Category: {category}",
            "",
            "Resume",
            "-" * 40,
        ]
    lines.append("{resume}")
    return "\n".join(lines)


class ResumeConverter(DatasetConverter):
    source = "resumes"
    profile_type = "resume"

    def raw_data_path(self):
        return DATA_DIR / "resumes" / "Resume.csv"

    def convert(self):
        df = pd.read_csv(self.raw_data_path())
        records = []
        template = build_prompt_template(has_demographics=True)

        for idx, row in df.iterrows():
            resume_text = clean_resume_text(row["Resume_str"])
            if not resume_text:
                continue

            category = row["Category"].strip()

            # Demographics default to empty — caller fills them in via serve()
            modifiable = {
                "gender": "",
                "age": "",
                "ethnicity": "",
                "nationality": "",
                "category": category,
            }

            text_fields = {
                "resume": resume_text,
            }

            records.append(
                {
                    "source": self.source,
                    "id": f"resume_{row['ID']}",
                    "profile_type": self.profile_type,
                    "modifiable_fields": modifiable,
                    "text_fields": text_fields,
                    "prompt_template": template,
                }
            )

        return records


if __name__ == "__main__":
    converter = ResumeConverter()
    converter.run()
