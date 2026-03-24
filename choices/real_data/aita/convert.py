"""Convert AITA posts to the standard real_data format.

Stores raw posts as-is. Demographic modification requires an LLM rewrite step
(see real_data/llm_rewrite.py). Filtering (e.g. "only posts with gender tags")
is handled by the preset's filter_pattern, not here.
"""

from __future__ import annotations

import pandas as pd

from choices.real_data.base import DATA_DIR, DatasetConverter


class AITAConverter(DatasetConverter):
    source = "aita"
    profile_type = "social_post"

    def raw_data_path(self):
        return DATA_DIR / "aita" / "aita_clean.csv"

    def convert(self):
        df = pd.read_csv(self.raw_data_path())
        records = []

        for idx, row in df.iterrows():
            body = str(row["body"]) if pd.notna(row["body"]) else ""
            title = str(row["title"])
            if not body.strip():
                continue

            modifiable_fields = {
                "verdict": row["verdict"],
                "score": int(row["score"]),
            }

            text_fields = {
                "title": title,
                "body": body,
            }

            prompt_template = "{title}\n\n{body}"

            records.append(
                {
                    "source": self.source,
                    "id": f"aita_{row['id']}",
                    "profile_type": self.profile_type,
                    "modifiable_fields": modifiable_fields,
                    "text_fields": text_fields,
                    "prompt_template": prompt_template,
                }
            )

        return records


if __name__ == "__main__":
    converter = AITAConverter()
    converter.run()
