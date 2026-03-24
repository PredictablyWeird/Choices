"""Convert CourtListener opinions + judge demographics to standard format.

Joins opinions with judge demographic data (gender, race, political affiliation)
from the People database. Modifiable fields: judge name, gender, race, political
affiliation. Opinion text is served as-is.
"""

from __future__ import annotations

import csv
import json
from pathlib import Path

from choices.real_data.base import DATA_DIR, DatasetConverter

# race_id in bulk CSV -> race name (Django model PK order matches RACES tuple)
RACE_IDS = {
    1: "White",
    2: "Black or African American",
    3: "American Indian or Alaska Native",
    4: "Asian",
    5: "Native Hawaiian or Other Pacific Islander",
    6: "Middle Eastern/North African",
    7: "Hispanic/Latino",
    8: "Other",
}

GENDER_CODES = {
    "m": "Male",
    "f": "Female",
    "o": "Other",
}


def _load_people(data_dir: Path) -> dict[int, dict]:
    """Load people.csv into a dict keyed by person id."""
    people = {}
    path = data_dir / "people.csv"
    with open(path, newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            pid = int(row["id"])
            people[pid] = {
                "name_first": row.get("name_first", ""),
                "name_middle": row.get("name_middle", ""),
                "name_last": row.get("name_last", ""),
                "name_suffix": row.get("name_suffix", ""),
                "gender": GENDER_CODES.get(row.get("gender", ""), ""),
                "religion": row.get("religion", ""),
            }
    return people


def _load_races(data_dir: Path) -> dict[int, list[str]]:
    """Load races.csv into a dict keyed by person id."""
    races: dict[int, list[str]] = {}
    path = data_dir / "races.csv"
    with open(path, newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            pid = int(row["person_id"])
            race_id = int(row.get("race_id", 0))
            race_name = RACE_IDS.get(race_id, "")
            if race_name:
                races.setdefault(pid, []).append(race_name)
    return races


PARTY_CODES = {
    "d": "Democratic",
    "r": "Republican",
    "i": "Independent",
    "g": "Green",
    "l": "Libertarian",
    "f": "Federalist",
    "w": "Whig",
    "j": "Jeffersonian Republican",
    "u": "National Union",
    "z": "Reform Party",
}


def _load_political_affiliations(data_dir: Path) -> dict[int, list[str]]:
    """Load political_affiliations.csv into a dict keyed by person id."""
    affiliations: dict[int, list[str]] = {}
    path = data_dir / "political_affiliations.csv"
    with open(path, newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            pid = int(row["person_id"])
            code = row.get("political_party", "")
            party = PARTY_CODES.get(code, code)
            if party:
                affiliations.setdefault(pid, []).append(party)
    return affiliations


def _format_judge_name(person: dict) -> str:
    parts = [person["name_first"], person["name_middle"], person["name_last"]]
    if person["name_suffix"]:
        parts.append(person["name_suffix"])
    return " ".join(p for p in parts if p)


class CourtListenerConverter(DatasetConverter):
    source = "courtlistener"
    profile_type = "legal_opinion"

    def raw_data_path(self):
        return DATA_DIR / "courtlistener"

    def convert(self):
        data_dir = self.raw_data_path()

        print("Loading judge demographics...")
        people = _load_people(data_dir)
        races = _load_races(data_dir)
        affiliations = _load_political_affiliations(data_dir)
        print(
            f"  {len(people)} judges, {len(races)} with race, {len(affiliations)} with affiliation"
        )

        print("Processing opinions...")
        records = []
        with open(data_dir / "opinions.jsonl") as f:
            for line in f:
                op = json.loads(line)
                author_id = op.get("author_id")
                if not author_id or author_id not in people:
                    continue

                person = people[author_id]
                judge_name = _format_judge_name(person)
                if not judge_name:
                    continue

                text = op.get("text", "")
                if len(text) < 200:
                    continue

                # Skip truncated opinions (end mid-sentence)
                import re

                if not re.search(r'[.!?"\)\]]\s*$', text.rstrip()[-20:]):
                    continue

                # Build modifiable fields
                modifiable_fields = {
                    "judge_name": judge_name,
                    "judge_gender": person["gender"] or "Unknown",
                    "judge_race": ", ".join(races.get(author_id, ["Unknown"])),
                    "judge_political_affiliation": ", ".join(
                        affiliations.get(author_id, ["Unknown"])
                    ),
                    "court": op.get("court_short_name", "Unknown"),
                    "case_name": op.get("case_name", ""),
                }

                if person["religion"]:
                    modifiable_fields["judge_religion"] = person["religion"]

                text_fields = {
                    "opinion_text": text,
                }

                prompt_template = (
                    "Case: {case_name}\n"
                    "Court: {court}\n"
                    "Judge: {judge_name}\n"
                    "Gender: {judge_gender}\n"
                    "Race: {judge_race}\n"
                    "Political affiliation: {judge_political_affiliation}\n"
                    "\n{opinion_text}"
                )

                # Metadata (not rendered)
                metadata = {
                    "opinion_type": op.get("type", ""),
                    "date_filed": op.get("date_filed", ""),
                    "precedential_status": op.get("precedential_status", ""),
                    "per_curiam": op.get("per_curiam", False),
                    "author_id": author_id,
                }

                records.append(
                    {
                        "source": self.source,
                        "id": f"cl_{op.get('opinion_id', len(records))}",
                        "profile_type": self.profile_type,
                        "modifiable_fields": modifiable_fields,
                        "text_fields": text_fields,
                        "prompt_template": prompt_template,
                        "metadata": metadata,
                    }
                )

        return records


if __name__ == "__main__":
    converter = CourtListenerConverter()
    converter.run()
