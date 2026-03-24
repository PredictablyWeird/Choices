"""Convert PeerRead to the standard real_data format.

Serves papers with author names/affiliations as modifiable fields.
Reviews and acceptance decisions are stored as metadata but not rendered
in the prompt template.
"""

from __future__ import annotations

import json

from choices.real_data.base import DATA_DIR, DatasetConverter


class PeerReadConverter(DatasetConverter):
    source = "peerread"
    profile_type = "paper"

    def raw_data_path(self):
        return DATA_DIR / "peerread"

    def convert(self):
        reviews_path = self.raw_data_path() / "reviews.jsonl"
        papers_path = self.raw_data_path() / "parsed_pdfs.jsonl"

        # Load parsed PDFs keyed by title for joining
        paper_by_title: dict[str, dict] = {}
        with open(papers_path) as f:
            for line in f:
                p = json.loads(line)
                meta = p.get("metadata", {})
                title = (meta.get("title") or "").strip()
                if title:
                    paper_by_title[title.lower()] = meta

        # Build records from reviews (has authors, conference, accepted)
        records = []
        with open(reviews_path) as f:
            for line in f:
                r = json.loads(line)

                title = (r.get("title") or "").strip()
                authors = r.get("authors", [])
                abstract = (r.get("abstract") or "").strip()
                conference = r.get("conference")
                accepted = r.get("accepted")

                if not title or not abstract:
                    continue

                # Try to get full paper sections from parsed_pdfs
                paper_meta = paper_by_title.get(title.lower(), {})
                sections = paper_meta.get("sections") or []

                # Build section text
                section_texts = {}
                for i, sec in enumerate(sections):
                    heading = sec.get("heading", f"Section {i + 1}")
                    text = sec.get("text", "")
                    if text.strip():
                        section_texts[f"section_{i}"] = f"{heading}\n{text}"

                # Modifiable: authors (for bias testing on author identity)
                modifiable_fields = {
                    "authors": ", ".join(authors) if authors else "Anonymous",
                    "conference": conference or "Unknown",
                }

                # Metadata: not rendered, but available for analysis
                metadata = {}
                if accepted is not None:
                    metadata["accepted"] = accepted
                review_data = r.get("reviews", [])
                if review_data:
                    metadata["reviews"] = review_data

                text_fields = {
                    "title": title,
                    "abstract": abstract,
                }
                # Add sections if available
                text_fields.update(section_texts)

                # Build template: authors + paper content
                template_parts = [
                    "Authors: {authors}",
                    "Conference: {conference}",
                    "",
                    "{title}",
                    "",
                    "Abstract:",
                    "{abstract}",
                ]
                for key in section_texts:
                    template_parts.append("")
                    template_parts.append("{" + key + "}")

                record = {
                    "source": self.source,
                    "id": f"peerread_{r.get('id', len(records))}",
                    "profile_type": self.profile_type,
                    "modifiable_fields": modifiable_fields,
                    "text_fields": text_fields,
                    "prompt_template": "\n".join(template_parts),
                }
                if metadata:
                    record["metadata"] = metadata

                records.append(record)

        return records


if __name__ == "__main__":
    converter = PeerReadConverter()
    converter.run()
