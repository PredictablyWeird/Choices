"""Convert ShareChat conversations to the standard real_data format.

Each record is a user's opening message in a conversation. The text is
the primary content — there are no structured demographic fields to swap.
Modifications happen via LLM rewrite (e.g. politeness, formality).
"""

from __future__ import annotations

import json

from choices.real_data.base import DATA_DIR, DatasetConverter


class ShareChatConverter(DatasetConverter):
    source = "sharechat"
    profile_type = "chat_message"

    def raw_data_path(self):
        return DATA_DIR / "sharechat" / "conversations.jsonl"

    def convert(self):
        records = []

        with open(self.raw_data_path()) as f:
            for i, line in enumerate(f):
                row = json.loads(line)

                text = row.get("user_message", "").strip()
                if not text:
                    continue

                modifiable_fields = {
                    "platform": row.get("platform", ""),
                    "model_version": row.get("model_version", ""),
                }

                text_fields = {
                    "user_message": text,
                }

                prompt_template = "{user_message}"

                record = {
                    "source": self.source,
                    "id": f"sharechat_{i}",
                    "profile_type": self.profile_type,
                    "modifiable_fields": modifiable_fields,
                    "text_fields": text_fields,
                    "prompt_template": prompt_template,
                }

                # Store model reply as metadata if available
                reply = row.get("model_reply", "")
                if reply:
                    record["metadata"] = {
                        "model_reply": reply,
                        "turns_count": row.get("turns_count", 0),
                        "url": row.get("url", ""),
                    }

                records.append(record)

        return records


if __name__ == "__main__":
    converter = ShareChatConverter()
    converter.run()
