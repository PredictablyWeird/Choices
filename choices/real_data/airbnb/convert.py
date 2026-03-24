"""Convert Inside Airbnb listings to the standard real_data format.

Modifiable fields: host_name, host_location, neighbourhood.
Text fields: listing name, description, host_about, neighborhood_overview.
Property details (room_type, price, beds, etc.) included as context.
"""

from __future__ import annotations

import pandas as pd

from choices.real_data.base import DATA_DIR, DatasetConverter


class AirbnbConverter(DatasetConverter):
    source = "airbnb"
    profile_type = "listing"

    def raw_data_path(self):
        # The detailed file with 74 columns
        return DATA_DIR / "airbnb" / "listings 2.csv"

    def convert(self):
        df = pd.read_csv(self.raw_data_path(), low_memory=False)
        records = []

        for idx, row in df.iterrows():
            name = str(row["name"]) if pd.notna(row["name"]) else ""
            description = (
                str(row["description"]) if pd.notna(row["description"]) else ""
            )
            if not name and not description:
                continue

            # Modifiable: host identity + location
            modifiable_fields = {
                "host_name": str(row["host_name"])
                if pd.notna(row["host_name"])
                else "Anonymous",
                "host_location": str(row["host_location"])
                if pd.notna(row["host_location"])
                else "",
                "neighbourhood": str(row["neighbourhood_cleansed"]),
            }

            # Property details as modifiable too (for context manipulation)
            price = str(row["price"]) if pd.notna(row["price"]) else ""
            modifiable_fields["property_type"] = str(row["property_type"])
            modifiable_fields["room_type"] = str(row["room_type"])
            modifiable_fields["price"] = price
            modifiable_fields["accommodates"] = str(int(row["accommodates"]))

            beds = f"{int(row['bedrooms'])} bed" if pd.notna(row["bedrooms"]) else ""
            if beds and int(row["bedrooms"]) != 1:
                beds += "rooms"
            else:
                beds += "room"
            baths = (
                str(row["bathrooms_text"]) if pd.notna(row["bathrooms_text"]) else ""
            )
            modifiable_fields["bedrooms"] = beds
            modifiable_fields["bathrooms"] = baths

            # Text fields
            text_fields = {
                "name": name,
                "description": description,
            }
            host_about = str(row["host_about"]) if pd.notna(row["host_about"]) else ""
            if host_about:
                text_fields["host_about"] = host_about
            neighborhood_overview = (
                str(row["neighborhood_overview"])
                if pd.notna(row["neighborhood_overview"])
                else ""
            )
            if neighborhood_overview:
                text_fields["neighborhood_overview"] = neighborhood_overview

            # Build template
            template_parts = [
                "Host: {host_name}",
                "Location: {host_location}",
                "Neighbourhood: {neighbourhood}",
                "",
                "{name}",
                "",
                "Property: {property_type} · {room_type}",
                "Price: {price} · Accommodates: {accommodates}",
                "{bedrooms} · {bathrooms}",
            ]
            if description:
                template_parts += ["", "{description}"]
            if host_about:
                template_parts += ["", "About the host:", "{host_about}"]
            if neighborhood_overview:
                template_parts += [
                    "",
                    "Neighbourhood overview:",
                    "{neighborhood_overview}",
                ]

            # Metadata (not rendered)
            metadata = {}
            if pd.notna(row.get("review_scores_rating")):
                metadata["review_scores_rating"] = float(row["review_scores_rating"])
            if pd.notna(row.get("number_of_reviews")):
                metadata["number_of_reviews"] = int(row["number_of_reviews"])
            if pd.notna(row.get("host_is_superhost")):
                metadata["host_is_superhost"] = str(row["host_is_superhost"]) == "t"

            record = {
                "source": self.source,
                "id": f"airbnb_{row['id']}",
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
    converter = AirbnbConverter()
    converter.run()
