"""JSON-backed person registry for cross-dataset identity linking.

Stores persistent person identities that span multiple recording sessions.
The registry file (``person_registry.json``) lives at the data root directory.

Example::

    registry = PersonRegistry.load("/data/TLV")
    registry.add_person("chetan", "Chetan")
    registry.register_dataset("chetan", "chetan_30_min_1")
    registry.save("/data/TLV")
"""

import json
import os
from dataclasses import dataclass, field
from typing import Dict, List

from loguru import logger

REGISTRY_FILENAME = "person_registry.json"


@dataclass
class PersonRecord:
    """A registered person with cross-dataset identity."""

    person_id: str
    display_name: str
    datasets: List[str] = field(default_factory=list)
    notes: str = ""


class PersonRegistry:
    """Registry of known persons across datasets.

    Backed by a JSON file at the data root. Provides bidirectional
    lookups: person -> datasets and dataset -> persons.
    """

    def __init__(self) -> None:
        self._persons: Dict[str, PersonRecord] = {}

    @staticmethod
    def load(base_dir: str) -> "PersonRegistry":
        """Load registry from ``base_dir/person_registry.json``.

        Returns an empty registry if the file does not exist.
        """
        registry = PersonRegistry()
        path = os.path.join(base_dir, REGISTRY_FILENAME)
        if not os.path.isfile(path):
            return registry

        try:
            with open(path, "r", encoding="utf-8") as f:
                data = json.load(f)
        except (json.JSONDecodeError, OSError) as e:
            logger.warning(f"Failed to load person registry: {e}")
            return registry

        for pid, info in data.get("persons", {}).items():
            registry._persons[pid] = PersonRecord(
                person_id=pid,
                display_name=info.get("display_name", pid),
                datasets=info.get("datasets", []),
                notes=info.get("notes", ""),
            )

        return registry

    def save(self, base_dir: str) -> None:
        """Save registry to ``base_dir/person_registry.json``."""
        data = {"version": 1, "persons": {}}
        for pid, record in sorted(self._persons.items()):
            data["persons"][pid] = {
                "display_name": record.display_name,
                "datasets": sorted(set(record.datasets)),
                "notes": record.notes,
            }

        path = os.path.join(base_dir, REGISTRY_FILENAME)
        with open(path, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2, ensure_ascii=False)

    def add_person(self, person_id: str, display_name: str) -> PersonRecord:
        """Add a new person or update display name if already exists."""
        if person_id in self._persons:
            self._persons[person_id].display_name = display_name
        else:
            self._persons[person_id] = PersonRecord(
                person_id=person_id, display_name=display_name
            )
        return self._persons[person_id]

    def register_dataset(self, person_id: str, dataset_rel_path: str) -> None:
        """Link a dataset to a person. Creates the person if needed."""
        if person_id not in self._persons:
            self.add_person(person_id, person_id)
        record = self._persons[person_id]
        if dataset_rel_path not in record.datasets:
            record.datasets.append(dataset_rel_path)

    def get_datasets_for_person(self, person_id: str) -> List[str]:
        """Return dataset paths linked to a person."""
        record = self._persons.get(person_id)
        return list(record.datasets) if record else []

    def get_persons_in_dataset(self, dataset_path: str) -> List[str]:
        """Return person IDs that appear in a given dataset."""
        return [
            pid
            for pid, record in self._persons.items()
            if dataset_path in record.datasets
        ]

    def list_persons(self) -> List[PersonRecord]:
        """Return all registered persons."""
        return list(self._persons.values())
