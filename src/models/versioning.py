"""
Model Versioning — minimal implementation sufficient for pipeline execution.

Tracks model versions by saving metadata alongside each training run.
"""

import os
import json
from datetime import datetime


class ModelVersion:

    def __init__(self, model_dir="data/models"):
        self.model_dir = model_dir
        self.versions_dir = os.path.join(model_dir, "versions")
        os.makedirs(self.versions_dir, exist_ok=True)

    def save_version(self, pipeline, label_source, metrics, notes=""):
        """Save a model version record with metadata."""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        existing = self.list_versions()
        version_num = len(existing) + 1
        version_id = f"v{version_num}_{label_source}_{timestamp}"

        version_dir = os.path.join(self.versions_dir, version_id)
        os.makedirs(version_dir, exist_ok=True)

        metadata = {
            "version_id": version_id,
            "version_num": version_num,
            "label_source": label_source,
            "training_date": timestamp,
            "metrics": metrics,
            "notes": notes,
        }
        with open(os.path.join(version_dir, "metadata.json"), "w") as f:
            json.dump(metadata, f, indent=2, default=str)

        return version_id

    def list_versions(self):
        """List all saved model versions with summary metrics."""
        versions = []
        if not os.path.isdir(self.versions_dir):
            return versions
        for entry in sorted(os.listdir(self.versions_dir)):
            meta_path = os.path.join(self.versions_dir, entry, "metadata.json")
            if os.path.isfile(meta_path):
                with open(meta_path) as f:
                    versions.append(json.load(f))
        return versions

    def compare_versions(self, version_ids):
        """Compare metrics across model versions."""
        all_versions = {v["version_id"]: v for v in self.list_versions()}
        return [all_versions[vid] for vid in version_ids if vid in all_versions]

    def load_version(self, version_id):
        """Load metadata for a specific model version."""
        meta_path = os.path.join(self.versions_dir, version_id, "metadata.json")
        if not os.path.isfile(meta_path):
            raise FileNotFoundError(f"Version {version_id} not found")
        with open(meta_path) as f:
            return json.load(f)
