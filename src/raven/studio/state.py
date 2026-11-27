"""Session-only state; no preferences or dataset/artifact writes."""

from copy import deepcopy
from dataclasses import dataclass, field

from raven.studio.services import pipeline_document


@dataclass
class PipelineEditor:
    rows: list = field(default_factory=list)

    def add(self, stage):
        validated = pipeline_document([stage])[0]
        self.rows.append({"enabled": True, "stage": validated})

    def replace(self, index, stage):
        self.rows[index]["stage"] = pipeline_document([stage])[0]

    def move(self, index, delta):
        target = index + delta
        if not 0 <= target < len(self.rows):
            return index
        self.rows[index], self.rows[target] = self.rows[target], self.rows[index]
        return target

    def active(self):
        return pipeline_document([deepcopy(r["stage"]) for r in self.rows if r["enabled"]])

    def load(self, stages):
        validated = pipeline_document(stages)
        self.rows = [{"enabled": True, "stage": row} for row in validated]
