"""CPU-only layer visibility/class-filter state for the existing desktop viewer."""

from dataclasses import dataclass

HELP = """Raven overlays (focus the viewer; macOS may require Fn + F-key):
F1 help/legend | F2 ground truth | F3 predictions | F4 both on/off
F5 Car only | F6 Pedestrian only | F7 Cyclist only | F8 all classes
F9 print visible prediction confidence/coordinates to the terminal
GT: solid red KITTI reference boxes (all classes)
Predictions: dashed Car cyan-blue / Pedestrian purple / Cyclist amber / others light gray
Predictions include an interior heading arrow/crossbar
Mouse navigation and Open3D's other keys remain unchanged.
"""


@dataclass
class OverlayState:
    ground_truth: tuple = ()
    predictions: tuple = ()
    show_ground_truth: bool = True
    show_predictions: bool = True
    class_filter: str | None = None

    def __post_init__(self):
        self.ground_truth = tuple(self.ground_truth)
        self.predictions = tuple(self.predictions)

    def _visible(self, boxes, enabled):
        return tuple(
            box
            for box in boxes
            if enabled and (self.class_filter is None or box.label == self.class_filter)
        )

    @property
    def visible_ground_truth(self):
        return self._visible(self.ground_truth, self.show_ground_truth)

    @property
    def visible_predictions(self):
        return self._visible(self.predictions, self.show_predictions)

    def apply_key(self, key):
        if key == 291:  # GLFW F2
            self.show_ground_truth = not self.show_ground_truth
        elif key == 292:
            self.show_predictions = not self.show_predictions
        elif key == 293:
            enabled = not (self.show_ground_truth and self.show_predictions)
            self.show_ground_truth = self.show_predictions = enabled
        elif key in (294, 295, 296, 297):
            self.class_filter = {294: "Car", 295: "Pedestrian", 296: "Cyclist", 297: None}[key]
        else:
            return False
        return True

    def status(self):
        return (
            f"GT {'ON' if self.show_ground_truth else 'OFF'} ({len(self.visible_ground_truth)}) | "
            f"Predictions {'ON' if self.show_predictions else 'OFF'} "
            f"({len(self.visible_predictions)}) | Class: {self.class_filter or 'all'}"
        )

    def prediction_details(self):
        lines = [
            f"{index}: {box.label} score={box.score:.9g} center={box.center_xyz} "
            f"LWH={box.size_lwh} yaw={box.yaw_rad:.9g} rad"
            for index, box in enumerate(self.visible_predictions)
        ]
        return (
            "\n".join(lines) or "No visible predictions (layer hidden, empty, or class filtered)."
        )
