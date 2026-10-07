"""Evaluation criteria independent of the shaped training reward."""
import math


class SuccessTracker:
    def __init__(self, dt=0.02, threshold=0.05, hold_seconds=0.3):
        self.threshold = threshold
        self.required = math.ceil(hold_seconds / dt - 1e-9)
        self.consecutive = 0
        self.success = False
        self.invalid = False

    def update(self, distance, valid=True):
        self.invalid |= not valid or not math.isfinite(distance)
        self.consecutive = self.consecutive + 1 if not self.invalid and distance <= self.threshold else 0
        self.success |= self.consecutive >= self.required
        return self.success and not self.invalid


def wilson_interval(successes, count):
    if count <= 0:
        raise ValueError("count must be positive")
    z = 1.95996398454
    p = successes / count
    divisor = 1 + z * z / count
    center = (p + z * z / (2 * count)) / divisor
    half = z * math.sqrt(p * (1 - p) / count + z * z / (4 * count * count)) / divisor
    return [max(0.0, center - half), min(1.0, center + half)]
