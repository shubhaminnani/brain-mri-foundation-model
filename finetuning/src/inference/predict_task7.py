"""
Inference for Task 7: Bias & Fairness (same linear probe as Task 6,
but output is evaluated per demographic / acquisition group).
Output format is identical to Task 6.
"""

# Task 7 uses the exact same inference as Task 6 — group-level evaluation
# happens in the challenge evaluation server using the group metadata CSV.
from inference.predict_task6 import predict  # noqa: F401
