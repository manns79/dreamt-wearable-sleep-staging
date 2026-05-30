"""Training entry point for sleep staging models.

This module will eventually support model training through command-line
arguments or configuration files, including data loading, optimization,
validation, checkpointing, and metric logging.
"""

# Training should use participant-level splits to reduce leakage risk and should
# save experiment outputs under results/ or another documented local artifact
# directory.
