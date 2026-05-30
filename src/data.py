"""Data loading and dataset construction utilities for DREAMT sleep staging.

This module will eventually include helpers for loading physiological signals,
loading sleep-stage labels, constructing fixed-length epochs, building PyTorch
datasets, and creating participant-level train/validation/test splits.
"""

# Participant-level splitting should be used to reduce leakage risk. Epochs from
# the same participant should not be split across training, validation, and test
# sets when estimating generalization performance.
