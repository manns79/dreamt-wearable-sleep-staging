"""Preprocessing utilities for wearable sleep staging signals.

This module will eventually handle signal resampling, normalization,
missing-value handling, label mapping, and other preprocessing steps needed
before feature extraction or model training.
"""

# Preprocessing decisions should be fit on training data only when they learn
# statistics, thresholds, or mappings that could otherwise leak validation/test
# information into the modeling workflow.
