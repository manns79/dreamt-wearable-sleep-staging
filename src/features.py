"""Engineered feature extraction for traditional sleep staging baselines.

This module will eventually contain reusable feature functions for models such
as logistic regression, random forest, and gradient boosting classifiers.
"""

# Feature extraction should operate consistently across splits, with any learned
# transformations estimated from the training set only.
