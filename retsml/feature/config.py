"""
retsml/feature/config.py

This module provides configuration for feature builders in the ReTSML project.

Functions:
- `feature_builder`: A utility function to retrieve a feature builder instance by name.

Constants:
- `ALL_FEATURE_GROUPS`: A dictionary defining groups of features.
"""


def feature_builder(name):
    return FeatureBuilder._registry[name]()


_BUILDER_CLASSES = [
    ReturnFeatureBuilder,
    TimeFBuilder,
]


ALL_FEATURE_GROUPS = {
    'all': ['nret'],
}
