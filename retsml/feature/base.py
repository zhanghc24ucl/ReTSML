from typing import Dict, Type

_REGISTERED_FEATURES: Dict[str, Type['FeatureBuilder']] = {}


def feature(name: str):
    """
    Decorator function to register a feature builder.
    """
    def decorator(cls: Type['FeatureBuilder']):
        _REGISTERED_FEATURES[name] = cls
        return cls
    return decorator


def feature_builder(name):
    if name not in _REGISTERED_FEATURES:
        raise ValueError(f"Feature {name} is not registered.")
    fb = _REGISTERED_FEATURES[name]()
    fb.name = name
    return fb


class FeatureBuilder:
    """
    Base class for building and saving features from financial data.

    This class provides a framework for creating custom feature builders. Each subclass
    should implement the `values` method to define how features are computed from the input data.
    """
    def __init__(self, feature_keys, upstreams=None, use_raw_data=False, universal=False):
        """
        Initialize the FeatureBuilder instance.

        :param feature_keys: A list of feature keys that the builder will generate.
        :param upstreams: A list of feature names that this builder depends on.
        :param use_raw_data: A boolean indicating whether to use raw data in feature computation.
        :param universal: Whether the feature is universal for different targets
        """
        self.feature_keys = tuple(feature_keys)
        self.upstreams = upstreams
        self.use_raw_data = use_raw_data
        self.universal = universal

    def zero_array(self, input_data):
        """
        Generate zero numpy array for feature value
        """
        s, m, _ = input_data.label_shape
        f = len(self.feature_keys)
        from numpy import zeros
        if self.universal:
            return zeros((s, f), dtype=float)
        return zeros((s, m, f), dtype=float)

    def compute(self):
        """Build feature value"""
        from ..data import load_sample_data, load_raw_data

        # Load the preprocessed sample data
        input_data = load_sample_data()

        # Attach raw data if required
        if self.use_raw_data:
            v = load_raw_data()
            input_data.set_raw_data(v)

        # Attach upstream features if required
        if self.upstreams:
            input_data.load_features(self.upstreams)

        # Compute the feature values
        rv = self.values(input_data)

        # Validate the shape of the computed features
        s, m, _ = input_data.label_shape
        f = len(self.feature_keys)
        assert rv.shape == (s, m, f) \
            or rv.shape == (s, f), \
            f"Expected shape {(s, m, f)} or {(s, f)}, but got {rv.shape}"

        return rv

    def build(self):
        """
        Build and save the features.

        This method loads the necessary data, computes the features using the `values` method,
        and saves the features to a file.
        """
        rv = self.compute()
        # Save the computed features to a file
        from numpy import savez_compressed
        from ..data import DATA_ROOT
        savez_compressed(f'{DATA_ROOT}/feature/{self.name}.npz', value=rv, feature_keys=self.feature_keys)
        return rv

    def values(self, input_data):
        """
        Compute the feature values from the input data.

        This method should be implemented by subclasses to define how features are computed.

        :param input_data: The input data object containing samples and features.
        :return: A numpy array of computed feature values with shape (#samples, #instrs, #features).
        """
        raise NotImplementedError("Subclasses should implement this method.")
