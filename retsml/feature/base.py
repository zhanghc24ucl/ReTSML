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
    fb = _REGISTERED_FEATURES[name]()
    fb.name = name
    return fb


class FeatureBuilder:
    """
    Base class for building and saving features from financial data.

    This class provides a framework for creating custom feature builders. Each subclass
    should implement the `values` method to define how features are computed from the input data.
    """
    def __init__(self, feature_keys, upstreams=None, use_raw_data=False):
        """
        Initialize the FeatureBuilder instance.

        :param feature_keys: A list of feature keys that the builder will generate.
        :param upstreams: A list of feature names that this builder depends on.
        :param use_raw_data: A boolean indicating whether to use raw data in feature computation.
        """
        self.feature_keys = tuple(feature_keys)
        self.upstreams = upstreams
        self.use_raw_data = use_raw_data

    def zero_array(self, input_data):
        """
        Generate zero numpy array for feature value
        """
        s, m, _ = input_data.label_shape
        f = len(self.feature_keys)
        from numpy import zeros
        return zeros((s, m, f), dtype=float)

    def build(self):
        """
        Build and save the features.

        This method loads the necessary data, computes the features using the `values` method,
        and saves the features to a file.
        """
        from ..data import load_sample_data, load_feature_data, load_raw_data
        from ..data import DATA_ROOT

        # Load the preprocessed sample data
        input_data = load_sample_data()

        # Attach raw data if required
        if self.use_raw_data:
            v = load_raw_data()
            input_data.set_raw_data(v)

        # Attach upstream features if required
        if self.upstreams:
            v, ks = load_feature_data(self.upstreams)
            input_data.attach_features(v, ks)

        # Compute the feature values
        rv = self.values(input_data)

        # Validate the shape of the computed features
        s, m, _ = input_data.label_shape
        f = len(self.feature_keys)
        assert rv.shape == (s, m, f), f"Expected shape {(s, m, f)}, but got {rv.shape}"

        # Save the computed features to a file
        from numpy import savez_compressed
        savez_compressed(f'{DATA_ROOT}/feature/{self.name}.npz', value=rv, feature_keys=self.feature_keys)
        return rv.shape

    def values(self, input_data):
        """
        Compute the feature values from the input data.

        This method should be implemented by subclasses to define how features are computed.

        :param input_data: The input data object containing samples and features.
        :return: A numpy array of computed feature values with shape (#samples, #instrs, #features).
        """
        raise NotImplementedError("Subclasses should implement this method.")
