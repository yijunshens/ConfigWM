"""Predictor module for ConfigurableEncoder."""

from .base import Predictor


def create_predictor(**kwargs):
    """Create a Predictor."""
    return Predictor(**kwargs)
