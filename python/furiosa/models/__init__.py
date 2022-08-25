"""Furiosa Models"""
from . import vision

# NOTE: Import rust bindings here
from .furiosa_models_vision_native import *

__all__ = ["vision"]
