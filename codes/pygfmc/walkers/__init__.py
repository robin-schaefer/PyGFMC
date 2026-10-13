"""Walkers: the walker population, its initializers and reconfiguration.

A GFMC simulation is a population of walkers, each a basis configuration with a
log weight.  This subpackage holds the ``Walker`` pytree, the functions that
prepare an initial population, and the systematic resampling
(``reconfiguration``) that keeps the population size fixed.  It does not import
the engine, which builds on it.
"""
from .walker import (Walker, prepare_walkers_random, prepare_walkers_sz,
                     prepare_walkers_from_configs)
from .reconfiguration import reconfiguration

__all__ = ["Walker", "prepare_walkers_random", "prepare_walkers_sz",
           "prepare_walkers_from_configs", "reconfiguration"]
