"""Dual-jurisdiction tax preparation engine for Australia and the United States.

Nothing in this package hardcodes a tax rate. Every figure is loaded from a
versioned rule pack under rulepacks/, each entry carrying its statutory citation
and a verification flag. See taxsys.rulepack.
"""
__version__ = "0.1.0"
