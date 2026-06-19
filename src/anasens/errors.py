"""Custom exceptions for AnaSens2."""


class AnaSensError(Exception):
    """Base exception for package-specific errors."""


class ConfigurationError(AnaSensError):
    """Raised when runtime configuration is invalid."""


class DataValidationError(AnaSensError):
    """Raised when input data does not satisfy required schema."""
