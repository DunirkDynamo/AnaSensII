"""Custom exception hierarchy for AnaSens2.

All package-specific exceptions inherit from :class:`AnaSensError` so
callers can catch the whole family with a single ``except AnaSensError``
clause while still being able to distinguish categories when needed.

Example::

    from anasens.errors import ConfigurationError

    try:
        loader.load_measurements(path, mode, source)
    except ConfigurationError as exc:
        print(f"Bad configuration: {exc}")
"""


class AnaSensError(Exception):
    """Base exception for all AnaSens2 package errors.

    Raise this directly only when the error does not fit a more specific
    subclass.  Prefer :class:`ConfigurationError` or
    :class:`DataValidationError` for the two most common failure modes.
    """


class ConfigurationError(AnaSensError):
    """Raised when runtime configuration is missing or inconsistent.

    Typical causes include an input path that does not exist, an
    unsupported ``input_source`` value, or a conflicting combination of
    ``analysis_mode`` and the selected path type.
    """


class DataValidationError(AnaSensError):
    """Raised when input data does not satisfy the required schema.

    Typical causes include a CSV file missing the required ``V`` or ``I``
    columns, or a JSON payload whose ``measurements`` list is empty.
    """
