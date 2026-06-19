"""Calibration strategy functions."""


def subcal(analyte_value: float, control_value: float) -> float:
    """Subtract control from analyte."""
    return analyte_value - control_value


def divcal(analyte_value: float, control_value: float) -> float:
    """Divide analyte by control, returning nan on division by zero."""
    return analyte_value / control_value if control_value != 0 else float("nan")
