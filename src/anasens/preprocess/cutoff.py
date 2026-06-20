"""Leading- and trailing-point cutoff operations for voltammogram sweeps.

Removing artefactual data points from the beginning and end of a scan
is typically the first preprocessing step.  Common causes of unwanted
edge points include instrument settling transients at scan start and
saturation artefacts at scan end.
"""

import pandas as pd


def cut_points(df: pd.DataFrame, cut_start: int = 0, cut_end: int = 0) -> pd.DataFrame:
    """Remove a fixed number of data points from each end of a sweep.

    Negative or out-of-range cut values are clamped to safe limits so
    the function never raises on bad user input.

    Args:
        df: Raw or partially processed sweep DataFrame.  Must have a
            conventional integer range index (reset_index is applied on
            the return value).
        cut_start: Number of rows to discard from the beginning of the
            scan.  Values ``<= 0`` leave the start untouched.
        cut_end: Number of rows to discard from the end of the scan.
            Values ``<= 0`` leave the end untouched.

    Returns:
        pd.DataFrame: Slice of the input with the requested points
        removed and the index reset to start from 0.  An empty
        DataFrame (with the original columns) is returned when the
        cut range leaves no data.

    Example::

        >>> import pandas as pd
        >>> df = pd.DataFrame({"V": range(10), "I": range(10)})
        >>> result = cut_points(df, cut_start=2, cut_end=3)
        >>> list(result["V"])
        [2, 3, 4, 5, 6]
    """
    # Total row count before any slicing.
    num_rows = len(df)

    # Clamp negative values to zero so both arguments are safe.
    start = max(0, int(cut_start))
    end = max(0, int(cut_end))

    # Return an empty frame (preserving column schema) if the input is empty.
    if num_rows == 0:
        return df.copy()

    # If the start cut consumes the whole frame, return an empty frame.
    if start >= num_rows:
        return df.iloc[0:0].copy().reset_index(drop=True)

    # Compute the exclusive stop index accounting for the end cut.
    stop_index = num_rows - end

    # If cuts overlap (nothing left), return an empty frame.
    if stop_index <= start:
        return df.iloc[0:0].copy().reset_index(drop=True)

    return df.iloc[start:stop_index].reset_index(drop=True)
