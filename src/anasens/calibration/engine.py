"""Calibration engine."""

import pandas as pd

from anasens.calibration.strategies import divcal, subcal


def calibrate_pairs(
    df: pd.DataFrame,
    analyte_label: str,
    control_label: str,
    match_filters: list[str],
    method: str,
) -> pd.DataFrame:
    """Create a calibration table from analyte-control row matching."""
    analyte_rows = df[df["Label"] == analyte_label].copy()
    control_rows = df[df["Label"] == control_label].copy()

    entries: list[dict[str, object]] = []
    for _, analyte_row in analyte_rows.iterrows():
        cond = control_rows["Sweep"] == analyte_row["Sweep"]
        for filt in match_filters:
            cond &= control_rows[filt] == analyte_row[filt]
        matches = control_rows[cond]
        if len(matches) != 1:
            continue

        control_row = matches.iloc[0]
        a_val = float(analyte_row["peak_heights"])
        c_val = float(control_row["peak_heights"])
        value = subcal(a_val, c_val) if method == "subcal" else divcal(a_val, c_val)

        entry = {
            "Sweep": analyte_row["Sweep"],
            "Analyte": analyte_label,
            "Control": control_label,
            "peak_heights_analyte": a_val,
            "peak_heights_control": c_val,
            method: value,
        }
        for filt in match_filters:
            entry[filt] = analyte_row[filt]
        entries.append(entry)

    return pd.DataFrame(entries)
