# SPDX-FileCopyrightText: 2025 Dan J. Bower <dbower@eaps.ethz.ch>
#
# SPDX-License-Identifier: GPL-3.0-or-later

"""Shared loading utilities for zircon dataset processing"""

import logging
from collections.abc import Iterable, Mapping, Sequence
from pathlib import Path

import pandas as pd

logger: logging.Logger = logging.getLogger(__name__)


def find_uncertainty_column(feature: str, columns: Iterable[str], suffixes: Sequence[str]) -> str:
    """Finds a feature's uncertainty column by trying each of ``suffixes`` in turn.

    Args:
        feature: Bare feature column name (e.g. ``"Ti"`` or ``"Eu/Eu*"``).
        columns: Columns to search for a match (e.g. a dataframe's ``.columns``).
        suffixes: Candidate uncertainty-column suffixes to try, in order.

    Returns:
        The matching uncertainty column name.

    Raises:
        ValueError: If no candidate suffix produces a column present in ``columns``.
    """
    columns = set(columns)
    for suffix in suffixes:
        candidate: str = f"{feature}{suffix}"
        if candidate in columns:
            return candidate

    raise ValueError(
        f"No uncertainty column found for feature {feature!r} (tried suffixes: {suffixes})"
    )


def load_zircon_excel(
    filepath: Path,
    *,
    sheet_name: str,
    name_columns: Sequence[str],
    feature_columns: Sequence[str],
    uncertainty_suffixes: Sequence[str],
    extra_columns: Mapping[str, str] | None = None,
    extra_renames: Mapping[str, str] | None = None,
) -> tuple[pd.DataFrame, dict[str, str]]:
    """Reads a zircon dataset's Excel sheet and selects/labels the columns needed for analysis.

    This covers the mechanical column handling shared by every zircon dataset loader. Dataset-
    specific value cleaning/filtering happens in the caller, on the raw-named columns of the
    returned dataframe, before calling :func:`finalize_feature_columns`.

    Args:
        filepath: Path to the Excel file.
        sheet_name: Name of the sheet containing the data.
        name_columns: Extra metadata columns to keep, as they appear in the raw sheet.
        feature_columns: Raw feature column names to keep (i.e. the keys a caller would later pass
            to ``DataContainer.from_dataframe``'s ``feature_renames``, not the renamed values).
        uncertainty_suffixes: Candidate suffixes tried (via :func:`find_uncertainty_column`) to
            resolve each feature's raw uncertainty column. A dataset with a single fixed suffix
            (e.g. ``"_Int2SE"``) should pass a one-element sequence.
        extra_columns: Optional mapping of new column name to a constant value, assigned via
            ``df[col] = value`` before column selection (e.g. ``{"Dataset": name}``).
        extra_renames: Optional extra column renames applied after selection (e.g.
            ``{"alternate_id": "Locality"}``).

    Raises:
        ValueError: If ``extra_columns``, ``name_columns``, and ``feature_columns`` don't have
            pairwise-disjoint names, or if a feature's uncertainty column can't be resolved (see
            :func:`find_uncertainty_column`).

    Returns:
        The selected dataframe (feature/uncertainty columns still under their raw names, "Type"
        capitalized, ``extra_renames`` applied), and the resolved mapping from each feature to its
        raw uncertainty column name.
    """
    extra_column_names: set[str] = set(extra_columns) if extra_columns else set()
    name_column_names: set[str] = set(name_columns)
    feature_column_names: set[str] = set(feature_columns)
    if not (
        extra_column_names.isdisjoint(name_column_names)
        and extra_column_names.isdisjoint(feature_column_names)
        and name_column_names.isdisjoint(feature_column_names)
    ):
        raise ValueError(
            "extra_columns, name_columns, and feature_columns must be pairwise disjoint "
            f"(got extra_columns={sorted(extra_column_names)}, "
            f"name_columns={sorted(name_column_names)}, "
            f"feature_columns={sorted(feature_column_names)})"
        )

    logger.info("Reading data: %s", filepath)
    df: pd.DataFrame = pd.read_excel(filepath, sheet_name=sheet_name)

    # Important to lock in the index name for later use in the analysis, underscore denotes
    # private usage to avoid conflicts with other columns
    df.index.name = "_index"

    uncertainty_columns: dict[str, str] = {
        feature: find_uncertainty_column(feature, df.columns, uncertainty_suffixes)
        for feature in feature_columns
    }

    if extra_columns:
        for column, value in extra_columns.items():
            df[column] = value

    df = df.loc[
        :,
        list(extra_column_names)
        + list(name_columns)
        + list(feature_columns)
        + list(uncertainty_columns.values()),
    ]

    df["Type"] = df["Type"].str.capitalize()

    if extra_renames:
        df.rename(columns=extra_renames, inplace=True)

    return df, uncertainty_columns


def require_features_present(df: pd.DataFrame, required_columns: Sequence[str]) -> pd.DataFrame:
    """Keeps only rows where every one of ``required_columns`` is not ``NaN``.

    Args:
        df: Dataframe to filter.
        required_columns: Columns that must all be non-``NaN`` for a row to be kept.

    Returns:
        The filtered dataframe.
    """
    return df.dropna(subset=list(required_columns), how="any")


def dump_zircon_excel(
    df: pd.DataFrame, output_directory: Path | None, filename: str, *, sheet_name: str = "Sheet1"
) -> None:
    """Writes ``df`` to ``output_directory / filename``, or does nothing if ``output_directory`` is
    ``None``.

    Args:
        df: Dataframe to write.
        output_directory: Directory to save the file. ``None`` for no output.
        filename: Filename (including extension) to save to.
        sheet_name: Name of the Excel worksheet. Defaults to ``"Sheet1"`` (pandas' own default).
    """
    if output_directory is None:
        return
    df.to_excel(output_directory / Path(filename), sheet_name=sheet_name)


def export_zircon_summary(
    df: pd.DataFrame,
    *,
    output_directory: Path | None,
    name: str,
    groupby_columns: Sequence[str],
    feature_columns: Sequence[str],
) -> None:
    """Writes a groupby-describe summary of ``feature_columns`` to Excel, or does nothing if
    ``output_directory`` is ``None``.

    Args:
        df: Dataframe to summarize.
        output_directory: Directory to save the summary. ``None`` for no output.
        name: Dataset name, used to build the output filename (``f"{name}_summary.xlsx"``).
        groupby_columns: Columns to group by before describing.
        feature_columns: Feature columns to describe.
    """
    if output_directory is None:
        return

    summary: pd.DataFrame = df.groupby(  # pyright: ignore[reportAssignmentType]
        list(groupby_columns)
    )[list(feature_columns)].describe()
    summary_filepath: Path = output_directory / Path(f"{name}_summary.xlsx")
    summary.to_excel(summary_filepath)
    logger.info("Summary statistics saved to %s", summary_filepath)
