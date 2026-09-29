# SPDX-FileCopyrightText: 2025 Dan J. Bower <dbower@eaps.ethz.ch>
#
# SPDX-License-Identifier: GPL-3.0-or-later

"""Shared loading, filtering, processing, and plotting utilities for zircon datasets"""

import logging
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import pandas as pd

from bedroc.core.data_container import DataContainer
from bedroc.core.type_aliases import NpArray

logger: logging.Logger = logging.getLogger(__name__)

PLOT_FEATURE_LABELS: Mapping[str, str] = {
    "Ti": "Ti (ppm)",
    "Hf": "Hf (ppm)",
    "Th": "Th (ppm)",
    "U": "U (ppm)",
    "Y": "Y (ppm)",
    "Nb": "Nb (ppm)",
    "Eu/Eu*": "Eu/Eu*",
    "Ce/Ce*": "Ce/Ce*",
}
"""Display labels (with units) for the zircon features, keyed by the clean feature names"""

LOG_TICK_VALUES: Mapping[str, Sequence[int | float]] = {
    "Ti": (10, 100, 500),
    "Hf": (5000, 10000, 20000),
    "Th": (10, 100, 1000, 5000),
    "U": (10, 100, 1000, 5000),
}
"""Tick values, in original concentration units, for the log-transformed zircon features"""


@dataclass(frozen=True)
class FeatureFilter:
    """Filtering criteria and transform for a single feature column.

    Rows with a missing (NaN) value are always kept, so missingness is handled downstream rather
    than here.

    Args:
        minimum: Exclusive lower bound. ``None`` for no lower bound.
        maximum: Exclusive upper bound. ``None`` for no upper bound.
        log_transform: Whether to log-transform the feature (after filtering) to mitigate right
            skewness. Its uncertainty is converted to a relative uncertainty to match.
    """

    minimum: float | None = None
    maximum: float | None = None
    log_transform: bool = False

    def mask(self, values: pd.Series) -> pd.Series:
        """Boolean mask of the rows to keep.

        Args:
            values: Feature values

        Returns:
            ``True`` for rows within the bounds or with a missing value
        """
        keep: pd.Series = pd.Series(True, index=values.index)
        if self.minimum is not None:
            keep &= values > self.minimum
        if self.maximum is not None:
            keep &= values < self.maximum

        return values.isna() | keep


@dataclass(frozen=True)
class ZirconFilter:
    """Filtering criteria and transforms for a zircon dataset.

    Args:
        features: Filter for each feature column, keyed by column name. Columns absent from the
            dataframe are skipped.
    """

    features: Mapping[str, FeatureFilter]

    def renamed(self, renames: Mapping[str, str]) -> "ZirconFilter":
        """A copy with the feature columns renamed.

        Args:
            renames: New column name for each column to rename. Other columns keep their name.

        Returns:
            The renamed filter
        """
        return ZirconFilter(
            {renames.get(column, column): rule for column, rule in self.features.items()}
        )

    def apply(self, df: pd.DataFrame, uncertainty_columns: Mapping[str, str]) -> pd.DataFrame:
        """Applies the filtering criteria and then the log transforms.

        Args:
            df: Dataframe to filter
            uncertainty_columns: Uncertainty column for each feature column

        Returns:
            The filtered and transformed dataframe
        """
        logger.info("Applying filtering criteria to the data")

        for column, feature_filter in self.features.items():
            if column not in df.columns:
                continue
            if feature_filter.minimum is not None:
                logger.info("Removing %s values less than %g", column, feature_filter.minimum)
            if feature_filter.maximum is not None:
                logger.info("Removing %s values greater than %g", column, feature_filter.maximum)
            df = df.loc[feature_filter.mask(df[column])]

        df = df.copy()
        for column, feature_filter in self.features.items():
            if column not in df.columns or not feature_filter.log_transform:
                continue
            df[uncertainty_columns[column]] = df[uncertainty_columns[column]] / df[column]
            df[column] = np.log(df[column])

        return df


ZIRCON_FILTER: ZirconFilter = ZirconFilter(
    {
        "Ti": FeatureFilter(minimum=0.0, maximum=200.0, log_transform=True),  # or max 300
        "Hf": FeatureFilter(minimum=5000.0, log_transform=True),
        "Th": FeatureFilter(maximum=2000.0, log_transform=True),
        "U": FeatureFilter(maximum=2000.0, log_transform=True),
        "Y": FeatureFilter(log_transform=True),
        "Nb": FeatureFilter(log_transform=True),
        "Eu/Eu*": FeatureFilter(log_transform=True),
        "Ce/Ce*": FeatureFilter(log_transform=True),
    }
)
"""Default zircon filtering criteria from Olivier and Tobias (7/8/2026). Bounds are in ppm. Keyed
by the standard feature names, so datasets with different raw column names should apply
:meth:`ZirconFilter.renamed` first."""


def _find_uncertainty_column(feature: str, columns: Iterable[str], suffixes: Sequence[str]) -> str:
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


def _require_features_present(df: pd.DataFrame, required_columns: Sequence[str]) -> pd.DataFrame:
    """Keeps only rows where every one of ``required_columns`` is not ``NaN``.

    Args:
        df: Dataframe to filter.
        required_columns: Columns that must all be non-``NaN`` for a row to be kept.

    Returns:
        The filtered dataframe.
    """
    return df.dropna(subset=list(required_columns), how="any")


def _clean_numeric_features(df: pd.DataFrame, feature_columns: Sequence[str]) -> pd.DataFrame:
    """Parses feature columns as plain floats.

    Some feature values are reported as below-detection-limit strings (e.g. ``"< 3.72"``), so the
    ``"<"``/``">"`` is stripped before parsing. Some ratio columns (e.g. Ce/Ce*) contain literal inf
    from a near-zero denominator in the source spreadsheet; these are treated as missing rather than
    propagating inf into the analysis.

    Args:
        df: Dataframe to clean
        feature_columns: Feature columns to parse

    Returns:
        The cleaned dataframe
    """
    df = df.copy()
    for feature in feature_columns:
        if df[feature].dtype == object:
            df[feature] = df[feature].astype(str).str.replace(r"[<>]", "", regex=True).str.strip()
            df[feature] = pd.to_numeric(df[feature])
    df[list(feature_columns)] = df[list(feature_columns)].replace([np.inf, -np.inf], np.nan)

    return df


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


@dataclass(frozen=True)
class ZirconSource:
    """Describes how to read and process one zircon source spreadsheet.

    Args:
        name: Name for the dataset, used for the container and output filenames
        filepath: Path to the Excel file
        sheet_name: Name of the sheet containing the data
        feature_columns: Feature columns to use, as ``{raw_column_name: clean_feature_name}``. Use
            the same name for both when the raw name is already clean.
        name_columns: Extra metadata columns to keep, as they appear in the raw sheet
        uncertainty_suffixes: Candidate suffixes for each feature's uncertainty column
        groupby_columns: Columns to group by for the summary statistics
        extra_renames: Optional extra column renames applied after loading. Defaults to ``None``.
        exclude: Rows to drop, as ``{column: values}``. Defaults to no exclusions.
        select_data_column: Optional column passed to :obj:`DataContainer` as
            ``select_data_column``. Defaults to ``None``.
        dump_raw: Whether to also write the loaded (unfiltered) data to Excel. Defaults to
            ``False``.
    """

    name: str
    filepath: Path
    sheet_name: str
    feature_columns: Mapping[str, str]
    name_columns: Sequence[str]
    uncertainty_suffixes: Sequence[str]
    groupby_columns: Sequence[str]
    extra_renames: Mapping[str, str] | None = None
    exclude: Mapping[str, Sequence[str]] = field(default_factory=dict)
    select_data_column: str | None = None
    dump_raw: bool = False

    def load(self) -> tuple[pd.DataFrame, dict[str, str]]:
        """Reads the Excel sheet and selects/labels the columns needed for analysis.

        Values are not cleaned or filtered here (see :meth:`process`), and the feature and
        uncertainty columns keep their raw names.

        Raises:
            ValueError: If :attr:`name_columns` and :attr:`feature_columns` share a name, or if a
                feature's uncertainty column can't be resolved from :attr:`uncertainty_suffixes`.

        Returns:
            The selected dataframe ("Type" capitalized, :attr:`extra_renames` applied), and the
            resolved mapping from each raw feature column to its raw uncertainty column.
        """
        feature_columns: list[str] = list(self.feature_columns)
        shared: set[str] = set(self.name_columns) & set(feature_columns)
        if shared:
            raise ValueError(
                f"name_columns and feature_columns must be disjoint (shared: {sorted(shared)})"
            )

        logger.info("Reading data: %s", self.filepath)
        df: pd.DataFrame = pd.read_excel(self.filepath, sheet_name=self.sheet_name)

        # Important to lock in the index name for later use in the analysis, underscore denotes
        # private usage to avoid conflicts with other columns
        df.index.name = "_index"

        uncertainty_columns: dict[str, str] = {
            feature: _find_uncertainty_column(feature, df.columns, self.uncertainty_suffixes)
            for feature in feature_columns
        }

        df = df.loc[
            :, list(self.name_columns) + feature_columns + list(uncertainty_columns.values())
        ]

        df["Type"] = df["Type"].str.capitalize()

        if self.extra_renames:
            df.rename(columns=self.extra_renames, inplace=True)

        return df, uncertainty_columns

    def process(
        self, *, output_directory: Path | None = None, zircon_filter: ZirconFilter = ZIRCON_FILTER
    ) -> DataContainer:
        """Processes the source spreadsheet into a :obj:`DataContainer`.

        Loads the sheet, parses the features as floats, requires every feature to be present,
        applies the filtering criteria and transforms, drops excluded rows, and writes the processed
        data and summary statistics.

        Args:
            output_directory: Directory to save the processed data. Defaults to ``None`` (no
                saving).
            zircon_filter: Filtering criteria and transforms to apply, keyed by the clean feature
                names. Defaults to :obj:`ZIRCON_FILTER`.

        Returns:
            A :obj:`DataContainer` containing the processed data
        """
        feature_columns: list[str] = list(self.feature_columns)

        df, uncertainty_columns = self.load()

        if self.dump_raw:
            dump_zircon_excel(df, output_directory, f"{self.name}_raw.xlsx")

        df = _clean_numeric_features(df, feature_columns)
        df = _require_features_present(df, feature_columns)

        # The filter is keyed by the clean feature names, but the dataframe still has the raw names
        raw_names: dict[str, str] = {clean: raw for raw, clean in self.feature_columns.items()}
        df = zircon_filter.renamed(raw_names).apply(df, uncertainty_columns)

        for column, values in self.exclude.items():
            logger.info("Removing %s values: %s", column, list(values))
            df = df.loc[~df[column].isin(values)]

        dump_zircon_excel(df, output_directory, f"{self.name}_processed.xlsx")
        export_zircon_summary(
            df,
            output_directory=output_directory,
            name=self.name,
            groupby_columns=self.groupby_columns,
            feature_columns=feature_columns,
        )

        optional_kwargs: dict[str, str] = {}
        if self.select_data_column is not None:
            optional_kwargs["select_data_column"] = self.select_data_column

        return DataContainer.from_dataframe(
            df,
            name=self.name,
            feature_columns=self.feature_columns,
            uncertainty_columns={
                raw_uncertainty: self.feature_columns[raw_feature]
                for raw_feature, raw_uncertainty in uncertainty_columns.items()
            },
            uncertainty_scale=2,
            category_column="Type",
            **optional_kwargs,
        )


def log_tick_overrides(
    zircon_filter: ZirconFilter = ZIRCON_FILTER,
) -> dict[str, tuple[NpArray, list[str]]]:
    """Tick overrides that show log-transformed features in their original units.

    Only features that ``zircon_filter`` log-transforms get overrides, so the ticks always match
    the transform actually applied.

    Args:
        zircon_filter: Filter that was applied to the data. Defaults to :obj:`ZIRCON_FILTER`.

    Returns:
        Mapping from display label (see :obj:`PLOT_FEATURE_LABELS`) to tick positions and labels,
        as expected by :func:`~bedroc.difference.plotting.plot_corner`
    """
    overrides: dict[str, tuple[NpArray, list[str]]] = {}
    for feature, values in LOG_TICK_VALUES.items():
        feature_filter = zircon_filter.features.get(feature)
        if feature_filter is None or not feature_filter.log_transform:
            continue
        overrides[PLOT_FEATURE_LABELS[feature]] = (np.log(values), [f"{v:g}" for v in values])

    return overrides
