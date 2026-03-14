from pathlib import Path

import numpy as np
import pandas as pd

from three_d_fin.processing.analysis import compute_quality_mask
from three_d_fin.processing.configuration import FinConfiguration


def export_tabular_data(
    config: FinConfiguration,
    basepath_output: Path,
    X_c: np.ndarray,
    Y_c: np.ndarray,
    R: np.ndarray,
    check_circle: np.ndarray,
    sector_perct: np.ndarray,
    n_points_in: np.ndarray,
    sections: np.ndarray,
    outliers: np.ndarray,
    dbh_values: np.ndarray,
    tree_locations: np.ndarray,
    tree_heights: np.ndarray,
    cloud_size: int,
    cloud_shape: int,
    tree_analysis: dict[str, np.ndarray] | None = None,
    plot_analysis: dict[str, float] | None = None,
):
    """Export tabular data in XLSX or TXT.

    Parameters
    ----------
    config : FinConfiguration
        A valid FinConfiguration instance.
    basepath_output : str
        A valid output path. it ends with a file base name (no extension).
    X_c : numpy.ndarray
        Matrix containing (x) coordinates of the center of the best-fit circles.
    Y_c : numpy.ndarray
        Matrix containing (y) coordinates of the center of the best-fit circles.
    R : numpy.ndarray
        Vector containing best-fit circle radii.
    sector_perct : numpy.ndarray
        Matrix containing the percentage of occupied sectors.
    check_circle : numpy.ndarray
       Matrix containing the 'check' status of every section of every tree
    n_points_in : numpy.ndarray
        Matrix containing the number of points in the inner circles.
    sections : numpy.ndarray
        Vector containing a Range of uniformly spaced values describing the section quantization
    outliers : numpy.ndarray
        Vector containing the 'outlier probability' of each section.
    dbh_values : numpy.ndarray
        Vector containing DBH values.
    tree_locations : numpy.ndarray
        Matrix containing (x, y, z) coordinates of each tree locator.
    tree_heights : numpy.ndarray
        Matrix containing the heights of individualized trees. It consists of
        (x), (y), (z) and (z0) coordinates of the highest point of the tree and
        a 5th column containing a binary indicator: 0 - tree was too deviated
        from vertical, and height may not be accurate, or 1 - tree was not too
        deviated from vertical, thus height may be trusted.
    cloud_size : int
        Number of point in the cloud (in M points)
    cloud_shape : int
        Area of the cloud in :math: m^2

    """
    # -------------------------------------------------------------------------------------------------------------
    # Exporting results
    # -------------------------------------------------------------------------------------------------------------

    # matrix with tree height, DBH and (x,y) coordinates of each tree
    dbh_and_heights = np.zeros((dbh_values.shape[0], 4))

    if tree_heights.shape[0] != dbh_values.shape[0]:
        tree_heights = tree_heights[0 : dbh_values.shape[0], :]

    dbh_and_heights[:, 0] = tree_heights[:, 3]
    dbh_and_heights[:, 1] = dbh_values[:, 0]
    dbh_and_heights[:, 2] = tree_locations[:, 0]
    dbh_and_heights[:, 3] = tree_locations[:, 1]

    if config.misc is not None and not config.misc.export_txt:
        # Generating aggregated quality value for each section
        quality_mask = compute_quality_mask(
            R, outliers, sector_perct, n_points_in,
            min_radius=config.expert.minimum_diameter / 2.0,
            max_radius=config.advanced.maximum_diameter / 2.0,
            min_sector_perct=config.expert.m_number_sectors / config.expert.number_sectors * 100,
            point_threshold=config.expert.point_threshold,
        )
        # 0: does not pass quality check - 1: passes quality checks
        quality = quality_mask.astype(float)

        # Function to convert data to pandas DataFrames
        def to_pandas(data):
            # Covers np.arrays of shape == 2 (almost every case)
            if len(data.shape) == 2:
                df = pd.DataFrame(
                    data=data,
                    index=["T" + str(i + 1) for i in range(data.shape[0])],
                    columns=["S" + str(i + 1) for i in range(data.shape[1])],
                )

            # Covers np.arrays of shape == 1 (basically, data regarding the normalized height of every section).
            if len(data.shape) == 1:
                df = pd.DataFrame(data=data).transpose()
                df.index = ["Z0"]
                df.columns = ["S" + str(i + 1) for i in range(data.shape[0])]

            return df

        # Converting data to pandas DataFrames for ease to output them as excel files.
        df_diameters = to_pandas(R) * 2
        df_X_c = to_pandas(X_c)
        df_Y_c = to_pandas(Y_c)
        df_sections = to_pandas(sections)
        df_quality = to_pandas(quality)
        df_outliers = to_pandas(outliers)
        df_sector_perct = to_pandas(sector_perct)
        df_n_points_in = to_pandas(n_points_in)

        df_dbh_and_heights = pd.DataFrame(
            data=dbh_and_heights,
            index=["T" + str(i + 1) for i in range(dbh_values.shape[0])],
            columns=["TH", "DBH", "X", "Y"],
        )

        # Description to be added to each excel sheet.
        info_diameters = """Diameter of every section (S) of every tree (T).
            Units are meters.
            """
        info_X_c = """(x) coordinate of the centre of every section (S) of every tree (T)."""
        info_Y_c = """(y) coordinate of the centre of every section (S) of every tree (T)."""
        info_sections = """Normalized height (Z0) of every section (S).
        Units are meters."""
        info_quality = """Overall quality of every section (S) of every tree (T).
        0: Section does not pass quality checks - 1: Section passes quality checks.
        """
        info_outliers = """'Outlier probability' of every section (S) of every tree (T).
        It takes values between 0 and 1.
        """
        info_sector_perct = """Percentage of occupied sectors of every section (S) of every tree (T).
        It takes values between 0 and 100.
        """
        info_n_points_in = """Number of points in the inner circle of every section (S) of every tree (T).
        The lowest, the better.
        """
        info_dbh_and_heights = """Total height (TH) of each tree (T).
        Diameter at breast height (DBH) of each tree (T).
        (x, y) coordinates (X and Y) of each tree (T).
        """
        info_cloud_size = f"This cloud has {cloud_size} million points and its area is {cloud_shape} m2"

        # Converting descriptions to pandas DataFrames for ease to include them in the excel file.
        df_info_diameters = pd.Series(info_diameters)
        df_info_X_c = pd.Series(info_X_c)
        df_info_Y_c = pd.Series(info_Y_c)
        df_info_sections = pd.Series(info_sections)
        df_info_quality = pd.Series(info_quality)
        df_info_outliers = pd.Series(info_outliers)
        df_info_sector_perct = pd.Series(info_sector_perct)
        df_info_n_points_in = pd.Series(info_n_points_in)
        df_info_dbh_and_heights = pd.Series(info_dbh_and_heights)
        df_info_cloud_size = pd.Series(info_cloud_size)

        xls_filename = str(basepath_output) + ".xlsx"

        # Export legacy sheets to a separate deprecated file
        deprecated_filename = str(basepath_output) + "_deprecated.xlsx"
        deprecated_writer = pd.ExcelWriter(deprecated_filename, engine="xlsxwriter")

        df_info_dbh_and_heights.to_excel(
            deprecated_writer, sheet_name="Plot Metrics", header=False, index=False, merge_cells=False,
        )
        df_info_cloud_size.to_excel(
            deprecated_writer, sheet_name="Plot Metrics", startrow=1, header=False, index=False, merge_cells=False,
        )
        df_info_diameters.to_excel(
            deprecated_writer, sheet_name="Diameters", header=False, index=False, merge_cells=False,
        )
        df_info_X_c.to_excel(deprecated_writer, sheet_name="X", header=False, index=False, merge_cells=False)
        df_info_Y_c.to_excel(deprecated_writer, sheet_name="Y", header=False, index=False, merge_cells=False)
        df_info_sections.to_excel(
            deprecated_writer, sheet_name="Sections", header=False, index=False, merge_cells=False,
        )
        df_info_quality.to_excel(
            deprecated_writer, sheet_name="Q(Overall Quality 0-1)", header=False, index=False, merge_cells=False,
        )
        df_info_outliers.to_excel(
            deprecated_writer, sheet_name="Q1(Outlier Probability)", header=False, index=False, merge_cells=False,
        )
        df_info_sector_perct.to_excel(
            deprecated_writer, sheet_name="Q2(Sector Occupancy)", header=False, index=False, merge_cells=False,
        )
        df_info_n_points_in.to_excel(
            deprecated_writer, sheet_name="Q3(Points Inner Circle)", header=False, index=False, merge_cells=False,
        )

        df_dbh_and_heights.to_excel(deprecated_writer, sheet_name="Plot Metrics", startrow=2, startcol=1)
        df_diameters.to_excel(deprecated_writer, sheet_name="Diameters", startrow=2, startcol=1)
        df_X_c.to_excel(deprecated_writer, sheet_name="X", startrow=2, startcol=1)
        df_Y_c.to_excel(deprecated_writer, sheet_name="Y", startrow=2, startcol=1)
        df_sections.to_excel(deprecated_writer, sheet_name="Sections", startrow=2, startcol=1)
        df_quality.to_excel(deprecated_writer, sheet_name="Q(Overall Quality 0-1)", startrow=2, startcol=1)
        df_outliers.to_excel(deprecated_writer, sheet_name="Q1(Outlier Probability)", startrow=2, startcol=1)
        df_sector_perct.to_excel(deprecated_writer, sheet_name="Q2(Sector Occupancy)", startrow=2, startcol=1)
        df_n_points_in.to_excel(deprecated_writer, sheet_name="Q3(Points Inner Circle)", startrow=2, startcol=1)

        deprecated_writer.close()

        # Main output file: only new Tree Analysis and Plot Summary
        writer = pd.ExcelWriter(xls_filename, engine="xlsxwriter")

        # Tree Analysis and Plot Summary sheets
        if tree_analysis is not None:
            df_tree_analysis = pd.DataFrame(
                data={
                    "DBH": tree_analysis["dbh"],
                    "Normal_Section_Area": tree_analysis["normal_section_area"],
                    "Tree_Height": tree_analysis["tree_height"],
                    "Crown_Height": tree_analysis["crown_height"],
                    "Stem_Volume": tree_analysis["stem_volume"],
                    "X": tree_locations[:, 0],
                    "Y": tree_locations[:, 1],
                },
                index=["T" + str(i + 1) for i in range(len(tree_analysis["dbh"]))],
            )

            info_tree_analysis = pd.Series(
                "Per-tree analysis: DBH (m), Normal Section Area (m^2), "
                "Tree Height (m), Crown Height (m), Stem Volume (m^3), Location (X, Y)."
            )
            info_tree_analysis.to_excel(
                writer, sheet_name="Tree Analysis", header=False, index=False, merge_cells=False,
            )
            df_tree_analysis.to_excel(writer, sheet_name="Tree Analysis", startrow=2, startcol=1)

        if plot_analysis is not None:
            metric_labels = {
                "n_trees": ("Number of Trees", "-"),
                "plot_area_m2": ("Plot Area", "m^2"),
                "total_stem_volume_m3": ("Total Stem Volume", "m^3"),
                "mean_dbh_m": ("Mean DBH", "m"),
                "mean_tree_height_m": ("Mean Tree Height", "m"),
                "mean_crown_height_m": ("Mean Crown Height", "m"),
                "mean_normal_section_area_m2": ("Mean Normal Section Area", "m^2"),
                "mean_stem_volume_m3": ("Mean Stem Volume", "m^3"),
                "stem_density_per_ha": ("Stem Density", "trees/ha"),
                "basal_area_m2_per_ha": ("Basal Area", "m^2/ha"),
                "crown_coverage_pct": ("Crown Coverage", "%"),
            }
            rows = []
            for key, value in plot_analysis.items():
                label, unit = metric_labels.get(key, (key, "-"))
                rows.append({"Metric": label, "Value": round(value, 4), "Unit": unit})

            df_plot_summary = pd.DataFrame(rows)
            info_plot_summary = pd.Series("Plot-level summary metrics.")
            info_plot_summary.to_excel(
                writer, sheet_name="Plot Summary", header=False, index=False, merge_cells=False,
            )
            df_plot_summary.to_excel(
                writer, sheet_name="Plot Summary", startrow=2, startcol=1, index=False,
            )

        writer.close()

    else:
        # Legacy TXT exports to deprecated files
        np.savetxt(str(basepath_output) + "_deprecated_diameters.txt", R * 2, fmt=("%.3f"))
        np.savetxt(str(basepath_output) + "_deprecated_X_c.txt", X_c, fmt=("%.3f"))
        np.savetxt(str(basepath_output) + "_deprecated_Y_c.txt", Y_c, fmt=("%.3f"))
        np.savetxt(str(basepath_output) + "_deprecated_check_circle.txt", check_circle, fmt=("%.3f"))
        np.savetxt(str(basepath_output) + "_deprecated_n_points_in.txt", n_points_in, fmt=("%.3f"))
        np.savetxt(str(basepath_output) + "_deprecated_sector_perct.txt", sector_perct, fmt=("%.3f"))
        np.savetxt(str(basepath_output) + "_deprecated_outliers.txt", outliers, fmt=("%.3f"))
        np.savetxt(str(basepath_output) + "_deprecated_dbh_and_heights.txt", dbh_and_heights, fmt=("%.3f"))
        np.savetxt(
            str(basepath_output) + "_deprecated_sections.txt",
            np.column_stack(sections),
            fmt=("%.3f"),
        )

        # New analysis TXT exports
        if tree_analysis is not None:
            tree_analysis_array = np.column_stack([
                tree_analysis["dbh"],
                tree_analysis["normal_section_area"],
                tree_analysis["tree_height"],
                tree_analysis["crown_height"],
                tree_analysis["stem_volume"],
            ])
            np.savetxt(
                str(basepath_output) + "_tree_analysis.txt",
                tree_analysis_array,
                fmt="%.3f",
                header="DBH\tNormal_Section_Area\tTree_Height\tCrown_Height\tStem_Volume",
                delimiter="\t",
            )

        if plot_analysis is not None:
            with open(str(basepath_output) + "_plot_summary.txt", "w") as f:
                for key, value in plot_analysis.items():
                    f.write(f"{key}\t{value:.4f}\n")
