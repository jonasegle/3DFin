import numpy as np


def compute_quality_mask(
    R: np.ndarray,
    outliers: np.ndarray,
    sector_perct: np.ndarray,
    n_points_in: np.ndarray,
    min_radius: float,
    max_radius: float,
    min_sector_perct: float,
    point_threshold: int,
    max_outlier_prob: float = 0.3,
) -> np.ndarray:
    """Compute a boolean quality mask for all sections of all trees.

    Parameters
    ----------
    R : np.ndarray
        Matrix of shape (n_trees, n_sections) with fitted circle radii.
    outliers : np.ndarray
        Matrix of shape (n_trees, n_sections) with outlier probabilities (0-1).
    sector_perct : np.ndarray
        Matrix of shape (n_trees, n_sections) with sector occupancy percentages (0-100).
    n_points_in : np.ndarray
        Matrix of shape (n_trees, n_sections) with points inside the inner circle.
    min_radius : float
        Minimum valid radius in meters.
    max_radius : float
        Maximum valid radius in meters.
    min_sector_perct : float
        Minimum sector occupancy percentage for a valid section.
    point_threshold : int
        Maximum number of inner-circle points for a valid section.
    max_outlier_prob : float
        Maximum outlier probability for a valid section. Defaults to 0.3.

    Returns
    -------
    np.ndarray
        Boolean mask of shape (n_trees, n_sections). True = valid section.
    """
    valid = (
        (R > 0)
        & (R >= min_radius)
        & (R <= max_radius)
        & (outliers <= max_outlier_prob)
        & (sector_perct >= min_sector_perct)
        & (n_points_in <= point_threshold)
    )
    return valid


def interpolate_dbh(
    dbh_values: np.ndarray,
    R: np.ndarray,
    sections: np.ndarray,
    quality_mask: np.ndarray,
    breast_height: float = 1.3,
) -> np.ndarray:
    """Interpolate DBH for trees where the breast-height section is invalid.

    When the circle at breast height failed quality checks (DBH == 0),
    approximate it via linear interpolation from the nearest valid
    neighbours above and below breast height.  If only one side has a
    valid neighbour, the radius of that single neighbour is used instead.

    Parameters
    ----------
    dbh_values : np.ndarray
        Vector of shape (n_trees, 1) with DBH in meters (0 where invalid).
    R : np.ndarray
        Matrix of shape (n_trees, n_sections) with fitted circle radii.
    sections : np.ndarray
        Vector of section heights (normalized z0).
    quality_mask : np.ndarray
        Boolean mask of shape (n_trees, n_sections). True = valid section.
    breast_height : float
        Breast height in meters. Defaults to 1.3.

    Returns
    -------
    np.ndarray
        Updated copy of dbh_values with interpolated values where possible.
    """
    dbh_out = dbh_values.copy()
    n_trees = R.shape[0]
    bh_idx = int(np.argmin(np.abs(sections - breast_height)))

    for i in range(n_trees):
        if dbh_out[i, 0] > 0:
            continue  # already has a valid DBH

        # Search for nearest valid neighbour below breast height
        below_idx = None
        for j in range(bh_idx - 1, -1, -1):
            if quality_mask[i, j]:
                below_idx = j
                break

        # Search for nearest valid neighbour above breast height
        above_idx = None
        for j in range(bh_idx + 1, len(sections)):
            if quality_mask[i, j]:
                above_idx = j
                break

        if below_idx is not None and above_idx is not None:
            # Linear interpolation
            h_below = sections[below_idx]
            h_above = sections[above_idx]
            r_below = R[i, below_idx]
            r_above = R[i, above_idx]
            t = (breast_height - h_below) / (h_above - h_below)
            r_bh = r_below + t * (r_above - r_below)
            dbh_out[i, 0] = r_bh * 2.0
        elif below_idx is not None:
            dbh_out[i, 0] = R[i, below_idx] * 2.0
        elif above_idx is not None:
            dbh_out[i, 0] = R[i, above_idx] * 2.0

    return dbh_out


def compute_normal_section_area(dbh_values: np.ndarray) -> np.ndarray:
    """Compute cross-sectional area at breast height.

    Parameters
    ----------
    dbh_values : np.ndarray
        Vector of shape (n_trees, 1) with diameter at breast height in meters.

    Returns
    -------
    np.ndarray
        Vector of shape (n_trees,) with area in m^2. Zero where DBH is zero.
    """
    dbh = dbh_values[:, 0]
    return np.pi * (dbh / 2.0) ** 2


def compute_crown_height(
    R: np.ndarray,
    sections: np.ndarray,
    quality_mask: np.ndarray,
) -> np.ndarray:
    """Compute crown height as the height of the highest valid stem section.

    Parameters
    ----------
    R : np.ndarray
        Matrix of shape (n_trees, n_sections) with fitted circle radii.
    sections : np.ndarray
        Vector of section heights (normalized z0).
    quality_mask : np.ndarray
        Boolean mask of shape (n_trees, n_sections). True = valid section.

    Returns
    -------
    np.ndarray
        Vector of shape (n_trees,) with crown height in meters. Zero if no valid sections.
    """
    n_trees = R.shape[0]
    crown_h = np.zeros(n_trees)
    for i in range(n_trees):
        valid_indices = np.where(quality_mask[i, :])[0]
        if len(valid_indices) > 0:
            crown_h[i] = sections[valid_indices[-1]]
    return crown_h


def compute_stem_volume(
    R: np.ndarray,
    sections: np.ndarray,
    quality_mask: np.ndarray,
    tree_heights: np.ndarray,
    top_shape: str = "cone",
) -> np.ndarray:
    """Compute stem volume using three approximation regions.

    1. Cylinder from ground (z0=0) to lowest valid section.
    2. Conic frustums between consecutive valid sections.
    3. Top shape (cone/paraboloid/neiloid) from highest valid section to tree top.

    Parameters
    ----------
    R : np.ndarray
        Matrix of shape (n_trees, n_sections) with fitted circle radii.
    sections : np.ndarray
        Vector of section heights (normalized z0).
    quality_mask : np.ndarray
        Boolean mask of shape (n_trees, n_sections). True = valid section.
    tree_heights : np.ndarray
        Matrix of shape (n_trees, 5). Column 3 is the normalized tree height (z0).
    top_shape : str
        Shape model for the top segment. One of "cone", "paraboloid", "neiloid".

    Returns
    -------
    np.ndarray
        Vector of shape (n_trees,) with stem volume in m^3.
    """
    top_factors = {"cone": 1.0 / 3.0, "paraboloid": 1.0 / 2.0, "neiloid": 1.0 / 4.0}
    top_factor = top_factors.get(top_shape, 1.0 / 3.0)

    n_trees = R.shape[0]
    volumes = np.zeros(n_trees)

    for i in range(n_trees):
        valid_idx = np.where(quality_mask[i, :])[0]
        if len(valid_idx) == 0:
            continue

        radii = R[i, valid_idx]
        heights = sections[valid_idx]

        # 1. Base cylinder: ground to lowest valid section
        r_low = radii[0]
        h_low = heights[0]
        v_base = np.pi * r_low**2 * h_low

        # 2. Conic frustums between consecutive valid sections
        v_frustums = 0.0
        for j in range(len(valid_idx) - 1):
            r1 = radii[j]
            r2 = radii[j + 1]
            h = heights[j + 1] - heights[j]
            v_frustums += np.pi / 3.0 * h * (r1**2 + r1 * r2 + r2**2)

        # 3. Top shape: highest valid section to tree top
        r_top = radii[-1]
        h_top = max(0.0, tree_heights[i, 3] - heights[-1])
        v_top = np.pi * top_factor * r_top**2 * h_top

        volumes[i] = v_base + v_frustums + v_top

    return volumes


def compute_tree_analysis(
    R: np.ndarray,
    sections: np.ndarray,
    outliers: np.ndarray,
    sector_perct: np.ndarray,
    n_points_in: np.ndarray,
    dbh_values: np.ndarray,
    tree_heights: np.ndarray,
    tree_locations: np.ndarray,
    min_radius: float,
    max_radius: float,
    min_sector_perct: float,
    point_threshold: int,
    top_shape: str = "cone",
) -> dict[str, np.ndarray]:
    """Compute all per-tree analysis metrics.

    Parameters
    ----------
    R : np.ndarray
        Matrix of shape (n_trees, n_sections) with fitted circle radii.
    sections : np.ndarray
        Vector of section heights (normalized z0).
    outliers : np.ndarray
        Matrix of shape (n_trees, n_sections) with outlier probabilities.
    sector_perct : np.ndarray
        Matrix of shape (n_trees, n_sections) with sector occupancy percentages.
    n_points_in : np.ndarray
        Matrix of shape (n_trees, n_sections) with inner circle point counts.
    dbh_values : np.ndarray
        Vector of shape (n_trees, 1) with DBH in meters.
    tree_heights : np.ndarray
        Matrix of shape (n_trees, 5). Column 3 is normalized height.
    tree_locations : np.ndarray
        Matrix of shape (n_trees, 3) with (x, y, z) coordinates.
    min_radius : float
        Minimum valid radius in meters.
    max_radius : float
        Maximum valid radius in meters.
    min_sector_perct : float
        Minimum sector occupancy percentage.
    point_threshold : int
        Maximum inner-circle point count for validity.
    top_shape : str
        Top volume shape model. One of "cone", "paraboloid", "neiloid".

    Returns
    -------
    dict[str, np.ndarray]
        Keys: quality_mask, normal_section_area, crown_height, stem_volume, dbh, tree_height.
    """
    # Align tree_heights rows with number of trees in R
    n_trees = R.shape[0]
    th = tree_heights[:n_trees, :] if tree_heights.shape[0] > n_trees else tree_heights

    quality_mask = compute_quality_mask(
        R, outliers, sector_perct, n_points_in,
        min_radius, max_radius, min_sector_perct, point_threshold,
    )

    # Interpolate DBH for trees where the breast-height section was invalid
    dbh_corrected = interpolate_dbh(dbh_values, R, sections, quality_mask)

    return {
        "quality_mask": quality_mask,
        "normal_section_area": compute_normal_section_area(dbh_corrected),
        "crown_height": compute_crown_height(R, sections, quality_mask),
        "stem_volume": compute_stem_volume(R, sections, quality_mask, th, top_shape),
        "dbh": dbh_corrected[:, 0].copy(),
        "tree_height": th[:, 3].copy(),
    }


def compute_crown_coverage(
    assigned_cloud: np.ndarray,
    avg_crown_height: float,
    pixel_resolution: float = 0.5,
) -> tuple[float, float, float]:
    """Compute crown coverage as the fraction of ground pixels occupied by canopy.

    Parameters
    ----------
    assigned_cloud : np.ndarray
        Point cloud of shape (n_points, 6): [x, y, z, z0, tree_id, dist_to_axis].
    avg_crown_height : float
        Average crown height. Points above this height are considered canopy.
    pixel_resolution : float
        XY pixel size in meters. Defaults to 0.5.

    Returns
    -------
    tuple[float, float, float]
        (coverage_percentage, occupied_pixels, total_pixels).
    """
    x_all = assigned_cloud[:, 0]
    y_all = assigned_cloud[:, 1]
    x_min, x_max = x_all.min(), x_all.max()
    y_min, y_max = y_all.min(), y_all.max()

    nx = max(1, int(np.ceil((x_max - x_min) / pixel_resolution)))
    ny = max(1, int(np.ceil((y_max - y_min) / pixel_resolution)))
    total_pixels = float(nx * ny)

    crown_mask = assigned_cloud[:, 3] > avg_crown_height
    if not np.any(crown_mask):
        return 0.0, 0.0, total_pixels

    crown_x = assigned_cloud[crown_mask, 0]
    crown_y = assigned_cloud[crown_mask, 1]

    px = ((crown_x - x_min) / pixel_resolution).astype(np.int64)
    py = ((crown_y - y_min) / pixel_resolution).astype(np.int64)
    np.clip(px, 0, nx - 1, out=px)
    np.clip(py, 0, ny - 1, out=py)

    # Encode (px, py) as single integer for efficient unique count
    pixel_ids = px * ny + py
    occupied_pixels = float(np.unique(pixel_ids).shape[0])

    coverage_pct = occupied_pixels / total_pixels * 100.0
    return coverage_pct, occupied_pixels, total_pixels


def compute_plot_analysis(
    tree_analysis: dict[str, np.ndarray],
    assigned_cloud: np.ndarray,
    cloud_shape: float,
    pixel_resolution: float = 0.5,
) -> dict[str, float]:
    """Compute plot-level summary metrics.

    Parameters
    ----------
    tree_analysis : dict[str, np.ndarray]
        Output from compute_tree_analysis.
    assigned_cloud : np.ndarray
        Full assigned point cloud of shape (n_points, 6).
    cloud_shape : float
        Ground area of the plot in m^2.
    pixel_resolution : float
        XY pixel size for crown coverage. Defaults to 0.5.

    Returns
    -------
    dict[str, float]
        Plot-level summary metrics.
    """
    dbh = tree_analysis["dbh"]
    tree_height = tree_analysis["tree_height"]
    crown_height = tree_analysis["crown_height"]
    normal_area = tree_analysis["normal_section_area"]
    stem_vol = tree_analysis["stem_volume"]

    n_trees = float(len(dbh))
    plot_area = float(cloud_shape)
    ha_factor = plot_area / 10000.0 if plot_area > 0 else 1.0

    def _mean_nonzero(arr: np.ndarray) -> float:
        valid = arr[arr > 0]
        return float(np.mean(valid)) if len(valid) > 0 else 0.0

    avg_crown_h = _mean_nonzero(crown_height)
    crown_cov_pct, _, _ = compute_crown_coverage(assigned_cloud, avg_crown_h, pixel_resolution)

    return {
        "n_trees": n_trees,
        "plot_area_m2": plot_area,
        "total_stem_volume_m3": float(np.sum(stem_vol)),
        "mean_dbh_m": _mean_nonzero(dbh),
        "mean_tree_height_m": _mean_nonzero(tree_height),
        "mean_crown_height_m": avg_crown_h,
        "mean_normal_section_area_m2": _mean_nonzero(normal_area),
        "mean_stem_volume_m3": _mean_nonzero(stem_vol),
        "stem_density_per_ha": n_trees / ha_factor,
        "basal_area_m2_per_ha": float(np.sum(normal_area)) / ha_factor,
        "crown_coverage_pct": crown_cov_pct,
    }
