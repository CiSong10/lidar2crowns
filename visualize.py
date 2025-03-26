"""
Generate concave hulls (alpha shapes) from LiDAR Point Cloud data
"""


import laspy
import geopandas as gpd
import alphashape
from shapely.geometry import Polygon
import pandas as pd
import numpy as np


def compute_alpha_shapes(las, CRS:str, alpha=0.5, output_filename="tree_polygons.shp"):
    """
    Compute alpha shapes with improved error handling and robustness
    
    Args:
        las (laspy.LasData): Input LAS file data
        CRS (str): Coordinate Reference System EPSG code
        alpha (float): Alpha value for shape concavity (smaller = tighter)
        output_filename (str): Output shapefile name
    """
    coords = np.vstack((las.x, las.y, las.PredInstance)).transpose()
    df = pd.DataFrame(coords, columns=["x", "y", "pred_instance"])

    polygons = []
    for instance_id in df["pred_instance"].unique():
        points = df[df["pred_instance"] == instance_id][["x", "y"]].values

        if alpha is None:
            alpha = compute_adaptive_alpha(points)

        try:
            if len(points) > 2:  # Require at least 3 points to form a polygon
                hull = alphashape.alphashape(points, alpha)
                if isinstance(hull, Polygon):
                    polygons.append({"geometry": hull, "TreeId": instance_id})
        except Exception as e:
            print(f"Error processing instance {instance_id}: {e}")

    gdf = gpd.GeoDataFrame(polygons, crs=f"EPSG:{CRS}")  # Set your coordinate system
    gdf.to_file(output_filename)


def compute_adaptive_alpha(points):
    """
    Compute an adaptive alpha value based on point distribution
    
    Args:
        points (np.ndarray): Array of points
    
    Returns:
        float: Computed alpha value
    """
    # Compute point spread and set alpha accordingly
    point_spread = np.ptp(points, axis=0)
    return min(0.1, np.mean(point_spread) * 0.01)


def main():
    # Read LAS file
    las = laspy.read("USGS_LPC_VA_NorthernVA_B22_w1871n7017_1_out.laz")
    CRS = "6593"
    alpha = 0.5
    output_filename = "tree_polygons.shp"
    compute_alpha_shapes(las, CRS, alpha, output_filename)


if __name__ == "__main__":
    main()