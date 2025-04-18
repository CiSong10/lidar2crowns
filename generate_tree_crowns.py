"""
Generate polygons from LiDAR Point Cloud data by tree ID
"""

import argparse
import logging
import os
from concurrent.futures import ProcessPoolExecutor

import alphashape
import geopandas as gpd
import laspy
import numpy as np
import pandas as pd
from shapely.errors import GEOSException
from shapely.geometry import MultiPoint, Polygon
from tqdm import tqdm


def parse_arguments():
    parser = argparse.ArgumentParser(description="Generate polygons from LAS point cloud by tree ID.")

    parser.add_argument("-i", "--input-path", required=True, help="Path to LAS/LAZ file")
    parser.add_argument("--field", default="treeID", help="Field name that defines individual trees")
    parser.add_argument("-c", "--crs", required=True, help="EPSG code of the coordinate system")
    parser.add_argument("--alpha", type=float, default=0.5, help="Alpha value for concave hull (smaller = tighter)")
    parser.add_argument("--workers", type=int, default=10, help="Number of parallel workers")
    parser.add_argument("-o", "--output-suffix", help="Suffix of output tree polygon file")

    return parser.parse_args()


def las_to_df(input_path, field):
    """Convert LAS file to pandas DataFrame with tree IDs."""
    logging.info(f"Reading LAS file: {input_path}")
    las = laspy.read(input_path)
    tree_ids = getattr(las, field)
    mask = las.z > 2  # Focus on mid-to-top canopy, eliminate noise points 
    coords = np.vstack((las.x[mask], las.y[mask], tree_ids[mask])).T

    df = pd.DataFrame(coords, columns=["x", "y", "tree_id"])

    return df[df['tree_id'] > 0]


def process_tree(tree_params):
    """Process a single tree to create a polygon using direct hull."""
    tree_id, points = tree_params

    if len(points) >= 5: # remove junk polygons that have less than 5 points
        try:
            hull = MultiPoint(points).convex_hull
        except GEOSException as e:
            logging.warning(f"Skipping tree {tree_id} due to error: {e}")
            return None
        
        hull = hull.buffer(0.5).buffer(-0.5)  # buffer-unbuffer smoothing
        hull = hull.simplify(0.2, preserve_topology=True) # Simplify the geometry

        if (
            isinstance(hull, Polygon)
            and not hull.is_empty
            and hull.is_valid
            and hull.area >= 2.0  # Remove too small trees
        ):
            return {"geometry": hull, "tree_id": int(tree_id)}
        
    return None


def df_to_polygons(df, args):
    """Convert dataframe of tree points to GeoDataFrame of polygons."""
    tasks = [
        (tree_id, df[df["tree_id"] == tree_id][["x", "y"]].values)
        for tree_id in df["tree_id"].unique()
    ]

    with ProcessPoolExecutor(max_workers=args.workers) as executor:
        results = list(tqdm(executor.map(process_tree, tasks), total=len(tasks), desc="Processing trees"))

    tree_list = [r for r in results if r is not None]
    tree_polygons = gpd.GeoDataFrame(tree_list, crs=f"EPSG:{args.crs}")

    return tree_polygons


def main():
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s:%(message)s")

    args = parse_arguments()
    df = las_to_df(args.input_path, args.field)
    tree_polygons = df_to_polygons(df, args)

    suffix = "_" + args.output_suffix if args.output_suffix else ""
    base = os.path.splitext(os.path.basename(args.input_path))[0]
    output_filename = f"{base}_trees{suffix}.gpkg"
    tree_polygons.to_file(output_filename, driver="GPKG")
    logging.info(f"Saved {len(tree_polygons)} tree canopy polygons to {output_filename}")

if __name__ == "__main__":
    main()
