import geopandas as gpd
import pandas as pd
import numpy as np
import scipy.stats as stats
from src.utils.paths import PROJECT_ROOT, DATA_DIR, EXTERNAL_DATA_DIR, RAW_DATA_DIR, INTERIM_DATA_DIR, PROCESSED_DATA_DIR, CACHE_DIR, RESULTS_DIR, LOGS_DIR, MODELS_DIR, IMAGERY_ROOT, ACS_ROOT_DIR

def load_and_prep(file_path, year, boundaries, crs="EPSG:3857"):
    """Loads a feather file and projects it to a metric CRS for area calculations."""
    print(f"Loading {year} data...")
    # Read the feather file. Assuming it was saved as a GeoDataFrame.
    gdf = gpd.read_feather(file_path)
    
    # Ensure it is projected to a metric CRS (like Web Mercator or NY State Plane)
    # This is crucial for accurate area calculations in square meters
    if gdf.crs != crs:
        gdf = gdf.to_crs(crs)

    # Clip to NYC boundary
    NYC_boundary = boundaries.to_crs(crs)
    gdf = gdf.clip(NYC_boundary.dissolve())

    # We only need the geoid, Geometry, Per Capita Income Estimate, and MOE
    # B19301_001E = Per Capita Income Estimate, B19301_001M = Margin of Error
    cols_to_keep = ['geoid', 'geometry', 'per_capita_income_usd', 'per_capita_income_usd_error']
    gdf = gdf[cols_to_keep].copy()
    gdf.rename(columns={col:col + f"_{year}" for col in gdf.columns if col != 'geometry'}, inplace=True)
    
    # 1. Convert MOE to Standard Error (Census uses 90% confidence level -> 1.645)
    gdf[f'SE_{year}'] = gdf[f'per_capita_income_usd_error_{year}'] / 1.645

    # 2. Take the Natural Log of Income
    gdf[f'Log_PCI_{year}'] = np.log(gdf[f'per_capita_income_usd_{year}'])
    
    # 3. Transform the Standard Error using the Delta Method: SE(ln(X)) ≈ SE(X) / X
    gdf[f'Log_SE_{year}'] = gdf[f'SE_{year}'] / gdf[f'per_capita_income_usd_{year}']
    
    # 4. Calculate City-Wide Macro Moments for the LOGGED year
    city_log_mean = gdf[f'Log_PCI_{year}'].mean()
    city_log_std = gdf[f'Log_PCI_{year}'].std()
    
    # 5. Calculate Relative Position (Z-score of Logs)
    gdf[f'Rel_Score_{year}'] = (gdf[f'Log_PCI_{year}'] - city_log_mean) / city_log_std
    
    # 6. Calculate Relative Standard Error
    gdf[f'Rel_SE_{year}'] = gdf[f'Log_SE_{year}'] / city_log_std
    
    return gdf

def spatial_align_max_overlap(target_gdf, source_gdf, target_year, source_year):
    """
    Aligns historical tracts to the modern tracts using Maximum Area Overlap.
    For every target tract, it finds the source tract that covers the most of its area.
    """
    print(f"Spatially matching {source_year} tracts to {target_year} boundaries...")
    
    # Compute the spatial intersection of both GeoDataFrames
    intersection = gpd.overlay(target_gdf, source_gdf, how='intersection', keep_geom_type=False)
    
    # Calculate the area of each overlapping polygon
    intersection['overlap_area'] = intersection.geometry.area
    
    # Sort by the target geoid and the overlap area (descending)
    intersection = intersection.sort_values(by=[f'geoid_{target_year}', 'overlap_area'], ascending=[True, False])
    
    # Keep only the source tract that has the largest overlap with the target tract
    best_match = intersection.drop_duplicates(subset=[f'geoid_{target_year}'], keep='first')
    
    # Drop geometry to return a pure DataFrame for merging, and drop the overlap area
    best_match = pd.DataFrame(best_match.drop(columns=['geometry', 'overlap_area']))
    
    return best_match

def test_significance(df, year1, year2):
    """
    Computes the Z-score and P-value for the difference between two ACS estimates.
    Formula: Z = |Est1 - Est2| / sqrt(SE1^2 + SE2^2)
    """
    print(f"Computing statistical significance: {year1} vs {year2}...")
    
    # Estimate difference
    diff = df[f'Rel_Score_{year2}'] - df[f'Rel_Score_{year1}']
    
    # Standard Error of the difference
    se_diff = np.sqrt(df[f'Rel_SE_{year2}']**2 + df[f'Rel_SE_{year1}']**2)
    
    # Z-score
    z_score = np.abs(diff) / se_diff
    
    # Two-tailed P-value
    p_value = 2 * (1 - stats.norm.cdf(z_score))
    
    # Add to dataframe
    df[f'diff_{year1}_{year2}'] = diff
    df[f'zscore_{year1}_{year2}'] = z_score
    df[f'pvalue_{year1}_{year2}'] = p_value
    
    # Boolean flag for statistical significance at 95% confidence (p < 0.05)
    df[f'significant_{year1}_{year2}'] = p_value < 0.05
    
    return df

if __name__ == "__main__":
    # 1. Define paths (Adjust these to point to your actual feather files)
    file_2013 = ACS_ROOT_DIR / "2013" / "ny_tracts_acs5_2013.feather"
    file_2018 = ACS_ROOT_DIR / "2018" / "ny_tracts_acs5_2018.feather"
    file_2024 = ACS_ROOT_DIR / "2024" / "ny_tracts_acs5_2024.feather"
    boundaries = gpd.read_file(EXTERNAL_DATA_DIR / "NYC Borough Boundaries" / "Borough_Boundaries_20260131.geojson")
    # 2. Load and prep data
    # We use 2024 as the base "target" geometry because it reflects the 2020 Census boundaries
    gdf_2013 = load_and_prep(file_2013, 2013, boundaries)
    gdf_2024 = load_and_prep(file_2024, 2024, boundaries)
    gdf_2018 = load_and_prep(file_2018, 2018, boundaries)
    
    # 3. Spatially align 2013 and 2018 data to the 2024 tract boundaries
    matched_2018 = spatial_align_max_overlap(gdf_2024[['geoid_2024', 'geometry']], gdf_2018, 2024, 2018)
    matched_2013 = spatial_align_max_overlap(gdf_2024[['geoid_2024', 'geometry']], gdf_2013, 2024, 2013)
    
    # 4. Merge everything into the 2024 base GeoDataFrame
    final_gdf = gdf_2024.merge(matched_2018, on='geoid_2024', how='left')
    final_gdf = final_gdf.merge(matched_2013, on='geoid_2024', how='left')
    
    # 5. Run the Statistical Tests
    # Compare 2013 to 2018
    final_gdf = test_significance(final_gdf, 2013, 2018)
    
    # Compare 2018 to 2024
    final_gdf = test_significance(final_gdf, 2018, 2024)
    
    # Compare 2013 to 2024 (Full Decade)
    final_gdf = test_significance(final_gdf, 2013, 2024)
    
    # 6. Create the Valid Structural Change Indicator (Filtering out Yo-Yos)
    print("Applying structural trend filter (removing transient 'yo-yo' shocks)...")
    
    # Extract significance flags
    sig_13_18 = final_gdf['significant_2013_2018']
    sig_18_23 = final_gdf['significant_2018_2024']
    sig_13_23 = final_gdf['significant_2013_2024']
    
    # Extract the difference values to check the direction of the change
    diff_13_18 = final_gdf['diff_2013_2018']
    diff_18_23 = final_gdf['diff_2018_2024']
    
    # A "Yo-Yo" tract is significant in both periods, but the direction of change reverses
    # (e.g., positive jump in 2018, negative drop in 2024)
    is_yo_yo = sig_13_18 & sig_18_23 & (np.sign(diff_13_18) != np.sign(diff_18_23))
    final_gdf['is_yo_yo'] = is_yo_yo
    
    # The tract has a valid change if ANY of the three periods are significant, 
    # AND it is NOT a yo-yo.
    any_significant = sig_13_18 | sig_18_23 | sig_13_23
    final_gdf['Valid_Structural_Change'] = any_significant & ~is_yo_yo
    

    print("Computing Inverse-Variance Weighted Mean for stable tracts...")
    # A tiny epsilon is added to avoid DivisionByZero errors just in case 
    # the Census reports an MOE of exactly 0 for any specific tract.
    eps = 1e-9 

    # Step A: Calculate the weights (Inverse of the Variance)
    # Variance is simply the Standard Error squared.
    w_13 = 1 / (final_gdf['Rel_SE_2013']**2 + eps)
    w_18 = 1 / (final_gdf['Rel_SE_2018']**2 + eps)
    w_24 = 1 / (final_gdf['Rel_SE_2024']**2 + eps)

    # Step B: Calculate the sum of weights
    sum_w = w_13 + w_18 + w_24

    # Step C: Calculate the Weighted Mean
    final_gdf['Weighted_Stable_Score'] = (
        (w_13 * final_gdf['Rel_Score_2013']) + 
        (w_18 * final_gdf['Rel_Score_2018']) + 
        (w_24 * final_gdf['Rel_Score_2024'])
    ) / sum_w

    # Step D: Construct the Final Training Labels for the Siamese Network
    # If the tract actually underwent structural change -> Use the specific year's score
    # If the tract was stable -> Overwrite with the smoothed Weighted Mean
    for year in [2013, 2018, 2024]:
        final_gdf[f'Training_Label_{year}'] = np.where(
            final_gdf['Valid_Structural_Change'] == True,
            final_gdf[f'Rel_Score_{year}'],      # Keep the real, changing trajectory
            final_gdf['Weighted_Stable_Score']   # Lock it to the smoothed structural baseline
        )
        
    print("Training labels successfully assigned!")

    num_yo_yo = is_yo_yo.sum()
    print(f"Filtered out {num_yo_yo} tracts due to reversing transient shocks (Yo-Yo effect).")
    print(f"Final count of tracts with valid structural change: {final_gdf['Valid_Structural_Change'].sum()} out of {len(final_gdf)} total tracts.")

    # 7. Save out the panel dataset
    output_name = PROCESSED_DATA_DIR / "ny_tracts_panel_2013_2024.feather"
    final_gdf.to_feather(output_name)
    print(f"Panel dataset successfully created and saved to {output_name}!")

    r2_valid = final_gdf[final_gdf['Valid_Structural_Change'] == False][['Rel_Score_2013','Rel_Score_2018','Rel_Score_2024','Weighted_Stable_Score']].corr() **2 
    r2_all = final_gdf[['Rel_Score_2013','Rel_Score_2018','Rel_Score_2024','Weighted_Stable_Score']].corr() **2 
    print("R-squared of Relative Scores across years for VALID structural change tracts:")
    print(r2_valid)
    print("\nR-squared of Relative Scores across years for ALL tracts:")
    print(r2_all)