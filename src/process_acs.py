import warnings
import geopandas as gpd
import pandas as pd
import numpy as np
import scipy.stats as stats
from src.utils.paths import PROJECT_ROOT, DATA_DIR, EXTERNAL_DATA_DIR, RAW_DATA_DIR, INTERIM_DATA_DIR, PROCESSED_DATA_DIR, CACHE_DIR, RESULTS_DIR, LOGS_DIR, MODELS_DIR, IMAGERY_ROOT, ACS_ROOT_DIR

def load_and_prep(file_path, year, boundary, crs="EPSG:3857"):
    """Loads a feather file and projects it to a metric CRS for area calculations."""
    print(f"Loading {year} data...")
    # Read the feather file. Assuming it was saved as a GeoDataFrame.
    gdf = gpd.read_feather(file_path)
    
    # Ensure it is projected to a metric CRS (like Web Mercator or NY State Plane)
    # This is crucial for accurate area calculations in square meters
    if gdf.crs is None:
        warnings.warn(f"CRS is not set for {file_path}. Setting it to ESPG:4326 by default. Make sure this is correct!")
        gdf = gdf.set_crs("EPSG:4326")
    
    if gdf.crs != crs:
        gdf = gdf.to_crs(crs)

    # Clip to NYC boundary
    gdf = gdf.clip(boundary)

    # We only need the geoid, Geometry, Per Capita Income Estimate, and MOE
    # B19301_001E = Per Capita Income Estimate, B19301_001M = Margin of Error
    cols_to_keep = ['geoid', 'geometry', 'per_capita_income_usd', 'per_capita_income_usd_error']
    gdf = gdf[cols_to_keep].copy()
    gdf.rename(columns={col:col + f"_{year}" for col in gdf.columns if col != 'geometry'}, inplace=True)
    
    # 1. Convert MOE to Standard Error (Census uses 90% confidence level -> 1.645)
    gdf[f'SE_{year}'] = gdf[f'per_capita_income_usd_error_{year}'] / 1.645

    # 2. Take the Natural Log of Income
    income = gdf[f'per_capita_income_usd_{year}'].replace(0, np.nan)
    gdf[f'Log_PCI_{year}'] = np.log(income)
    
    # 3. Transform the Standard Error using the Delta Method: SE(ln(X)) ≈ SE(X) / X
    gdf[f'Log_SE_{year}'] = gdf[f'SE_{year}'] / income
    
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

def process_panel(years: list[int], base_year: int, boundaries_path):
    """
    Programmatically loads, aligns, and processes a panel of ACS data for any given list of years.
    """
    years = sorted(years)
    print(f"\n{'='*50}\nProcessing panel for years: {years} | Base: {base_year}\n{'='*50}")
    
    boundaries = gpd.read_file(boundaries_path)
    NYC_boundary = boundaries.to_crs("EPSG:3857")
    boundary = NYC_boundary.dissolve()
    
    # 1. & 2. Load and prep data dynamically
    gdfs = {}
    for year in years:
        file_path = ACS_ROOT_DIR / str(year) / f"ny_tracts_acs5_{year}.feather"
        gdfs[year] = load_and_prep(file_path, year, boundary)
        
    # 3. Spatially align historical years to the base year
    base_gdf = gdfs[base_year]
    matched_gdfs = {}
    for year in years:
        if year != base_year:
            matched_gdfs[year] = spatial_align_max_overlap(
                base_gdf[[f'geoid_{base_year}', 'geometry']], 
                gdfs[year], 
                target_year=base_year, 
                source_year=year
            )
            
    # 4. Merge everything into the base GeoDataFrame
    final_gdf = base_gdf.copy()
    for year in years:
        if year != base_year:
            final_gdf = final_gdf.merge(matched_gdfs[year], on=f'geoid_{base_year}', how='left')
            
    # 5. Run Statistical Tests (Consecutive periods + Full decade)
    consecutive_pairs = [(years[i], years[i+1]) for i in range(len(years)-1)]
    
    for y1, y2 in consecutive_pairs:
        final_gdf = test_significance(final_gdf, y1, y2)
        
    # Test the full start-to-end period (if it's more than 2 years)
    start_year, end_year = years[0], years[-1]
    if (start_year, end_year) not in consecutive_pairs:
        final_gdf = test_significance(final_gdf, start_year, end_year)
        
    # 6. Create the Valid Structural Change Indicator (Filtering out Yo-Yos)
    print("Applying structural trend filter (removing transient 'yo-yo' shocks)...")
    
    # Identify Yo-Yos across any adjacent consecutive periods
    is_yo_yo = pd.Series(False, index=final_gdf.index)
    for i in range(len(consecutive_pairs) - 1):
        p1_start, p1_end = consecutive_pairs[i]
        p2_start, p2_end = consecutive_pairs[i+1]
        
        sig1 = final_gdf[f'significant_{p1_start}_{p1_end}']
        sig2 = final_gdf[f'significant_{p2_start}_{p2_end}']
        diff1 = final_gdf[f'diff_{p1_start}_{p1_end}']
        diff2 = final_gdf[f'diff_{p2_start}_{p2_end}']
        
        yo_yo_here = sig1 & sig2 & (np.sign(diff1) != np.sign(diff2))
        is_yo_yo = is_yo_yo | yo_yo_here
        
    final_gdf['is_yo_yo'] = is_yo_yo
    
    # Any period significant?
    any_significant = pd.Series(False, index=final_gdf.index)
    for y1, y2 in consecutive_pairs:
        any_significant = any_significant | final_gdf[f'significant_{y1}_{y2}']
    if (start_year, end_year) not in consecutive_pairs:
        any_significant = any_significant | final_gdf[f'significant_{start_year}_{end_year}']
        
    final_gdf['Valid_Structural_Change'] = any_significant & ~is_yo_yo
    
    # 7. Computing Inverse-Variance Weighted Mean for stable tracts
    # 7. Construct Final Training Labels
    print("Assigning training labels (True Z-scores for all tracts)...")
    for year in years:
        final_gdf[f'Training_Label_{year}'] = final_gdf[f'Rel_Score_{year}']
        
    print("Training labels successfully assigned!")
    print(f"Filtered out {is_yo_yo.sum()} tracts due to reversing transient shocks (Yo-Yo effect).")
    print(f"Final count of tracts with valid structural change: {final_gdf['Valid_Structural_Change'].sum()} out of {len(final_gdf)} total tracts.")

    # 8. Save out the panel dataset dynamically named
    years_str = "_".join(map(str, years))
    output_name = PROCESSED_DATA_DIR / f"ny_tracts_panel_{years_str}.feather"
    final_gdf.to_feather(output_name)
    print(f"Panel dataset successfully created and saved to {output_name}!")

    # 9. Print Statistics
    score_cols = [f'Rel_Score_{y}' for y in years]
    r2_valid = final_gdf[final_gdf['Valid_Structural_Change'] == True][score_cols].corr() ** 2 
    r2_unchanged = final_gdf[final_gdf['Valid_Structural_Change'] == False][score_cols].corr() ** 2
    r2_all = final_gdf[score_cols].corr() ** 2 
    
    print("\nR-squared of Relative Scores across years for VALID structural change tracts:")
    print(r2_valid)
    print("\nR-squared of Relative Scores across years for unchanged/invalid change tracts:")
    print(r2_unchanged)
    print("\nR-squared of Relative Scores across years for ALL tracts:")
    print(r2_all)
    
    # Average change magnitude for valid structural change tracts vs unchanged tracts
    avg_change_valid = final_gdf[final_gdf['Valid_Structural_Change'] == True][[f'diff_{y1}_{y2}' for y1, y2 in consecutive_pairs]].abs().mean()
    avg_change_unchanged = final_gdf[final_gdf['Valid_Structural_Change'] == False][[f'diff_{y1}_{y2}' for y1, y2 in consecutive_pairs]].abs().mean()
    avg_change_valid_full_period = final_gdf[final_gdf['Valid_Structural_Change'] == True][f'diff_{start_year}_{end_year}'].abs().mean()
    avg_change_unchanged_full_period = final_gdf[final_gdf['Valid_Structural_Change'] == False][f'diff_{start_year}_{end_year}'].abs().mean()
    
    print("\nAverage absolute change magnitude for VALID structural change tracts (by period):")
    print(avg_change_valid)
    print("\nAverage absolute change magnitude for UNCHANGED/invalid change tracts (by period):")
    print(avg_change_unchanged)
    print(f"\nAverage absolute change magnitude for VALID structural change tracts (full period {start_year} to {end_year}): {avg_change_valid_full_period}")
    print(f"Average absolute change magnitude for UNCHANGED/invalid change tracts (full period {start_year} to {end_year}): {avg_change_unchanged_full_period}")


    return final_gdf
    
if __name__ == "__main__":
    
    # Define your parameters here programmatically!
    PANEL_YEARS = [2009, 2014, 2019, 2024]
    BASE_YEAR = 2024
    BOUNDARIES_PATH = EXTERNAL_DATA_DIR / "NYC Borough Boundaries" / "Borough_Boundaries_20260131.geojson"
    
    # Run the pipeline
    final_panel = process_panel(
        years=PANEL_YEARS, 
        base_year=BASE_YEAR, 
        boundaries_path=BOUNDARIES_PATH
    )