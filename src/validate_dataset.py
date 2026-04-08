"""
Quick validation script to check for NaN values in the dataset BEFORE training.
Run this after build_initial_cache() to diagnose data issues early.
"""
import torch
import pandas as pd
from pathlib import Path
import sys

def validate_shards(cache_dir):
    """Check all cached shards for NaN values."""
    cache_path = Path(cache_dir)
    shard_files = sorted(cache_path.glob("shard_*.pt"))
    
    if not shard_files:
        print("❌ No shard files found in cache directory!")
        return False
    
    print(f"🔍 Validating {len(shard_files)} shards in {cache_dir}\n")
    
    all_valid = True
    total_images = 0
    total_labels = 0
    nan_images = 0
    nan_labels = 0
    
    for shard_file in shard_files:
        try:
            data = torch.load(shard_file, weights_only=True)
            images = data["images"]
            labels = data["labels"]
            
            # Check for NaN in images
            image_nans = torch.isnan(images).any(dim=[1, 2, 3])
            image_nan_count = image_nans.sum().item()
            
            # Check for NaN in labels
            label_nans = torch.isnan(labels)
            label_nan_count = label_nans.sum().item()
            
            total_images += images.shape[0]
            total_labels += labels.shape[0]
            nan_images += image_nan_count
            nan_labels += label_nan_count
            
            status = "✅" if (image_nan_count == 0 and label_nan_count == 0) else "⚠️"
            
            print(f"{status} {shard_file.name}: {images.shape[0]:,} images, {labels.shape[0]:,} labels", end="")
            
            if image_nan_count > 0:
                print(f" | {image_nan_count} NaN images", end="")
                all_valid = False
            if label_nan_count > 0:
                print(f" | {label_nan_count} NaN labels", end="")
                all_valid = False
            print()
            
        except Exception as e:
            print(f"❌ Error loading {shard_file.name}: {e}")
            all_valid = False
    
    print(f"\n{'='*60}")
    print(f"📊 SUMMARY:")
    print(f"   Total images: {total_images:,}")
    print(f"   Total labels: {total_labels:,}")
    print(f"   NaN images: {nan_images:,} ({100*nan_images/total_images:.2f}%)" if total_images > 0 else "   NaN images: 0")
    print(f"   NaN labels: {nan_labels:,} ({100*nan_labels/total_labels:.2f}%)" if total_labels > 0 else "   NaN labels: 0")
    print(f"{'='*60}")
    
    if all_valid:
        print("✅ Dataset is CLEAN - no NaN values detected!")
        return True
    else:
        print("⚠️  Dataset contains NaN values - training may fail!")
        return False


def validate_parquet(parquet_path):
    """Check the original parquet file for NaN values."""
    try:
        df = pd.read_parquet(parquet_path)
        
        print(f"\n🔍 Checking parquet: {parquet_path}")
        print(f"   Shape: {df.shape}")
        
        # Check each column for NaNs
        nan_summary = df.isna().sum()
        cols_with_nans = nan_summary[nan_summary > 0]
        
        if len(cols_with_nans) > 0:
            print(f"\n⚠️  Columns with NaN values:")
            for col, count in cols_with_nans.items():
                pct = 100 * count / len(df)
                print(f"   {col}: {count:,} NaN ({pct:.2f}%)")
            return False
        else:
            print("✅ No NaN values in parquet file")
            return True
            
    except Exception as e:
        print(f"❌ Error loading parquet: {e}")
        return False


if __name__ == "__main__":
    # Adjust these paths as needed
    cache_dir = "/home/abbatenicolas/data/cache/train_cache"
    parquet_path = "/mnt/c/Working Papers/NY State Aerial Imagery Prototype/ny_state_aerial_imagery_prototype/data/processed/temporal_data_t100_years2022-2022.parquet"
    
    # Validate parquet first
    parquet_ok = validate_parquet(parquet_path)
    
    # Validate cached shards
    shards_ok = validate_shards(cache_dir)
    
    sys.exit(0 if (parquet_ok and shards_ok) else 1)
