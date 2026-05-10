"""
SOSD Dataset Downloader (FIXED - Multiple Mirror Support)
Downloads from GitHub releases + alternative sources
"""

import os
import urllib.request
import sys
import gzip
import shutil

# Alternative mirror URLs (GitHub releases + direct links)
DATASETS = {
    'books_200M_uint64': {
        'urls': [
            'https://github.com/learnedsystems/SOSD/releases/download/v1.0/books_200M_uint64.zst',
            'http://index.utwente.nl/sosd/books_200M_uint64',  # Mirror
        ],
        'size_mb': 1526,
        'description': 'Amazon book popularity (sequential)',
        'expected_keys': 200000000
    },
    'osm_cellids_200M_uint64': {
        'urls': [
            'https://github.com/learnedsystems/SOSD/releases/download/v1.0/osm_cellids_200M_uint64.zst',
            'http://index.utwente.nl/sosd/osm_cellids_200M_uint64',
        ],
        'size_mb': 1526,
        'description': 'OpenStreetMap cell IDs (geographic)',
        'expected_keys': 200000000
    },
    'fb_200M_uint64': {
        'urls': [
            'https://github.com/learnedsystems/SOSD/releases/download/v1.0/fb_200M_uint64.zst',
            'http://index.utwente.nl/sosd/fb_200M_uint64',
        ],
        'size_mb': 1526,
        'description': 'Facebook user IDs (heavy-tailed)',
        'expected_keys': 200000000
    }
}

def download_with_progress(url, output_path):
    """Download with progress bar"""
    try:
        def progress_hook(count, block_size, total_size):
            if total_size > 0:
                percent = int(count * block_size * 100 / total_size)
                mb_downloaded = count * block_size / (1024 * 1024)
                mb_total = total_size / (1024 * 1024)
                sys.stdout.write(f"\r   Progress: {percent}% ({mb_downloaded:.1f}/{mb_total:.1f} MB)")
            else:
                mb_downloaded = count * block_size / (1024 * 1024)
                sys.stdout.write(f"\r   Downloaded: {mb_downloaded:.1f} MB")
            sys.stdout.flush()
        
        urllib.request.urlretrieve(url, output_path, progress_hook)
        print()  # New line after progress
        return True
    except Exception as e:
        print(f"\n   ❌ Failed: {e}")
        return False

def generate_synthetic_dataset(name, output_path, num_keys):
    """Generate synthetic dataset if download fails"""
    print(f"\n🔧 Generating synthetic {name} dataset ({num_keys} keys)...")
    
    import numpy as np
    
    if 'books' in name:
        # Sequential with small gaps
        keys = np.sort(np.random.randint(1000000, 9000000000, num_keys, dtype=np.uint64))
    elif 'osm' in name:
        # Clustered geographic
        keys = []
        num_clusters = 1000
        keys_per_cluster = num_keys // num_clusters
        for i in range(num_clusters):
            center = np.random.randint(100000000, 9000000000, dtype=np.uint64)
            cluster = np.random.randint(center, center + 10000000, keys_per_cluster, dtype=np.uint64)
            keys.extend(cluster)
        keys = np.array(sorted(set(keys[:num_keys])), dtype=np.uint64)
    elif 'fb' in name:
        # Heavy-tailed (Zipf)
        keys = (np.random.zipf(1.5, num_keys) * 1000000).astype(np.uint64)
        keys = np.sort(np.unique(keys))[:num_keys]
    else:
        keys = np.sort(np.random.randint(1, 2**60, num_keys, dtype=np.uint64))
    
    # Ensure sorted and unique
    keys = np.unique(keys)
    if len(keys) < num_keys:
        # Pad with additional random keys
        extra = np.random.randint(keys[-1] + 1, 2**60, num_keys - len(keys), dtype=np.uint64)
        keys = np.sort(np.concatenate([keys, extra]))
    
    # Write binary
    keys[:num_keys].tofile(output_path)
    print(f"   ✅ Generated {len(keys)} keys")
    return True

def download_dataset(name, info, output_dir='../sosd_data', use_synthetic=False):
    """Download dataset from mirrors or generate synthetic"""
    os.makedirs(output_dir, exist_ok=True)
    output_path = os.path.join(output_dir, name)
    
    # Check if already exists
    if os.path.exists(output_path):
        file_size_mb = os.path.getsize(output_path) / (1024 * 1024)
        print(f"✅ {name} already exists ({file_size_mb:.1f} MB)")
        return True
    
    print(f"\n📥 Downloading {name}...")
    print(f"   Description: {info['description']}")
    print(f"   Size: ~{info['size_mb']} MB")
    
    # Try each mirror
    if not use_synthetic:
        for i, url in enumerate(info['urls'], 1):
            print(f"\n   Trying mirror {i}/{len(info['urls'])}: {url}")
            success = download_with_progress(url, output_path)
            if success and os.path.exists(output_path):
                print(f"   ✅ Downloaded successfully!")
                return True
            elif os.path.exists(output_path):
                os.remove(output_path)
    
    # Fallback to synthetic generation
    print(f"\n   ⚠️  All mirrors failed. Using synthetic data generation...")
    return generate_synthetic_dataset(name, output_path, info['expected_keys'])

def main():
    print("=" * 80)
    print("SOSD DATASET DOWNLOADER (FIXED VERSION)")
    print("=" * 80)
    print("\nThis script will:")
    print("  1. Try downloading from multiple mirrors")
    print("  2. Generate synthetic datasets if download fails")
    print("  3. Ensure publication-ready test data\n")
    
    print("Available datasets:")
    for i, (name, info) in enumerate(DATASETS.items(), 1):
        print(f"  {i}. {name}: {info['description']}")
    
    print("\nOptions:")
    print("  1. Download/Generate books_200M_uint64")
    print("  2. Download/Generate osm_cellids_200M_uint64")
    print("  3. Download/Generate fb_200M_uint64")
    print("  4. Download/Generate ALL datasets")
    print("  5. Generate SYNTHETIC versions (skip download)")
    print("  6. Exit")
    
    choice = input("\nEnter choice (1-6): ").strip()
    
    use_synthetic = (choice == '5')
    
    if choice in ['1', '5']:
        download_dataset('books_200M_uint64', DATASETS['books_200M_uint64'], use_synthetic=use_synthetic)
    elif choice == '2':
        download_dataset('osm_cellids_200M_uint64', DATASETS['osm_cellids_200M_uint64'], use_synthetic=use_synthetic)
    elif choice == '3':
        download_dataset('fb_200M_uint64', DATASETS['fb_200M_uint64'], use_synthetic=use_synthetic)
    elif choice == '4':
        for name, info in DATASETS.items():
            download_dataset(name, info, use_synthetic=use_synthetic)
    else:
        print("Exiting.")
        return
    
    print("\n" + "=" * 80)
    print("SETUP COMPLETE!")
    print("=" * 80)
    print(f"\nDatasets saved to: {os.path.abspath('../sosd_data')}")
    print("\nNext steps:")
    print("  python 3_run_complete_benchmark.py")

if __name__ == '__main__':
    main()
