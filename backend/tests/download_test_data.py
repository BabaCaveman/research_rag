"""
Script to download or generate test data for the application
"""
import os
import sys
import pandas as pd
import numpy as np
import argparse

# Add parent directory to path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from backend.core.config import Config

# Load configuration
config = Config()

def generate_synthetic_data(output_path, num_samples=50):
    """
    Generate synthetic spectral data for testing
    
    Args:
        output_path: Path to save the generated CSV
        num_samples: Number of samples to generate
    """
    print(f"Generating synthetic data with {num_samples} samples...")
    
    # Create wavelength columns (e.g., 400nm to 2500nm with 10nm steps)
    wavelengths = range(400, 2510, 10)
    columns = [f"wl_{w}" for w in wavelengths]
    
    # Create synthetic data
    # Use a combination of random noise and some patterns to make it somewhat realistic
    data = np.zeros((num_samples, len(wavelengths)))
    
    for i in range(num_samples):
        # Base pattern (simplified vegetation reflectance curve)
        x = np.array(wavelengths)
        base = np.zeros_like(x, dtype=float)
        
        # Add low reflectance in visible region (400-700nm)
        visible_mask = (x >= 400) & (x <= 700)
        base[visible_mask] = 0.1 + 0.05 * np.sin((x[visible_mask] - 400) * np.pi / 300)
        
        # Add high reflectance in NIR region (700-1300nm)
        nir_mask = (x > 700) & (x <= 1300)
        base[nir_mask] = 0.4 + 0.1 * np.sin((x[nir_mask] - 700) * np.pi / 600)
        
        # Add water absorption features in the SWIR region (1300-2500nm)
        swir_mask = (x > 1300)
        base[swir_mask] = 0.3 + 0.05 * np.sin((x[swir_mask] - 1300) * np.pi / 1200)
        
        # Add dips at water absorption bands
        water_bands = [1450, 1950]
        for wb in water_bands:
            band_effect = 0.2 * np.exp(-((x - wb) ** 2) / (2 * 50 ** 2))
            base = base - band_effect
        
        # Add random noise
        noise = np.random.normal(0, 0.02, len(wavelengths))
        
        # Combine and clip to valid range [0, 1]
        data[i] = np.clip(base + noise, 0, 1)
    
    # Create classes (for demonstration purposes)
    classes = np.random.choice(['healthy', 'stressed', 'diseased'], size=num_samples)
    
    # Create DataFrame
    df = pd.DataFrame(data, columns=columns)
    df.insert(0, 'class', classes)
    df.insert(0, 'sample_id', [f"sample_{i+1}" for i in range(num_samples)])
    
    # Ensure directory exists
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    
    # Save to CSV
    df.to_csv(output_path, index=False)
    print(f"Synthetic data saved to {output_path}")
    print(f"Generated {num_samples} samples with {len(wavelengths)} spectral bands")

def main():
    """Main function"""
    parser = argparse.ArgumentParser(description="Download or generate test data")
    parser.add_argument("-o", "--output", type=str, default="backend/data/test_data.csv",
                        help="Output path for the generated data")
    parser.add_argument("-n", "--num-samples", type=int, default=50,
                        help="Number of samples to generate")
    
    args = parser.parse_args()
    
    generate_synthetic_data(args.output, args.num_samples)

if __name__ == "__main__":
    main()