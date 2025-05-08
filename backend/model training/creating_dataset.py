import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.signal import savgol_filter
from sklearn.preprocessing import StandardScaler
from datetime import datetime

# Set random seed for reproducibility
np.random.seed(42)

# Define wavelength range
wavelengths = np.arange(500, 2010, 10)  # 500nm to 2000nm with 10nm resolution
num_wavelengths = len(wavelengths)
num_samples = 20000

# Create empty dataframe for the dataset
column_names = [f'R_{int(wl)}nm' for wl in wavelengths]
df = pd.DataFrame(columns=column_names)

# Parameters for the simulation
# Define base reflectance curves for different crop types
# These are approximate reflectance patterns based on literature
def crop_base_reflectance(wavelengths, crop_type):
    """Generate base reflectance pattern for different crop types"""
    if crop_type == 'wheat':
        # Wheat has characteristic reflectance with peaks around 750-800nm and 1100nm
        base = 0.3 + 0.4 * np.exp(-((wavelengths - 750) ** 2) / 10000) + 0.2 * np.exp(-((wavelengths - 1100) ** 2) / 20000)
    elif crop_type == 'corn':
        # Corn has higher overall reflectance with peaks around 800nm and 1200nm
        base = 0.35 + 0.45 * np.exp(-((wavelengths - 800) ** 2) / 12000) + 0.25 * np.exp(-((wavelengths - 1200) ** 2) / 25000)
    elif crop_type == 'soybean':
        # Soybeans have distinct patterns in NIR region
        base = 0.32 + 0.38 * np.exp(-((wavelengths - 780) ** 2) / 11000) + 0.22 * np.exp(-((wavelengths - 1150) ** 2) / 22000)
    else:  # Default/mixed
        base = 0.33 + 0.42 * np.exp(-((wavelengths - 770) ** 2) / 10500) + 0.23 * np.exp(-((wavelengths - 1130) ** 2) / 21000)
    
    # Vegetation has low reflectance in visible region (especially blue and red) due to chlorophyll absorption
    visible_mask = wavelengths < 700
    base[visible_mask] *= 0.3 + 0.2 * np.sin(wavelengths[visible_mask] * 0.02)
    
    # Water absorption bands around 1450nm and 1940nm
    water_absorption1 = 0.3 * np.exp(-((wavelengths - 1450) ** 2) / 2000)
    water_absorption2 = 0.4 * np.exp(-((wavelengths - 1940) ** 2) / 2000)
    
    # Apply water absorption
    base = base - water_absorption1 - water_absorption2
    
    # Ensure reflectance stays within reasonable bounds
    base = np.clip(base, 0.05, 0.95)
    
    return base

# Function to add moisture effects to reflectance
def add_moisture_effect(base_reflectance, moisture_content):
    """Modify reflectance based on moisture content"""
    # Higher moisture increases absorption in water bands and decreases overall reflectance
    modified = base_reflectance.copy()
    
    # Water absorption bands become more pronounced with higher moisture
    water_band1_mask = (wavelengths > 1350) & (wavelengths < 1550)
    water_band2_mask = (wavelengths > 1850) & (wavelengths < 2000)
    
    # Scale effect based on moisture content
    moisture_effect = moisture_content / 30.0  # Normalize to typical range
    
    # Apply moisture effect to water absorption bands
    modified[water_band1_mask] -= moisture_effect * 0.3 * base_reflectance[water_band1_mask]
    modified[water_band2_mask] -= moisture_effect * 0.4 * base_reflectance[water_band2_mask]
    
    # Overall NIR reflectance decreases with moisture
    nir_mask = wavelengths >= 700
    modified[nir_mask] -= moisture_effect * 0.1 * base_reflectance[nir_mask]
    
    # Ensure reflectance stays within reasonable bounds
    modified = np.clip(modified, 0.02, 0.95)
    
    return modified

# Function to add fiber effects to reflectance
def add_fiber_effect(reflectance, fiber_content):
    """Modify reflectance based on fiber content"""
    modified = reflectance.copy()
    
    # Fiber affects reflectance primarily in certain NIR regions (1200nm, 1700-1800nm)
    fiber_regions = (wavelengths > 1150) & (wavelengths < 1250) | \
                    (wavelengths > 1650) & (wavelengths < 1850)
    
    # Scale effect based on fiber content
    fiber_effect = fiber_content / 40.0  # Normalize to typical range
    
    # Apply fiber effect - higher fiber generally increases reflectance in these regions
    modified[fiber_regions] += fiber_effect * 0.15 * (1 - reflectance[fiber_regions])
    
    # Ensure reflectance stays within reasonable bounds
    modified = np.clip(modified, 0.02, 0.95)
    
    return modified

# Function to add protein effects to reflectance
def add_protein_effect(reflectance, protein_content):
    """Modify reflectance based on protein content"""
    modified = reflectance.copy()
    
    # Protein affects reflectance primarily around 2050-2150nm (beyond our range) and 1500-1700nm
    protein_regions = (wavelengths > 1500) & (wavelengths < 1700)
    
    # Scale effect based on protein content
    protein_effect = protein_content / 15.0  # Normalize to typical range
    
    # Apply protein effect - protein has specific absorption features
    modified[protein_regions] -= protein_effect * 0.15 * reflectance[protein_regions]
    
    # Ensure reflectance stays within reasonable bounds
    modified = np.clip(modified, 0.02, 0.95)
    
    return modified

# Function to add realistic noise to reflectance
def add_noise(reflectance):
    """Add realistic noise to the reflectance spectra"""
    # Instrument noise - higher at edges of spectrum
    noise_level = 0.005 + 0.01 * np.exp(-((wavelengths - 500) ** 2) / 500000) + 0.01 * np.exp(-((wavelengths - 2000) ** 2) / 500000)
    noise = np.random.normal(0, noise_level, size=len(wavelengths))
    
    # Add some random baseline shifts
    baseline_shift = np.random.normal(0, 0.02) + np.random.normal(0, 0.01) * np.sin(wavelengths * 0.001)
    
    # Combine and apply limits
    noisy_reflectance = reflectance + noise + baseline_shift
    return np.clip(noisy_reflectance, 0.01, 0.99)

# Applying Savitzky-Golay smoothing to make spectra more realistic
def smooth_spectrum(reflectance):
    """Apply Savitzky-Golay filter to smooth the spectrum"""
    return savgol_filter(reflectance, window_length=11, polyorder=3)

# Generate dataset
print(f"Generating {num_samples} simulated crop reflectance spectra...")
reflectance_data = np.zeros((num_samples, num_wavelengths))
moisture_values = np.zeros(num_samples)
fiber_values = np.zeros(num_samples)
protein_values = np.zeros(num_samples)

# Create different crop types for variety
crop_types = np.random.choice(['wheat', 'corn', 'soybean', 'mixed'], size=num_samples, 
                             p=[0.3, 0.3, 0.3, 0.1])

for i in range(num_samples):
    # Generate target values within realistic ranges
    moisture = np.random.uniform(5, 35)  # 5-35% moisture content
    fiber = np.random.uniform(10, 50)    # 10-50% fiber content
    protein = np.random.uniform(5, 25)   # 5-25% protein content
    
    # Base reflectance for the crop type
    base_reflectance = crop_base_reflectance(wavelengths, crop_types[i])
    
    # Apply compositional effects
    modified = add_moisture_effect(base_reflectance, moisture)
    modified = add_fiber_effect(modified, fiber)
    modified = add_protein_effect(modified, protein)
    
    # Add realistic noise
    modified = add_noise(modified)
    
    # Smooth the spectrum for realism
    modified = smooth_spectrum(modified)
    
    # Store the data
    reflectance_data[i, :] = modified
    moisture_values[i] = moisture
    fiber_values[i] = fiber
    protein_values[i] = protein
    
    # Progress indicator
    if (i+1) % 2000 == 0:
        print(f"Generated {i+1}/{num_samples} samples")

# Create the dataframe with reflectance data
df = pd.DataFrame(reflectance_data, columns=column_names)

# Add target variables
df['Moisture'] = moisture_values
df['Fiber'] = fiber_values
df['Protein'] = protein_values

# Add crop type as categorical variable (could be useful for analysis)
df['Crop_Type'] = crop_types

# Add some metadata
df['Sample_ID'] = [f'S{i:05d}' for i in range(num_samples)]
df['Timestamp'] = [datetime.now().strftime('%Y-%m-%d %H:%M:%S') for _ in range(num_samples)]

# Reorder columns to put metadata first, then composition values, then spectral data
first_cols = ['Sample_ID', 'Timestamp', 'Crop_Type', 'Moisture', 'Fiber', 'Protein']
spectral_cols = [col for col in df.columns if col.startswith('R_')]
df = df[first_cols + spectral_cols]

# Export to CSV
output_file = 'crop_reflectance_dataset.csv'
df.to_csv(output_file, index=False)
print(f"Dataset saved to {output_file}")

# Plot a few sample spectra to verify
plt.figure(figsize=(12, 8))

# Plot 5 random spectra
sample_indices = np.random.choice(num_samples, 5, replace=False)
for idx in sample_indices:
    moisture = df.loc[idx, 'Moisture']
    fiber = df.loc[idx, 'Fiber']
    protein = df.loc[idx, 'Protein']
    crop = df.loc[idx, 'Crop_Type']
    
    spectrum = df.iloc[idx][spectral_cols].values
    plt.plot(wavelengths, spectrum, 
             label=f"Sample {idx}: {crop.capitalize()}, M={moisture:.1f}%, F={fiber:.1f}%, P={protein:.1f}%")

plt.xlabel('Wavelength (nm)')
plt.ylabel('Reflectance')
plt.title('Sample Crop Reflectance Spectra')
plt.legend()
plt.grid(True, alpha=0.3)
plt.savefig('sample_spectra.png')
plt.close()

# Create a second plot showing the effect of moisture on spectra
plt.figure(figsize=(12, 8))

# Select wheat samples with similar fiber/protein but different moisture
wheat_samples = df[df['Crop_Type'] == 'wheat']
moisture_levels = [10, 15, 20, 25, 30]

for moisture in moisture_levels:
    # Find sample closest to this moisture level
    sample = wheat_samples.iloc[(wheat_samples['Moisture'] - moisture).abs().argsort()[0]]
    spectrum = sample[spectral_cols].values
    plt.plot(wavelengths, spectrum, 
             label=f"Moisture: {sample['Moisture']:.1f}%, Fiber: {sample['Fiber']:.1f}%, Protein: {sample['Protein']:.1f}%")

plt.xlabel('Wavelength (nm)')
plt.ylabel('Reflectance')
plt.title('Effect of Moisture Content on Wheat Reflectance Spectra')
plt.legend()
plt.grid(True, alpha=0.3)
plt.savefig('moisture_effect_spectra.png')
plt.close()

print("Sample visualization plots created: sample_spectra.png and moisture_effect_spectra.png")
print("Dataset generation complete!")

# Print dataset summary
print("\nDataset Summary:")
print(f"Number of samples: {len(df)}")
print(f"Number of wavelengths: {len(wavelengths)}")
print(f"Wavelength range: {wavelengths[0]}nm to {wavelengths[-1]}nm")
print(f"Moisture range: {df['Moisture'].min():.2f}% to {df['Moisture'].max():.2f}%")
print(f"Fiber range: {df['Fiber'].min():.2f}% to {df['Fiber'].max():.2f}%")
print(f"Protein range: {df['Protein'].min():.2f}% to {df['Protein'].max():.2f}%")
print(f"Crop type distribution: {df['Crop_Type'].value_counts().to_dict()}")