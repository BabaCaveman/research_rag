#!/usr/bin/env python
import pickle

# Add this to your project
def print_metadata_info(metadata_path):
    """Print the contents of the metadata file to understand what columns are expected."""
    try:
        with open(metadata_path, 'rb') as f:
            metadata = pickle.load(f)
        
        print("Metadata contents:")
        print(f"Feature columns: {metadata.get('feature_columns', 'Not found')}")
        print(f"Target column: {metadata.get('target_column', 'Not found')}")
        print(f"Other keys: {list(metadata.keys())}")
        
        return metadata
    except Exception as e:
        print(f"Error reading metadata: {str(e)}")
        return None

# Example usage
if __name__ == "__main__":
    metadata = print_metadata_info("./models/model_Moisture_metadata.pkl")