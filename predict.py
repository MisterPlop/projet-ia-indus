import os
import sys
import pickle
import pandas as pd
import numpy as np

# pip package -> python module name
PKG_TO_MODULE = {
    'scikit-learn': 'sklearn',
    'PyYAML': 'yaml',
    'pillow': 'PIL',
}

# Read requirements.txt and set REQUIRED_PACKAGES accordingly
def get_required_packages():
    req_file = os.path.join(os.path.dirname(__file__), 'requirements.txt')
    if not os.path.exists(req_file):
        print(f"Warning: {req_file} not found. Using default packages.")
        return ['sklearn', 'pandas', 'numpy']
    with open(req_file) as f:
        pkgs = []
        for line in f:
            line = line.strip()
            if line and not line.startswith('#'):
                # Keep the full package name with version specifier
                pkgs.append(line)
        return pkgs

REQUIRED_PACKAGES = get_required_packages()

def check_requirements():
    missing = []
    for pkg in REQUIRED_PACKAGES:
        base = pkg.split('[')[0].split('=')[0].split('<')[0].split('>')[0].strip()
        module = PKG_TO_MODULE.get(base.lower(), base.lower())
        try:
            __import__(module)
        except ImportError as e:
            print(f"Could not import package '{pkg}' (tried importing '{module}'): {e}")
            missing.append(pkg)
    if missing:
        print("Missing packages:", ', '.join(missing))
        print("Please install them using pip : pip install -r requirements.txt")
        sys.exit(1)

def load_model():
    """Load the model and associated data from pickle files"""
    pickles_dir = 'pickles'
    if not os.path.exists(pickles_dir):
        print(f"Directory '{pickles_dir}' not found!")
        sys.exit(1)
        
    for fname in os.listdir(pickles_dir):
        if fname.startswith('airbnb') and fname.endswith('.pkl'):
            print(f"Loading model from: {fname}")
            with open(os.path.join(pickles_dir, fname), 'rb') as f:
                return pickle.load(f)
    print("No airbnb model found in pickles/")
    sys.exit(1)

def preprocess_input(data, model_data):
    """Apply the same preprocessing as during training"""
    # Convert to DataFrame
    df = pd.DataFrame([data])
    
    # Extract components from model_data
    encoders = model_data.get('encoders', {})
    features = model_data.get('features', [])
    
    # Print debug info
    print("Available encoders:", list(encoders.keys()))
    print("Expected features:", features)
    print("Input data columns:", list(df.columns))
    
    # Encode categorical variables (same as in training)
    for col in ['neighbourhood_group', 'room_type']:
        if col in df.columns and col in encoders:
            print(f"Encoding {col}: {df[col].iloc[0]} -> ", end='')
            try:
                df[col] = encoders[col].transform(df[col])
                print(f"{df[col].iloc[0]}")
            except ValueError as e:
                print(f"Error encoding {col}: {e}")
                # Handle unknown categories
                # For now, we'll use a default value (first class)
                df[col] = 0
                print(f"Using default value: 0")
    
    # Order columns according to training features
    if features:
        # Ensure we have all required features
        missing_features = [f for f in features if f not in df.columns]
        if missing_features:
            print(f"Missing features: {missing_features}")
            # Add missing features with default values
            for feat in missing_features:
                df[feat] = 0
        
        # Reorder columns to match training
        df = df[features]
    
    return df

def predict_price(model_data, listing_data):
    """Predict price for a listing"""

    # Check that all required fields are present in the input
    # Determine required fields from model_data if possible, else use listing_data keys
    if isinstance(model_data, dict) and 'features' in model_data:
        required_fields = set(model_data['features'])
    else:
        required_fields = set(listing_data.keys())

    # Check for missing fields
    missing_fields = required_fields - set(listing_data.keys())
    if missing_fields:
        raise ValueError(f"Error: Missing required input fields: {missing_fields}")

    # Check for extra/unexpected fields
    extra_fields = set(listing_data.keys()) - required_fields
    if extra_fields:
        raise ValueError(f"Warning: Unexpected input fields: {extra_fields}")

    # Check for empty or None values
    empty_fields = [k for k, v in listing_data.items() if v is None or (isinstance(v, str) and v.strip() == "")]
    if empty_fields:
        raise ValueError(f"Error: The following fields are empty or None: {empty_fields}")
        
    # Check for out-of-range values (example: latitude/longitude, nights, reviews, availability)
    if 'latitude' in listing_data and not (-90 <= listing_data['latitude'] <= 90):
        raise ValueError("Error: 'latitude' must be between -90 and 90.")
    if 'longitude' in listing_data and not (-180 <= listing_data['longitude'] <= 180):
        raise ValueError("Error: 'longitude' must be between -180 and 180.")
    if 'minimum_nights' in listing_data and listing_data['minimum_nights'] < 1:
        raise ValueError("Error: 'minimum_nights' must be at least 1.")
    if 'number_of_reviews' in listing_data and listing_data['number_of_reviews'] < 0:
        raise ValueError("Error: 'number_of_reviews' cannot be negative.")
    if 'availability_365' in listing_data and not (0 <= listing_data['availability_365'] <= 365):
        raise ValueError("Error: 'availability_365' must be between 0 and 365.")

    # Extract model
    model = model_data.get('model')
    if model is None:
        raise ValueError("No model found in the loaded data")
    
    # Preprocess input
    X_processed = preprocess_input(listing_data, model_data)
    
    print("Preprocessed input shape:", X_processed.shape)
    print("Preprocessed input:")
    print(X_processed)
    
    # Make prediction
    prediction = model.predict(X_processed)
    return prediction[0]

def main():
    check_requirements()
    
    # Load model
    print("Loading model...")
    model_data = load_model()
    
    # Print model structure for debugging
    print("\nModel data keys:", list(model_data.keys()) if isinstance(model_data, dict) else "Not a dictionary")
    
    # Example input (adjust these values as needed)
    example = {
        'neighbourhood_group': 'Manhattan',
        'room_type': 'Entire home/apt',
        'latitude': 40.7589,
        'longitude': -73.9851,
        'minimum_nights': 2,
        'number_of_reviews': 10,
        'availability_365': 365
    }
    print(f"\nPredicting price for:")
    for key, value in example.items():
        print(f"  {key}: {value}")
    
    try:
        predicted_price = predict_price(model_data, example)
        print(f"\n🎯 Predicted price: ${predicted_price:.2f}")
    except Exception as e:
        print(f"\n❌ Error during prediction: {e}")
        
        # Debug information
        print("\nDebugging information:")
        if isinstance(model_data, dict):
            print("Model data contents:")
            for key, value in model_data.items():
                print(f"  {key}: {type(value)}")
        
        # Re-raise for full traceback
        raise

if __name__ == "__main__":
    main()