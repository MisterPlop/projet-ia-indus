import os
import sys
import pickle
import pandas as pd

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
    pickles_dir = 'pickles'
    for fname in os.listdir(pickles_dir):
        if fname.startswith('airbnb') and fname.endswith('.pkl'):
            with open(os.path.join(pickles_dir, fname), 'rb') as f:
                return pickle.load(f)
    print("No airbnb model found in pickles/")
    sys.exit(1)

def main():
    check_requirements()
    model_dict = load_model()
    # If your pickle contains a dict, extract the model object (commonly under 'model' or similar key)
    if isinstance(model_dict, dict):
        # Try common keys; adjust as needed for your pickle structure
        model = model_dict.get('model') or model_dict.get('estimator') or model_dict.get('pipeline')
        if model is None:
            raise ValueError("Could not find a model object in the loaded dictionary. Check your pickle file structure.")
    else:
        model = model_dict

    # Example input
    example = {
        'neighbourhood_group': 'Manhattan',
        'room_type': 'Entire home/apt',
        'latitude': 40.7589,
        'longitude': -73.9851,
        'minimum_nights': 2,
        'number_of_reviews': 10,
        'availability_365': 365
    }

    # Adapt this part to your model's expected input format
    X = pd.DataFrame([example])
    prediction = model.predict(X)
    print("Predicted price:", prediction[0])

if __name__ == "__main__":
    main()