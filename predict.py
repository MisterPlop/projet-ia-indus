import os
import sys
import pickle
import pandas as pd

# Read requirements.txt and set REQUIRED_PACKAGES accordingly
def get_required_packages():
    req_file = os.path.join(os.path.dirname(__file__), 'requirements.txt')
    if not os.path.exists(req_file):
        return ['sklearn', 'pandas', 'numpy']
    with open(req_file) as f:
        pkgs = []
        for line in f:
            line = line.strip()
            if line and not line.startswith('#'):
                # Get package name before any version specifier
                pkg = line.split('==')[0].split('>=')[0].split('<=')[0].strip()
                if pkg:
                    pkgs.append(pkg)
        return pkgs

REQUIRED_PACKAGES = get_required_packages()

def check_requirements():
    missing = []
    for pkg in REQUIRED_PACKAGES:
        try:
            __import__(pkg)
        except ImportError as e:
            print(f"Could not import package '{pkg}': {e}")
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
    model = load_model()

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