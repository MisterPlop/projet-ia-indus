from predict import predict_price, load_model
import pytest

@pytest.fixture
def model_data():
    return load_model()

def test_predict_price_partial_input(model_data):
    # Only some features provided, others missing
    listing = {
        'neighbourhood_group': 'Manhattan',
        'room_type': 'Entire home/apt'
    }
    with pytest.raises(Exception) as exc_info:
        predict_price(model_data, listing)

def test_predict_price_correct(model_data):
    # Numeric fields as strings
    listing = {
        'neighbourhood_group': 'Brooklyn',
        'room_type': 'Private room',
        'latitude': 40.6782,
        'longitude': -73.9442,
        'minimum_nights': 1,
        'number_of_reviews': 5,
        'availability_365': 200
    }
    
    price = int(predict_price(model_data, listing))
    assert price == 80

def test_predict_price_extra_fields(model_data):
    # Listing contains extra irrelevant fields
    listing = {
        'neighbourhood_group': 'Manhattan',
        'room_type': 'Entire home/apt',
        'latitude': 40.7589,
        'longitude': -73.9851,
        'minimum_nights': 2,
        'number_of_reviews': 10,
        'availability_365': 365,
        'extra_field': 'should be ignored'
    }
    with pytest.raises(Exception) as exc_info:
        predict_price(model_data, listing)

def test_predict_price_missing_categorical(model_data):
    # Missing categorical fields, only numerics provided
    listing = {
        'latitude': 40.7589,
        'longitude': -73.9851,
        'minimum_nights': 2,
        'number_of_reviews': 10,
        'availability_365': 365
    }
    with pytest.raises(Exception) as exc_info:
        predict_price(model_data, listing)

def test_predict_price_all_zeros(model_data):
    # All features set to zero
    listing = {
        'neighbourhood_group': 0,
        'room_type': 0,
        'latitude': 0,
        'longitude': 0,
        'minimum_nights': 0,
        'number_of_reviews': 0,
        'availability_365': 0
    }
    with pytest.raises(Exception) as exc_info:
        predict_price(model_data, listing)
