import pytest
from pydantic import ValidationError
from app.schemas import ETARequest

class TestETARequestValid:
    def test_basic_valid_request(self):
        req = ETARequest(
            origin_lat=5.6037, origin_lon=-0.1870, #Accra
            dest_lat=6.6885, dest_lon=-1.6244, #Kumasi
            cargo_weight_kg=500, hour_of_day=10, day_of_week=1,
        )
        assert req.cargo_weight_kg == 500
        assert req.vehicle_type == 'truck'  # default value
        assert req.num_stops == 1

    def test_computed_distance_accra_to_kumasi(self):
        req = ETARequest(
            origin_lat=5.6037, origin_lon=-0.1870, #Accra
            dest_lat=6.6885, dest_lon=-1.6244, #Kumasi
            cargo_weight_kg=200, hour_of_day=0, day_of_week=1,
        )

        assert 190 < req.distance_km < 210, f'Expected about 200km, got {req.distance_km}'

    def test_rush_hour_detection_morning(self):
        req = ETARequest(
            origin_lat=5.6, origin_lon=-0.2,
            dest_lat=6.7, dest_lon=-1.6,
            cargo_weight_kg=100, hour_of_day=8, day_of_week=1,
        )
        assert req.is_rush_hour is True

    def test_rush_hour_detection_midday(self):
        req = ETARequest(
            origin_lat=5.6, origin_lon=-0.2,
            dest_lat=6.7, dest_lon=-1.6,
            cargo_weight_kg=100, hour_of_day=14, day_of_week=1,
        )
        assert req.is_rush_hour is False

    def test_feature_vector_length(self):
        req = ETARequest(
            origin_lat=5.6, origin_lon=-0.2,
            dest_lat=6.7, dest_lon=-1.6,
            cargo_weight_kg=100, hour_of_day=10, day_of_week=2,
        )
        features = req.to_feature_vector()
        assert len(features) == 10  # Adjust the expected length based on your feature vector implementation
        assert all(isinstance(f, float) for f in features), 'All features should be floats'

    def test_feature_vector_rush_hour_flag(self):
        rush_req = ETARequest(
            origin_lat=5.6, origin_lon=-0.2,
            dest_lat=6.7, dest_lon=-1.6,
            cargo_weight_kg=100, hour_of_day=8, day_of_week=1,
        )
        off_req = ETARequest(
            origin_lat=5.6, origin_lon=-0.2,
            dest_lat=6.7, dest_lon=-1.6,
            cargo_weight_kg=100, hour_of_day=14, day_of_week=1,
        )
        assert rush_req.to_feature_vector()[2] == 1.0, 'Rush hour flag should be 1.0 during rush hours'
        assert off_req.to_feature_vector()[2] == 0.0, 'Rush hour flag should be 0.0 outside rush hours'

class TestETARequestInvalid:
    def test_latitude_out_of_range(self):
        with pytest.raises(ValidationError) as exc_info:
            ETARequest(
                origin_lat=999,
                origin_lon=-0.2, dest_lat=6.7, dest_lon=-1.6,
                cargo_weight_kg=100, hour_of_day=10, day_of_week=1
            )
        assert 'origin_lat' in str(exc_info.value)

    def test_negative_cargo_weight(self):
        with pytest.raises(ValidationError) as exc_info:
            ETARequest(
                origin_lat=5.6, origin_lon=-0.2,
                dest_lat=6.7, dest_lon=-1.6,
                cargo_weight_kg=-50, hour_of_day=10, day_of_week=1
            )

    def test_hour_out_of_range(self):
        with pytest.raises(ValidationError):
            ETARequest(
                origin_lat=5.6, origin_lon=-0.2,
                dest_lat=6.7, dest_lon=-1.6,
                cargo_weight_kg=100, hour_of_day=25, day_of_week=1

            )

    def test_motorcycle_overloaded(self):
        with pytest.raises(ValidationError) as exc_info:
            ETARequest(
                origin_lat=5.6, origin_lon=-0.2,
                dest_lat=6.7, dest_lon=-1.6,
                cargo_weight_kg=200, hour_of_day=10, day_of_week=1,
                vehicle_type='motorcycle'
            )
        assert 'motorcycle' in str(exc_info.value).lower()

    def test_same_origin_and_destination(self):
        with pytest.raises(ValidationError):
            ETARequest(
                origin_lat=5.6037, origin_lon=-0.1870,
                dest_lat=5.6037, dest_lon=-0.1870,
                cargo_weight_kg=100, hour_of_day=10, day_of_week=1
            )
    
    def test_missing_required_field(self):
        with pytest.raises(ValidationError):
            ETARequest(
                origin_lat=5.6, origin_lon=-0.2,
                dest_lat=6.7, dest_lon=-1.6,
                hour_of_day=10, day_of_week=1
            )
            
    def test_extra_fields_forbidden(self):
        with pytest.raises(ValidationError):
            ETARequest(
                origin_lat=5.6, origin_lon=-0.2,
                dest_lat=6.7, dest_lon=-1.6,
                cargo_weight_kg=100, hour_of_day=10, day_of_week=1,
                extra_field='not allowed'
            )
