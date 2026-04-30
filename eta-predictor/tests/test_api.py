import pytest
from fastapi.testclient import TestClient
from unittest.mock import patch, MagicMock
import numpy as np
from app.main import app

client = TestClient(app)

VALID_PAYLOAD = {
    "origin_lat": 5.6037,
    "origin_lon": -0.1870,
    "dest_lat": 6.6885,
    "dest_lon": -1.6244,
    "cargo_weight_kg": 500,
    "hour_of_day": 10,
    "day_of_week": 1,
    "num_stops": 3,
    "traffic_index": 1.2,
}

class TestHealthEndpoint:
    def test_health_returns_200(self):
        response = client.get('/health')
        assert response.status_code == 200

    def test_health_response_has_required_fields(self):
        response = client.get('/health')
        data = response.json()
        assert 'status' in data
        assert 'model_loaded' in data
        assert 'api_version' in data
        assert data['api_version'] == '1.0.0'

    def test_health_status_values(self):
        response = client.get('/health')
        data = response.json()
        assert data['status'] in ['healthy', 'degraded', 'unhealthy']

class TestPredictEndpoint:
    @patch('app.main.predictor')
    def test_predict_returns_200_when_model_loaded(self, mock_predictor):
        mock_predictor.is_loaded = True
        mock_predictor.version = 'test-v1'
        mock_predictor.predict.return_value = (185.0, 148.0, 222.0)

        response = client.post('/predict', json=VALID_PAYLOAD)
        data = response.json()
        assert 'eta_minutes' in data
        assert 'eta_human_readable' in data
        assert 'distance_km' in data
        assert 'confidence_low' in data
        assert 'confidence_high' in data

    def test_predict_returns_422_for_invalid_latitude(self):
        bad_payload =  {**VALID_PAYLOAD, 'origin_lat': 999}
        response = client.post('/predict', json=bad_payload)
        assert response.status_code == 422

    def test_predict_returns_422_for_missing_field(self):
        bad_payload = {k: v for k, v in VALID_PAYLOAD.items() if k != 'cargo_weight_kg'}
        response = client.post('/predict', json=bad_payload)
        assert response.status_code == 422
    
    @patch('app.main.predictor')
    def test_predict_returns_503_when_model_not_loaded(self, mock_predictor):
        mock_predictor.is_loaded = False
        response = client.post('/predict', json=VALID_PAYLOAD)
        assert response.status_code == 503
   
