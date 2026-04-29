from pydantic import BaseModel, Field, computed_field, field_validator, model_validator
from pydantic import ConfigDict
from typing import Literal
from datetime import datetime
import math

# === Request Schema ======================================================

class LogisticsRequest(BaseModel):

    # model config controls Pydantic behaviour
    model_config = ConfigDict(
        str_strip_whitespace=True,
        extra='forbid',
    )

    origin_lat: float = Field(..., ge=-90, le=90, description="Latitude of the origin point")
    origin_lon: float = Field(..., ge=-180, le=180, description="Longitude of the origin point")
    dest_lat: float = Field(..., ge=-90, le=90, description="Latitude of the destination point")
    dest_lon: float = Field(..., ge=-180, le=180, description="Longitude of the destination point")
    cargo_weight_kg: float = Field(..., gt=0, le=200000, description="Weight of the cargo in kilograms")
    hour_of_day: int = Field(..., ge=0, le=23, description="Hour of the day (0-23)")
    day_of_week: int = Field(..., ge=0, le=6, description="Day of the week (0=Monday, 6=Sunday)")
    num_stops: int = Field(1, ge=1, le=20, description="Number of stops during the trip")
    traffic_index: float = Field(1.0, ge=0.5, le=5.0, description="Traffic congestion index (1.0 = normal traffic)")
    vehicle_type: Literal['van', 'truck', 'motorcycle'] = Field('truck', description="Type of vehicle used for transportation")

    # === Field Validators ==================================================
    @field_validator('origin_lat', 'origin_lon', 'dest_lat', 'dest_lon', model='before')
    @classmethod
    def round_coordinates_to_6dp(cls, v) -> float:
        return round(float(v), 6)
    
    # === Cross-Field Validators (model_validator) ================================================
    @model_validator(model='after')
    def origin_and_destination_must_differ(self) -> 'ETARequest':
        same_lat = abs (self.origin_lat - self.dest_lat) < 0.001
        same_lon = abs (self.origin_lon - self.dest_lon) < 0.001
        if same_lat and same_lon:
            raise ValueError("Origin and destination coordinates must differ by at least 0.001 degrees.")
        return self
    
    @model_validator(model='after')
    def motorcycle_weight_limit(self) -> 'ETARequest':
        if self.vehicle_type == 'motorcycle' and self.cargo_weight_kg > 100:
            raise ValueError("Motorcycles cannot carry more than 100 kg of cargo.")
        return self
    
    # === Computed Fields ==================================================
    @computed_field
    @property
    def distance_km(self) -> float:
        R = 6371
        lat1 = math.radians(self.origin_lat)
        lat2 = math.radians(self.dest_lat)
        dlat = lat2 - lat1
        dlon = math.radians(self.dest_lon - self.origin_lon)

        a = math.sin(dlat / 2) ** 2 + math.cos(lat1) * math.cos(lat2) * math.sin(dlon / 2) ** 2
        return R * 2 * math.asin(math.sqrt(a))
    
    @computed_field
    @property
    def is_rush_hour(self) -> bool:

        morning_rush = list(range(7, 10))
        evening_rush = list(range(17, 20))
        return self.hour_of_day in morning_rush + evening_rush
    
    def to_feature_vector(self) -> list[float]:
        return [
            self.distance_km,
            self.cargo_weight_kg,
            float(self.is_rush_hour),
            float(self.day_of_week),
            float(self.num_stops),
            float(self.hour_of_day),
            self.traffic_index,
            1.0 if self.vehicle_type == 'van' else 0.0,
            1.0 if self.vehicle_type == 'truck' else 0.0,
            1.0 if self.vehicle_type == 'motorcycle' else 0.0
        ]
    
    FEATURES_NAMES = [
        'distance_km',
        'cargo_weight_kg',
        'is_rush_hour',
        'day_of_week',
        'num_stops',
        'hour_of_day',
        'traffic_index',
        'vehicle_van',
        'vehicle_truck',
        'vehicle_motorcycle'
    ]

    # === Response Schema =====================================================

    class ETAResponse(BaseModel):
        eta_minutes: float
        eta_human_readable: str
        model_version: str
        distance_km: float
        confidence_low: float
        confidence_high: float
        is_rush_hour: bool
        prediction_timestamp: datetime = Field(default_factory=datetime.utcnow)


    # === Health Check Response Schema =====================================================    
    class HealthResponse(BaseModel):
        status: Literal['healthy', 'degraded', 'unhealthy']
        model_loaded: bool
        api_version: str