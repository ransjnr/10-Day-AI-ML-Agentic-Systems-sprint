import time
from datetime import datetime
from fastapi import FastAPI, HTTPException, status
from fastapi.middleware.cors import CORSMiddleware
from app.schemas import ETARequest, ETAResponse, HealthResponse
from app.predictor import ETAPredictor

# ====== APP SETUP ======

app = FastAPI(
    title='ETA Predictor API',
    description= """

    This API predicts the Estimated Time of Arrival (ETA) for delivery routes based on various features such as distance, cargo weight, traffic conditions, and more. It uses a machine learning model trained on historical delivery data to provide accurate ETA predictions along with confidence intervals.
    ## How to use
    1. POST /predict with a JSON body containing the required features to get an ETA prediction.
    2. GET /health to check the health status of the API.
    ## Features
    - distance_km: Distance of the delivery route in kilometers.
    - cargo_weight_kg: Weight of the cargo in kilograms.
    - is_rush_hour: Boolean indicating if the delivery is during rush hour.
    - day_of_week: Day of the week (0=Monday, 6=Sunday).
    - num_stops: Number of stops in the delivery route.
    - hour_of_day: Hour of the day (0-23).
    - traffic_index: Traffic congestion index (1.0 = normal traffic).
    - vehicle_type: Type of vehicle used for delivery (van, truck, motorcycle).
    ## Response
    The response includes the predicted ETA in minutes, along with a confidence interval (lower and upper
    bounds) to indicate the uncertainty of the prediction.
    ## Model Management
    The API loads a pre-trained machine learning model at startup. If the model file is not found, the API will return an error when trying to predict. Make sure to train and save a model before using the prediction endpoint.
    """,

    version='1.0.0',
    docs_url='/docs',
    redoc_url='/redoc'
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=['GET', 'POST'],
    allow_credentials=True,
    allow_headers=["*"],
)

predictor = ETAPredictor()
STARTUP_TIME = time.time()

# ===STARTUP EVENT ===
@app.on_event('startup')
async def load_model_on_startup():
    success = predictor.load()
    if not success:
        print('Failed to load model on startup. The API will not be able to predict until a model is loaded.')
        print('POST /predict will return an error until a model is successfully loaded.')

# == ENDPOINTS ===

@app.get('/', include_in_schema=False)
async def root():
    return {"message": "ETA Predictor API", "docs": "/docs"}

@app.get('/health', response_model=HealthResponse, tags=['System'])
async def health_check():
    return HealthResponse(
        status='healthy' if predictor.is_loaded else 'degraded',
        model_loaded=predictor.is_loaded,
        api_version='1.0.0',
    )

@app.post('/predict', response_model=ETAResponse, tags=['Predictions'])
async def predict_eta(request: ETARequest):

    if not predictor.is_loaded:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail='Model is not loaded. Please try again later.'   
        )
    try:
        features = request.to_feature_vector()

        eta_min, ci_low, ci_high = predictor.predict(features)

        hours = int(eta_min // 60)
        minutes = int(eta_min % 60)
        if hours > 0:
            eta_human = f"{hours} hour{'s' if hours != 1 else ''} {minutes} minute{'s' if minutes != 1 else ''}"
        else:
            eta_human = f"{minutes} minute{'s' if minutes != 1 else ''}"

        return ETAResponse(
            eta_minutes=eta_min,
            eta_human_readable=eta_human,
            model_version=predictor.version,
            distance_km=request.distance_km,
            confidence_low=ci_low,
            confidence_high=ci_high,
            is_rush_hour=request.is_rush_hour
        )
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f'Error during prediction: {str(e)}',
        )
