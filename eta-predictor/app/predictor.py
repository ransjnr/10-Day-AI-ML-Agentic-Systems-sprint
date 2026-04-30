import joblib
import numpy as np
from pathlib import Path

MODEL_PATH = Path('models/eta_model_latest.joblib')

class ETAPredictor:
    def __init__(self):
        self.model = None
        self.model_version = 'unknown'

    def load(self, path: Path = MODEL_PATH):
        try:
            self.model = joblib.load(path)
            self.model_version = path.stem
            print(f'Model loaded from {path}')
            return True
        except FileNotFoundError:
            print(f'No model found at {path}. Please train a model first.')
            return False
        except Exception as e:
            print(f'Error loading model: {e}')
            return False

    @property
    def is_loaded(self) -> bool:
        return self.model is not None

    @property
    def version(self) -> str:
        return self.model_version

    def predict(self, feature_vector: list[float]) -> tuple[float, float, float]:
        if not self.is_loaded:
            raise RuntimeError('Model is not loaded. Call load() before predict().')
        
        X = np.array([feature_vector])
        eta = float(self.model.predict(X)[0])

        confidence_low = max(0, eta * 0.80)  # Dummy confidence interval
        confidence_high = eta * 1.20  # Dummy confidence interval

        return round(eta, 1), round(confidence_low, 1), round(confidence_high, 1)
        