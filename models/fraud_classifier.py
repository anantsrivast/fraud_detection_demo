import joblib
from models.transaction import Transaction

class FraudClassifier:
    def __init__(self, model_path: str):
        self.model = joblib.load(model_path)
        self.is_trained = True

    def predict(self, transaction: Transaction):
        features = [
            transaction.amount,
            transaction.timestamp.hour,
        ]
        prob = self.model.predict_proba([features])[0][1]
        return prob > 0.5, prob
