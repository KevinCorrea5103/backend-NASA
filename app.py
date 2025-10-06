from flask import Flask, request, jsonify
from flask_cors import CORS
import joblib
import numpy as np
import os

app = Flask(__name__)
CORS(app)

# Load your model and scaler
model = joblib.load("modelV1.joblib")
scaler = joblib.load("modelV1_scaler.joblib")

features = [
    'orbper', 'trandur', 'trandep', 'rade', 'insol', 'eqt', 'teff',
    'rad', 'logg', 'depth_per_duration', 'insol_per_radius2',
    'radius_teff_ratio', 'insol_orbper_ratio', 'logg_teff_product'
]

@app.route("/predict", methods=["POST"])
def predict():
    try:
        data = request.get_json()
        X = np.array([[data[f] for f in features]])
        X_scaled = scaler.transform(X)
        prob = model.predict_proba(X_scaled)[0, 1]
        label = "Exoplanet" if prob > 0.5 else "Not Exoplanet"
        return jsonify({
            "prediction": label,
            "probability": round(float(prob), 3)
        })
    except Exception as e:
        return jsonify({"error": str(e)})

@app.route("/", methods=["GET"])
def health_check():
    return jsonify({"status": "API is running"})

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    app.run(debug=False, host="0.0.0.0", port=port)