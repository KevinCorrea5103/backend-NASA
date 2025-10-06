from flask import Flask, request, jsonify
from flask_cors import CORS
import joblib
import numpy as np

app = Flask(__name__)
CORS(app)  # Allow browser to communicate with this program

# Load your model and scaler
model = joblib.load("modelV1.joblib")
scaler = joblib.load("modelV1_scaler.joblib")

# These are the columns expected by the model
features = [
    'orbper',         # Orbital period (days)
    'trandur',        # Transit duration (hours)
    'trandep',        # Transit depth (ppm)
    'rade',           # Planetary radius (R⊕)
    'insol',          # Insolation (S⊕)
    'eqt',            # Equilibrium temperature (K)
    'teff',           # Stellar temperature (K)
    'rad',            # Stellar radius (R☉)
    'logg',           # Surface gravity log(g)
    'depth_per_duration',   # trandep / trandur
    'insol_per_radius2',    # insol / rade^2
    'radius_teff_ratio',    # rade / teff
    'insol_orbper_ratio',   # insol / orbper
    'logg_teff_product'     # logg * teff
]

@app.route("/predict", methods=["POST"])
def predict():
    try:
        data = request.get_json()  # Data coming from the web
        X = np.array([[data[f] for f in features]])  # Convert to a list of numbers (14 features)
        X_scaled = scaler.transform(X)  # Scale the values
        prob = model.predict_proba(X_scaled)[0, 1]  # Calculate probability
        label = "Exoplanet" if prob > 0.5 else "Not Exoplanet"
        return jsonify({
            "prediction": label,
            "probability": round(float(prob), 3)
        })
    except Exception as e:
        return jsonify({"error": str(e)})

if __name__ == "__main__":
    app.run(debug=True, host="0.0.0.0", port=5000)


