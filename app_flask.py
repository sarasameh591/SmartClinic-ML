
from flask import Flask, jsonify, request

from ml_service import load_model, predict_diabetes


app = Flask(__name__)

model = load_model()


@app.get("/")
def home():
    return jsonify({
        "message": "Welcome to SmartClinic Diabetes API",
        "endpoint": "POST /predict/diabetes",
    })


@app.get("/health")
def health():
    return jsonify({
        "status": "ok",
        "model_loaded": model is not None,
    })


@app.post("/predict/diabetes")
def predict():
    try:
        data = request.get_json(silent=True)

        if not isinstance(data, dict):
            return jsonify({
                "error": "Request body must be a JSON object."
            }), 400

        features = data.get("features", data)

        if not isinstance(features, dict):
            return jsonify({
                "error": "features must be a JSON object."
            }), 400

        result = predict_diabetes(model, features)
        return jsonify(result), 200

    except ValueError as error:
        return jsonify({"error": str(error)}), 422

    except Exception:
        app.logger.exception("Prediction failed")
        return jsonify({
            "error": "Prediction failed. Check server logs."
        }), 500


if __name__ == "__main__":
    app.run(
        host="127.0.0.1",
        port=5001,
        debug=False,
    )