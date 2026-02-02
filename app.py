from flask import Flask, jsonify, request
import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression


data = {
    "year": [2021,2022,2023 ],
    "temperature": [14.2,15.1,16.0,17.3]
}

df = pd.DataFrame(data)

X = df[["year"]]
y = df["temperature"]

model = LinearRegression()
model.fit(X, y)

app = Flask(__name__)

@app.route("/")
def home():
    return jsonify({"message": "EnviroCast Risk AI API is running"})

@app.route("/predict", methods=["GET"])
def predict():
    year = request.args.get("year")

    if not year:
        return jsonify({"error": "Year parameter is required"}), 400

    try:
        year = int(year)
    except ValueError:
        return jsonify({"error": "Year must be an integer"}), 400

    prediction = model.predict(np.array([[year]]))[0]

    if prediction < 15:
        risk = "Low"
    elif prediction < 16:
        risk = "Medium"
    else:
        risk = "High"

    return jsonify({
        "year": year,
        "predicted_temperature": round(prediction, 2),
        "risk_level": risk
    })

if __name__ == "__main__":
    app.run(debug=True)

Add Flask AI prediction backend

