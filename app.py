from flask import Flask, render_template, request
import pickle
import numpy as np

app = Flask(__name__)

# Load model and scaler
model = pickle.load(open("app/model.pkl", "rb"))
scaler = pickle.load(open("app/scaler.pkl", "rb"))

@app.route('/')
def home():
    return render_template("index.html")

@app.route('/predict', methods=["POST"])
def predict():
    # Get form data
    sepal_length = float(request.form['sepal_length'])
    sepal_width = float(request.form['sepal_width'])
    petal_length = float(request.form['petal_length'])
    petal_width = float(request.form['petal_width'])

    # Prepare input
    features = np.array([[sepal_length, sepal_width, petal_length, petal_width]])
    scaled_features = scaler.transform(features)

    # Predict
    prediction = model.predict(scaled_features)[0]

    # Map target number to species name (optional)
    species_map = {0: "Setosa", 1: "Versicolor", 2: "Virginica"}
    species = species_map.get(prediction, "Unknown")

    return render_template("index.html", prediction=f"Predicted Species: {species}")

if __name__ == '__main__':
    app.run(debug=True)
