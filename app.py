from flask import Flask, render_template, request
import joblib
import numpy as np
import os


# Initialize the Flask app
app = Flask(__name__)

# Load the trained model (ensure the pickle file is in your project folder)
model = joblib.load("wine_quality_model.pkl")

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    # Get the input values from the form
    var_1 = float(request.form['fixed_acidity'])
    var_2 = float(request.form['volatile_acidity'])
    var_3 = float(request.form['citric_acid'])
    var_4 = float(request.form['residual_sugar'])
    var_5 = float(request.form['chlorides'])
    var_6 = float(request.form['free_sulfur_dioxide'])
    var_7 = float(request.form['total_sulfur_dioxide'])
    var_8 = float(request.form['density'])
    var_9 = float(request.form['pH'])
    var_10 = float(request.form['sulphates'])
    var_11 = float(request.form['alcohol'])

    # Create an array of input data
    input_data = np.array([[var_1, var_2, var_3, var_4, var_5, var_6, var_7, var_8, var_9, var_10, var_11]])

    # Make a prediction using the model
    prediction = model.predict(input_data)

    # Clip the predicted value to be between 0 and 10
    result = np.clip(prediction[0], 0, 10)

    return render_template('index.html', prediction=result)

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=int(os.environ.get("PORT", 5002)))
