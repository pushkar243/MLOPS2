from flask import Flask, request, jsonify
import pandas as pd
from sklearn.preprocessing import StandardScaler
import joblib

app = Flask(__name__)

# Load the trained Random Forest model
model_path = "best_random_forest_model.pkl"  # The path to your saved RandomForest model
best_rf = joblib.load(model_path)

# Load the fitted scaler
scaler = joblib.load('scaler.pkl')

# Class label mapping for Iris dataset (since it's classification)
class_labels = ['Setosa', 'Versicolor', 'Virginica']

# Assume these were the original feature names during training
feature_names = ['sepal length (cm)', 'sepal width (cm)', 'petal length (cm)', 'petal width (cm)']

@app.route('/predict', methods=['POST'])
def predict():
    # Get the input data from the POST request
    data = request.json['data']

    # Convert the input data to a DataFrame
    data_df = pd.DataFrame(data, columns=feature_names)

    # Apply the same scaling used during training
    data_scaled = scaler.transform(data_df)

    # Make predictions using the loaded RandomForest model
    predictions = best_rf.predict(data_scaled)

    # Convert numeric predictions to class labels
    prediction_labels = [class_labels[int(pred)] for pred in predictions]

    # Return the predictions as a JSON response
    return jsonify({"predictions": prediction_labels})

if __name__ == "__main__":
    app.run(host='0.0.0.0', port=5000, debug=True)
