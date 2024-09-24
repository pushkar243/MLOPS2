from flask import Flask, request, jsonify
import h2o
import pandas as pd
from sklearn.preprocessing import StandardScaler
import joblib

app = Flask(__name__)

# Initialize H2O
h2o.init()

# Class label mapping for Iris dataset
class_labels = ['Setosa', 'Versicolor', 'Virginica']

# Load the trained model
model_path = "./best_model_h2o/DeepLearning_grid_1_AutoML_1_20240918_23807_model_1"
best_model = h2o.load_model(model_path)

# Load the fitted scaler
scaler = joblib.load('scaler.pkl')

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

    # Convert to H2OFrame for prediction
    data_h2o = h2o.H2OFrame(data_scaled, column_names=feature_names)

    # Make predictions using the loaded model
    prediction = best_model.predict(data_h2o)

    # Convert numeric predictions to class labels
    prediction_df = prediction.as_data_frame()
    prediction_df['predict'] = prediction_df['predict'].map(lambda x: class_labels[int(x)])

    # Return the prediction as a JSON response
    return jsonify(prediction.as_data_frame().to_dict())
    #return jsonify(prediction_df.to_dict())

if __name__ == "__main__":
    app.run(debug=True)
