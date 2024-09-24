import pandas as pd
import sweetviz as sv
from sklearn.preprocessing import StandardScaler
from sklearn.datasets import load_iris
import joblib

# Load the Iris dataset
iris = load_iris()
data = pd.DataFrame(iris.data, columns=iris.feature_names)
data['target'] = iris.target  # Add the target (0 = Setosa, 1 = Versicolor, 2 = Virginica)

# AutoEDA using Sweetviz
report = sv.analyze(data)
report.show_html('Iris_AutoEDA_Report.html')

# Separate features and target variable
X = data.drop(columns=['target'])
y = data['target']

# Feature Scaling
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Save the fitted scaler
joblib.dump(scaler, 'scaler.pkl')

import h2o
from h2o.automl import H2OAutoML

# Initialize H2O
h2o.init()

# Convert the Iris dataset to H2OFrame
h2o_data = h2o.H2OFrame(data)

# Set target as a factor (for classification)
h2o_data['target'] = h2o_data['target'].asfactor()

# Split the data into train and test sets
train, test = h2o_data.split_frame(ratios=[0.8], seed=42)

# Define AutoML and train
aml = H2OAutoML(max_runtime_secs=300, seed=42)
aml.train(y='target', training_frame=train)

# Get the best model
best_model = aml.leader
print(f"Best Model: {best_model}")

# Save the best model to a file
model_path = h2o.save_model(model=best_model, path="./best_model_h2o", force=True)
print(f"Model saved to: {model_path}")

from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GridSearchCV

# Define the model
rf = RandomForestClassifier(random_state=42)

# Define hyperparameters to tune
param_grid = {
    'n_estimators': [50, 100, 200],
    'max_depth': [5, 10, None]
}

# GridSearchCV
grid_search = GridSearchCV(rf, param_grid, cv=5)
grid_search.fit(X_scaled, y)

# Best Model
best_rf = grid_search.best_estimator_
print(f"Best RandomForest Model: {best_rf}")
# Save the best RandomForest model using joblib
joblib.dump(best_rf, 'best_random_forest_model.pkl')
print("Best RandomForest Model saved successfully.")


import lime
import lime.lime_tabular

# Convert test data to Pandas DataFrame
test_df = test.as_data_frame()
X_test = test_df.drop(columns=['target'])  # Drop the target column for explanation

# Use a small sample for LIME explanation
X_test_sample = X_test.sample(1, random_state=42)

# Initialize LIME explainer
explainer = lime.lime_tabular.LimeTabularExplainer(
    training_data=X_test.values,
    feature_names=iris.feature_names,
    class_names=iris.target_names,
    mode='classification'
)

# Explain a single instance
exp = explainer.explain_instance(
    X_test_sample.values[0],  # Single data point for explanation
    lambda x: best_model.predict(h2o.H2OFrame(pd.DataFrame(x, columns=iris.feature_names))).as_data_frame().values,  # Predict function wrapped for LIME
    num_features=4
)

# Show explanation
#exp.show_in_notebook(show_table=True)
exp.save_to_file('lime_explanation_iris.html')


from flask import Flask, request, jsonify
app = Flask(__name__)
import h2o
import pandas as pd
from sklearn.preprocessing import StandardScaler

# Initialize H2O
h2o.init()

# Load the trained model
#model_path = "./best_model_h2o"
#best_model = h2o.load_model(model_path)

# Assume these were the original feature names during training
feature_names = ['sepal length (cm)', 'sepal width (cm)', 'petal length (cm)', 'petal width (cm)']

# Initialize a scaler (it should be the same scaler used during training)
# scaler = StandardScaler()


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

    # Return the prediction as a JSON response
    return jsonify(prediction.as_data_frame().to_dict())


if __name__ == "__main__":
    app.run(debug=True)
