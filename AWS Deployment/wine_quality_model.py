from __future__ import print_function

import argparse
import os
import pandas as pd

from sklearn.externals import joblib
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    # Sagemaker specific arguments. Defaults are set in the environment variables.

    # Save checkpoints and graphs
    output_data_dir = os.environ.get('SM_OUTPUT_DATA', '/default/output/directory')
    parser.add_argument('--output-data-dir', type=str, default=output_data_dir)


    # Save model artifacts
    parser.add_argument('--model-dir', type=str, default=os.environ['SM_MODEL_DIR'])

    # Train data
    parser.add_argument('--train', type=str, default=os.environ['SM_CHANNEL_TRAIN'])

    args = parser.parse_args()

    # Load the data
    file = os.path.join(args.train, "wine_quality.csv")
    df = pd.read_csv(file, engine="python")

    # Separate Features from Label
    X = df.drop("quality", axis=1)
    y = df["quality"]

    # Split into training & test
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20, random_state=88)

    # Train model
    model = LinearRegression()
    model.fit(X_train, y_train)

    # Print the coefficients of the trained regressor and save the model
    joblib.dump(model, os.path.join(args.model_dir, "model.joblib"))

# Function to load the model for inference
def model_fn(model_dir):
    model = joblib.load(os.path.join(model_dir, "model.joblib"))
    return model