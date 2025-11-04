# for data manipulation
import pandas as pd
import sklearn
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import make_column_transformer
from sklearn.pipeline import make_pipeline
# for model training, tuning, and evaluation
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import accuracy_score, classification_report, recall_score, f1_score
# for model serialization
import joblib
# for creating a folder
import os
# for hugging face space authentication to upload files
from huggingface_hub import login, HfApi, create_repo
from huggingface_hub.utils import RepositoryNotFoundError, HfHubHTTPError

from datetime import datetime

api = HfApi()

Xtrain_path = "hf://datasets/josequinonez/PIMA-Diabetes-Prediction-FastAPI/Xtrain.csv"                    # enter the Hugging Face username here
Xtest_path = "hf://datasets/josequinonez/PIMA-Diabetes-Prediction-FastAPI/Xtest.csv"                      # enter the Hugging Face username here
ytrain_path = "hf://datasets/josequinonez/PIMA-Diabetes-Prediction-FastAPI/ytrain.csv"                    # enter the Hugging Face username here
ytest_path = "hf://datasets/josequinonez/PIMA-Diabetes-Prediction-FastAPI/ytest.csv"                      # enter the Hugging Face username here

Xtrain = pd.read_csv(Xtrain_path)
Xtest = pd.read_csv(Xtest_path)
ytrain = pd.read_csv(ytrain_path)
ytest = pd.read_csv(ytest_path)


# scale numeric features
numeric_features = [
    'preg',
    'plas',
    'pres',
    'skin',
    'test',
    'mass',
    'pedi',
    'age'
]


# Preprocessing pipeline
preprocessor = make_column_transformer(
    (StandardScaler(), numeric_features)
)

# Define GB model
gb_model = GradientBoostingClassifier(random_state=42)

# Define hyperparameter grid
param_grid = {
    'gradientboostingclassifier__n_estimators': [75, 100, 125],
    'gradientboostingclassifier__max_depth': [2, 3, 4],
    'gradientboostingclassifier__subsample': [0.5, 0.6]
}

# Create pipeline
model_pipeline = make_pipeline(preprocessor, gb_model)

# Grid search with cross-validation
grid_search = GridSearchCV(model_pipeline, param_grid, cv=5, scoring='recall', n_jobs=-1)
grid_search.fit(Xtrain, ytrain)


# Best model
best_model = grid_search.best_estimator_
print("Best Params:\n", grid_search.best_params_)

# Predict on training set
y_pred_train = best_model.predict(Xtrain)

# Predict on test set
y_pred_test = best_model.predict(Xtest)

# Evaluation
print("\nTraining Classification Report:")
print(classification_report(ytrain, y_pred_train))

print("\nTest Classification Report:")
print(classification_report(ytest, y_pred_test))

# Calculate and save metrics
test_recall = recall_score(ytest, y_pred_test)
test_f1 = f1_score(ytest, y_pred_test)

metrics_data = {
    'Date': [datetime.now().strftime("%Y-%m-%d")],
    'Test Recall': [test_recall],
    'Test F1-score': [test_f1]
}
metrics_df = pd.DataFrame(metrics_data)

metrics_filename = f"metrics_{datetime.now().strftime('%Y%m%d')}.csv"
metrics_df.to_csv(metrics_filename, index=False)

print(f"\nMetrics saved to {metrics_filename}")


latest_name = "best_pima_diabetes_model_latest.joblib"
current_date = datetime.now().strftime("%Y%m%d")
date_name = f"best_pima_diabetes_model_{current_date}.joblib"


# Save best model
joblib.dump(best_model, latest_name)
joblib.dump(best_model, date_name)

# Upload to Hugging Face
repo_id = "josequinonez/PIMA-Diabetes-Prediction-FastAPI"                                         # enter the Hugging Face username here
repo_type = "model"

api = HfApi(token=os.getenv("HF_TOKEN"))

# Step 1: Check if the space exists
try:
    api.repo_info(repo_id=repo_id, repo_type=repo_type)
    print(f"Model Space '{repo_id}' already exists. Using it.")
except RepositoryNotFoundError:
    print(f"Model Space '{repo_id}' not found. Creating new space...")
    create_repo(repo_id=repo_id, repo_type=repo_type, private=False)
    print(f"Model Space '{repo_id}' created.")

# create_repo("best_machine_failure_model", repo_type="model", private=False)
api.upload_file(
    path_or_fileobj=latest_name,
    path_in_repo=latest_name,
    repo_id=repo_id,
    repo_type=repo_type,
)

api.upload_file(
    path_or_fileobj=date_name,
    path_in_repo=date_name,
    repo_id=repo_id,
    repo_type=repo_type,
)
