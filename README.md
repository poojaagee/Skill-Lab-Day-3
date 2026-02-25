# Machine Learning Model Training and Deployment using FastAPI

## 📌 Project Overview
This project demonstrates an end-to-end Machine Learning workflow that includes
data handling, model training, experiment tracking, API development, and deployment
preparation using modern ML tools.

The system trains a classification model on the Iris dataset and exposes the trained
model through a REST API built with FastAPI for real-time predictions.


## 🛠️ Technologies Used
- Python 3.x
- Pandas
- Scikit-learn
- FastAPI
- MLflow
- Joblib
- Docker
- GitHub

## 📂 Project Structure
├── app.py # FastAPI application for model inference
├── train.py # Model training script
├── iris_data.csv # Dataset used for training
├── model.pkl # Trained ML model
├── mlflow.db # MLflow experiment tracking database
├── Dockerfile # Docker configuration file
├── .dvcignore # Data version control ignore file
├── .github/workflows # CI/CD workflow files

🔹 Project Outline / Steps
Step 1: Dataset Collection
Used the Iris dataset (iris_data.csv) containing flower measurements and species labels.
This dataset is used for supervised classification.

Step 2: Data Preprocessing
Loaded the dataset using Pandas.
Separated input features (sepal length, sepal width, petal length, petal width) and target labels.
Prepared the data in a format suitable for model training.

Step 3: Model Training
Implemented a training script (train.py).
Trained a machine learning classification model on the Iris dataset.
Evaluated the model performance.
Saved the trained model as a serialized file (model.pkl) using joblib.

Step 4: Experiment Tracking
Used MLflow for tracking experiments.
Stored experiment metadata such as parameters, metrics, and artifacts.
Maintained experiment data using mlflow.db.

Step 5: API Development
Built a REST API using FastAPI (app.py).
Loaded the trained model (model.pkl) into the API.
Implemented endpoints:
/health → checks whether the API is running
/predict → accepts input data and returns model predictions

Step 6: Model Inference
User sends feature values in JSON format to the /predict endpoint.
API converts input into a DataFrame
Model predicts the class label.
Prediction result is returned as a JSON response.

Step 7: Containerization
Created a Dockerfile to containerize the application.
Ensured consistent execution across different systems.
Simplified deployment and portability.

Step 8: Version Control & Automation
Used GitHub for version control
Maintained structured project files.
Included .github/workflows for automation (CI/CD readiness).
Used .dvcignore to manage data versioning efficiently.

FASTAPI INPUT

![FAST API INPUT](https://github.com/poojaagee/Skill-Lab-Day-3/blob/main/FAST%20API%20INPUT.jpg?raw=true)

FASTAPI OUTPUT

![FAST API OUTPUT](https://github.com/poojaagee/Skill-Lab-Day-3/blob/main/FAST%20API%20OUTPUT.jpg?raw=true)


