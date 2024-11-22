import os
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import classification_report, accuracy_score, mean_squared_error, r2_score, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns
import io
import base64
from flask import Flask, render_template

# Initialize Flask app
app = Flask(__name__)

# Check if Flask should be started based on environment variable
run_flask = os.getenv('RUN_FLASK', 'true').lower() == 'true'

if run_flask:
    @app.route('/')
    def home():
        # Load and prepare the dataset
        file_path = 'data/DevOps AWS Azure Effectiveness Deployment.csv'
        data = pd.read_csv(file_path)

        # Step 1: Data Preparation for Logistic Regression and SVM
        # Define the target variable based on the median of 'DevOps Efficiency Score'
        median_efficiency = data['DevOps Efficiency Score'].median()
        data['High_Efficiency'] = (data['DevOps Efficiency Score'] >= median_efficiency).astype(int)

        # Features for classification models
        X_classification = data.drop(columns=['Organization Name', 'DevOps Efficiency Score', 'High_Efficiency'])
        y_classification = data['High_Efficiency']

        # Step 2: Data Preparation for Random Forest
        # Encoding categorical variables if needed
        data['Organization Name'] = data['Organization Name'].astype('category').cat.codes

        # Define the target variable and features for Random Forest
        y_rf = data['Cost Efficiency ($)']  # Target variable for regression
        X_rf = data.drop(columns=['Cost Efficiency ($)', 'Organization Name'])

        # Step 3: Split the data into training and testing sets for both models
        # Split for classification models (Logistic Regression & SVM)
        X_train_class, X_test_class, y_train_class, y_test_class = train_test_split(X_classification, y_classification, test_size=0.2, random_state=42)

        # Split for Random Forest
        X_train_rf, X_test_rf, y_train_rf, y_test_rf = train_test_split(X_rf, y_rf, test_size=0.2, random_state=42)

        # Step 4: Standardize the features for classification models
        scaler = StandardScaler()
        X_train_class_scaled = scaler.fit_transform(X_train_class)
        X_test_class_scaled = scaler.transform(X_test_class)

        # Step 5: Logistic Regression Model Training and Evaluation
        logistic_model = LogisticRegression(random_state=42)
        logistic_model.fit(X_train_class_scaled, y_train_class)
        y_pred_logistic = logistic_model.predict(X_test_class_scaled)

        accuracy_logistic = accuracy_score(y_test_class, y_pred_logistic)
        logistic_report = classification_report(y_test_class, y_pred_logistic)
        conf_matrix_logistic = confusion_matrix(y_test_class, y_pred_logistic)

        # Step 6: SVM Model Training and Evaluation
        svm_model = SVC(kernel='linear', random_state=42)
        svm_model.fit(X_train_class_scaled, y_train_class)
        y_pred_svm = svm_model.predict(X_test_class_scaled)

        accuracy_svm = accuracy_score(y_test_class, y_pred_svm)
        svm_report = classification_report(y_test_class, y_pred_svm)
        conf_matrix_svm = confusion_matrix(y_test_class, y_pred_svm)

        # Step 7: Random Forest Model Training and Evaluation
        rf_model = RandomForestRegressor(n_estimators=100, random_state=42)
        rf_model.fit(X_train_rf, y_train_rf)
        y_pred_rf = rf_model.predict(X_test_rf)

        mse_rf = mean_squared_error(y_test_rf, y_pred_rf)
        r2_rf = r2_score(y_test_rf, y_pred_rf)

        # Collect the model results
        model_results = {
            'Logistic Regression': {
                'accuracy': accuracy_logistic,
                'classification_report': logistic_report,
                'confusion_matrix': conf_matrix_logistic
            },
            'SVM': {
                'accuracy': accuracy_svm,
                'classification_report': svm_report,
                'confusion_matrix': conf_matrix_svm
            },
            'Random Forest': {
                'mse': mse_rf,
                'r2': r2_rf
            }
        }

        # Generate confusion matrix image for Logistic Regression
        fig_logistic = plt.figure(figsize=(8, 6))
        sns.heatmap(conf_matrix_logistic, annot=True, fmt='d', cmap='Blues', xticklabels=['Low Efficiency', 'High Efficiency'], yticklabels=['Low Efficiency', 'High Efficiency'])
        plt.xlabel('Predicted')
        plt.ylabel('Actual')
        plt.title('Confusion Matrix for Logistic Regression')
        img_logistic = io.BytesIO()
        fig_logistic.savefig(img_logistic, format='png')
        img_logistic.seek(0)
        img_logistic_base64 = base64.b64encode(img_logistic.getvalue()).decode()

        # Generate confusion matrix image for SVM
        fig_svm = plt.figure(figsize=(8, 6))
        sns.heatmap(conf_matrix_svm, annot=True, fmt='d', cmap='Blues', xticklabels=['Low Efficiency', 'High Efficiency'], yticklabels=['Low Efficiency', 'High Efficiency'])
        plt.xlabel('Predicted')
        plt.ylabel('Actual')
        plt.title('Confusion Matrix for SVM')
        img_svm = io.BytesIO()
        fig_svm.savefig(img_svm, format='png')
        img_svm.seek(0)
        img_svm_base64 = base64.b64encode(img_svm.getvalue()).decode()

        # Generate Random Forest plot (Actual vs Predicted)
        fig_rf = plt.figure(figsize=(10, 6))
        sns.scatterplot(x=y_test_rf, y=y_pred_rf)
        plt.xlabel('Actual Cost Efficiency')
        plt.ylabel('Predicted Cost Efficiency')
        plt.title('Actual vs Predicted Cost Efficiency')
        plt.plot([0, max(y_test_rf)], [0, max(y_test_rf)], 'r--')
        img_rf = io.BytesIO()
        fig_rf.savefig(img_rf, format='png')
        img_rf.seek(0)
        img_rf_base64 = base64.b64encode(img_rf.getvalue()).decode()

        return render_template('index.html', model_results=model_results, img_logistic=img_logistic_base64, img_svm=img_svm_base64, img_rf=img_rf_base64)

# Only run Flask if 'RUN_FLASK' is true
if run_flask:
    app.run(host="0.0.0.0", port=5000, debug=True)
else:
    print("Flask server not started. Running script in background mode.")
