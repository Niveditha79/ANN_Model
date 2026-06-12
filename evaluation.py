import os
import pickle
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score, roc_curve
import tensorflow as tf

def evaluate_saved_model():
    # Load Data
    data_path = os.path.join('data', 'Churn_Modelling.csv')
    df = pd.read_csv(data_path)
    X = df.drop(columns=['RowNumber', 'CustomerId', 'Surname', 'Exited'])
    y = df['Exited'].values
    
    # Split to get identical test set
    _, X_test, _, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
    
    # Load Preprocessor and Model
    with open(os.path.join('models', 'preprocessor.pkl'), 'rb') as f:
        preprocessor = pickle.load(f)
        
    model = tf.keras.models.load_model(os.path.join('models', 'ann_model.keras'))
    
    # Transform test features
    X_test_transformed = preprocessor.transform(X_test)
    
    # Generate predictions
    y_pred_prob = model.predict(X_test_transformed)
    y_pred = (y_pred_prob > 0.5).astype(int).flatten()
    
    # Print Performance Metrics
    print("\n--- Model Evaluation Report ---")
    print(classification_report(y_test, y_pred))
    
    auc = roc_auc_score(y_test, y_pred_prob)
    print(f"ROC-AUC Score: {auc:.4f}")
    
    # Plot Confusion Matrix
    cm = confusion_matrix(y_test, y_pred)
    plt.figure(figsize=(6, 4))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=['Stayed', 'Exited'], yticklabels=['Stayed', 'Exited'])
    plt.title('Confusion Matrix')
    plt.ylabel('Actual Class')
    plt.xlabel('Predicted Class')
    plt.tight_layout()
    plt.savefig('evaluation_confusion_matrix.png')
    print("Saved confusion matrix visualization as 'evaluation_confusion_matrix.png'.")

if __name__ == '__main__':
    evaluate_saved_model()