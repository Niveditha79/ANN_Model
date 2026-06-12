import os
import pickle
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score
import tensorflow as tf
from model_training import custom_preprocess  # Import transformation function

def evaluate_saved_model():
    data_path = os.path.join('data', 'Churn_Modelling.csv')
    df = pd.read_csv(data_path)
    X = df.drop(columns=['RowNumber', 'CustomerId', 'Surname', 'Exited'])
    y = df['Exited'].values
    
    _, X_test, _, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
    
    # Load metadata dictionary and neural network
    with open(os.path.join('models', 'preprocessor_meta.pkl'), 'rb') as f:
        meta = pickle.load(f)
        
    model = tf.keras.models.load_model(os.path.join('models', 'ann_model.keras'))
    
    # Transform test set
    X_test_transformed = custom_preprocess(X_test, meta)
    
    # Evaluate
    y_pred_prob = model.predict(X_test_transformed)
    y_pred = (y_pred_prob > 0.5).astype(int).flatten()
    
    print("\n--- Model Evaluation Report ---")
    print(classification_report(y_test, y_pred))
    print(f"ROC-AUC Score: {roc_auc_score(y_test, y_pred_prob):.4f}")

if __name__ == '__main__':
    evaluate_saved_model()