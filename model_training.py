import os
import pickle
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout
from tensorflow.keras.callbacks import EarlyStopping

def custom_preprocess(df, meta):
    """
    Robust, version-agnostic transformation function that matches 
    scikit-learn's preprocessing behavior exactly.
    """
    scaled_df = df.copy()
    
    # 1. Scale numerical columns
    for col in meta['num_cols']:
        mean = meta['means'][col]
        std = meta['stds'][col]
        scaled_df[col] = (scaled_df[col] - mean) / (std if std > 0 else 1e-9)
        
    # 2. One-hot encode Geography (alphabetical sorting matches sklearn)
    geographies = ['France', 'Germany', 'Spain']
    for geo in geographies:
        scaled_df[f'Geography_{geo}'] = (scaled_df['Geography'] == geo).astype(float)
        
    # 3. One-hot encode Gender (alphabetical sorting matches sklearn)
    genders = ['Female', 'Male']
    for gen in genders:
        scaled_df[f'Gender_{gen}'] = (scaled_df['Gender'] == gen).astype(float)
        
    # Order columns exactly as expected by the ANN input layer
    ordered_cols = meta['num_cols'] + \
                   [f'Geography_{g}' for g in geographies] + \
                   [f'Gender_{g}' for g in genders]
                   
    return scaled_df[ordered_cols].values

def train_model():
    os.makedirs('models', exist_ok=True)
    
    # Load Data
    data_path = os.path.join('data', 'Churn_Modelling.csv')
    if not os.path.exists(data_path):
        raise FileNotFoundError(f"Dataset not found at {data_path}.")
        
    df = pd.read_csv(data_path)
    X = df.drop(columns=['RowNumber', 'CustomerId', 'Surname', 'Exited'])
    y = df['Exited'].values
    
    num_cols = ['CreditScore', 'Age', 'Tenure', 'Balance', 'NumOfProducts', 'HasCrCard', 'IsActiveMember', 'EstimatedSalary']
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
    
    # Calculate means and population std deviations (matching StandardScaler)
    means = {col: float(np.mean(X_train[col])) for col in num_cols}
    stds = {col: float(np.std(X_train[col])) for col in num_cols}
    
    # Save metadata as a plain dictionary
    meta = {
        'means': means,
        'stds': stds,
        'num_cols': num_cols
    }
    
    with open(os.path.join('models', 'preprocessor_meta.pkl'), 'wb') as f:
        pickle.dump(meta, f)
    print("Preprocessor metadata saved successfully.")
    
    # Transform datasets
    X_train_transformed = custom_preprocess(X_train, meta)
    
    # Build the ANN
    input_dim = X_train_transformed.shape[1]
    model = Sequential([
        Dense(32, activation='relu', input_dim=input_dim),
        Dropout(0.2),
        Dense(16, activation='relu'),
        Dropout(0.2),
        Dense(1, activation='sigmoid')
    ])
    
    model.compile(
        optimizer='adam',
        loss='binary_crossentropy',
        metrics=['accuracy']
    )
    
    early_stopping = EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True)
    
    print("Starting ANN model training...")
    model.fit(
        X_train_transformed, 
        y_train, 
        epochs=100, 
        batch_size=32, 
        validation_split=0.15,
        callbacks=[early_stopping],
        verbose=1
    )
    
    model.save(os.path.join('models', 'ann_model.keras'))
    print("Model training and saving completed.")

if __name__ == '__main__':
    train_model()