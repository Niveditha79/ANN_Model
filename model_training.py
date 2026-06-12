import os
import pickle
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout
from tensorflow.keras.callbacks import EarlyStopping

def train_model():
    # Ensure directories exist
    os.makedirs('models', exist_ok=True)
    
    # 1. Load Data
    data_path = os.path.join('data', 'Churn_Modelling.csv')
    if not os.path.exists(data_path):
        raise FileNotFoundError(f"Dataset not found at {data_path}. Please place the CSV file there.")
        
    df = pd.read_csv(data_path)
    
    # 2. Separate Features and Target
    X = df.drop(columns=['RowNumber', 'CustomerId', 'Surname', 'Exited'])
    y = df['Exited'].values
    
    # Define categorical and numerical columns
    num_cols = ['CreditScore', 'Age', 'Tenure', 'Balance', 'NumOfProducts', 'HasCrCard', 'IsActiveMember', 'EstimatedSalary']
    cat_cols = ['Geography', 'Gender']
    
    # 3. Create Preprocessing Pipeline
    preprocessor = ColumnTransformer(
        transformers=[
            ('num', StandardScaler(), num_cols),
            ('cat', OneHotEncoder(handle_unknown='ignore', sparse_output=False), cat_cols)
        ]
    )
    
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
    
    # Fit and transform
    X_train_transformed = preprocessor.fit_transform(X_train)
    X_test_transformed = preprocessor.transform(X_test)
    
    # Save the preprocessor
    with open(os.path.join('models', 'preprocessor.pkl'), 'wb') as f:
        pickle.dump(preprocessor, f)
    print("Preprocessor saved successfully.")
    
    # 4. Build the ANN Model
    input_dim = X_train_transformed.shape[1]
    
    model = Sequential([
        Dense(32, activation='relu', input_dim=input_dim),
        Dropout(0.2),
        Dense(16, activation='relu'),
        Dropout(0.2),
        Dense(1, activation='sigmoid')
    ])
    
    # Compile model
    model.compile(
        optimizer='adam',
        loss='binary_crossentropy',
        metrics=['accuracy']
    )
    
    # Define Early Stopping
    early_stopping = EarlyStopping(
        monitor='val_loss', 
        patience=10, 
        restore_best_weights=True
    )
    
    # Train the model
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
    
    # Save the model
    model_save_path = os.path.join('models', 'ann_model.keras')
    model.save(model_save_path)
    print(f"Model saved successfully to {model_save_path}")

if __name__ == '__main__':
    train_model()