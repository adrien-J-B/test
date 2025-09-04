import torch
import torch.nn as nn
import numpy as np
import pandas as pd
import pickle
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
import os

# Define the model architecture
class BurnoutPredictor(nn.Module):
    def __init__(self, input_size):
        super().__init__()
        self.fc = nn.Sequential(
            nn.Linear(input_size, 16),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(16, 8),
            nn.ReLU(),
            nn.Linear(8, 1)
        )
    
    def forward(self, x):
        return self.fc(x)

# Create a more realistic dataset for training
def create_sample_data(n_samples=200):
    np.random.seed(42)
    
    # Create sample data with correlations that make sense
    data = {
        'attendance_rate': np.clip(np.random.normal(0.75, 0.15, n_samples), 0, 1),
        'avg_grade': np.clip(np.random.normal(65, 15, n_samples), 0, 100),
        'hours_studied_per_week': np.clip(np.random.normal(5, 2, n_samples), 0, 40),
        'sleep_hours': np.clip(np.random.normal(6.5, 1.5, n_samples), 0, 12),
        'exercise_per_week': np.clip(np.random.poisson(2, n_samples), 0, 7),
    }
    
    df = pd.DataFrame(data)
    
    # Add faculty with some distribution
    faculties = ['Science', 'Arts', 'Business']
    df['faculty'] = np.random.choice(faculties, n_samples, p=[0.5, 0.3, 0.2])
    
    # Create target variable based on some logic
    # Higher risk for low attendance, low grades, low sleep, low exercise
    risk_score = (
        (1 - df['attendance_rate']) * 0.3 +
        ((100 - df['avg_grade']) / 100) * 0.3 +
        ((8 - df['sleep_hours']) / 8) * 0.2 +
        ((3 - df['exercise_per_week']) / 3) * 0.2
    )
    
    df['at_risk'] = (risk_score > 0.4).astype(int)
    
    return df

def main():
    # Create and prepare the dataset
    df = create_sample_data(200)
    
    # One-hot encode faculty
    df_encoded = pd.get_dummies(df, columns=['faculty'], drop_first=False)
    
    # Define feature columns
    feature_cols = [
        'attendance_rate', 'avg_grade', 'hours_studied_per_week',
        'sleep_hours', 'exercise_per_week', 
        'faculty_Arts', 'faculty_Business', 'faculty_Science'
    ]
    
    # Ensure all columns are numeric
    for col in feature_cols:
        df_encoded[col] = pd.to_numeric(df_encoded[col], errors='coerce')
    
    X = df_encoded[feature_cols].values
    y = df_encoded['at_risk'].values
    
    # Convert to float32 to avoid PyTorch tensor issues
    X = X.astype(np.float32)
    
    # Split and scale the data
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    scaler = StandardScaler().fit(X_train)
    X_train_scaled = scaler.transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    # Create and train model
    model = BurnoutPredictor(X_train.shape[1])
    criterion = nn.BCEWithLogitsLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
    
    # Convert to tensors
    X_train_tensor = torch.tensor(X_train_scaled, dtype=torch.float32)
    y_train_tensor = torch.tensor(y_train, dtype=torch.float32).unsqueeze(1)
    
    # Train for a few epochs
    model.train()
    for epoch in range(100):
        optimizer.zero_grad()
        outputs = model(X_train_tensor)
        loss = criterion(outputs, y_train_tensor)
        loss.backward()
        optimizer.step()
        
        if epoch % 20 == 0:
            print(f'Epoch {epoch}, Loss: {loss.item()}')
    
    # Save model state dict instead of entire model (safer approach)
    torch.save(model.state_dict(), "model_state_dict.pt")
    
    # Save the model architecture info
    model_info = {
        'input_size': X_train.shape[1],
        'feature_cols': feature_cols,
        'faculty_options': ['Science', 'Arts', 'Business']
    }
    
    with open("model_info.pkl", "wb") as f:
        pickle.dump(model_info, f)
    
    # Save scaler
    with open("scaler.pkl", "wb") as f:
        pickle.dump(scaler, f)
    
    # Also save the sample dataset for testing
    df.to_csv("sample_student_data.csv", index=False)
    
    print("Model, scaler, and sample dataset created successfully!")
    print(f"Input shape: {X_train.shape[1]} features")
    print(f"Dataset sample saved as 'sample_student_data.csv'")
    
    # Verify files were created
    files = ['model_state_dict.pt', 'model_info.pkl', 'scaler.pkl', 'sample_student_data.csv']
    for file in files:
        if os.path.exists(file):
            print(f"✓ {file} created successfully")
        else:
            print(f"✗ {file} was not created")

if __name__ == "__main__":
    main()