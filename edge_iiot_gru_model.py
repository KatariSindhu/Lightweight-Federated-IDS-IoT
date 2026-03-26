import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix

# 1. SETUP AND DATA LOADING
script_dir = os.path.dirname(os.path.abspath(__file__))
dataset_path = os.path.join(script_dir, '..', 'Dataset_Edge_IIoT', 'ML-EdgeIIoT-dataset.csv')
results_dir = os.path.join(script_dir, '..', 'Results')

print("Loading data for GRU training...")
df = pd.read_csv(dataset_path, low_memory=False)

# 2. ROBUST DATA CLEANING
# First, drop the specific columns we know are problematic
cols_to_drop = ['frame.time', 'ip.src_host', 'ip.dst_host', 'arp.src.proto_ipv4', 
                'arp.dst.proto_ipv4', 'http.file_data', 'http.request.full_uri', 
                'mqtt.msg', 'tcp.options', 'tcp.payload', 'http.request.uri.query', 
                'http.request.method', 'http.referer', 'http.user_agent']
df.drop(columns=[c for c in cols_to_drop if c in df.columns], inplace=True)

# Encode target label first
le = LabelEncoder()
df['target'] = le.fit_transform(df['Attack_type'])

# AUTOMATICALLY SELECT ONLY NUMERIC COLUMNS for X
# This prevents the "could not convert string to float" error permanently
X_numeric = df.select_dtypes(include=[np.number]).drop(columns=['target'])
y = df['target'].values

print(f"Features selected for training: {X_numeric.columns.tolist()}")

# Handle any remaining NaN values
X_numeric.fillna(0, inplace=True)

# 3. SCALING AND RESHAPING
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X_numeric)

# Reshape for GRU (samples, time_steps=1, features)
X_reshaped = X_scaled.reshape((X_scaled.shape[0], 1, X_scaled.shape[1]))

X_train, X_test, y_train, y_test = train_test_split(X_reshaped, y, test_size=0.2, random_state=42)

# Convert to PyTorch Tensors
X_train_t = torch.tensor(X_train, dtype=torch.float32)
y_train_t = torch.tensor(y_train, dtype=torch.long)
X_test_t = torch.tensor(X_test, dtype=torch.float32)
y_test_t = torch.tensor(y_test, dtype=torch.long)

train_loader = DataLoader(TensorDataset(X_train_t, y_train_t), batch_size=128, shuffle=True)

# 4. DEFINE LIGHTWEIGHT GRU MODEL
class LightweightGRU(nn.Module):
    def __init__(self, input_size, hidden_size, num_classes):
        super(LightweightGRU, self).__init__()
        self.gru = nn.GRU(input_size, hidden_size, batch_first=True)
        self.fc = nn.Linear(hidden_size, num_classes)
    
    def forward(self, x):
        _, h_n = self.gru(x)
        out = self.fc(h_n[-1])
        return out

model = LightweightGRU(input_size=X_reshaped.shape[2], hidden_size=32, num_classes=len(le.classes_))
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

# 5. TRAINING LOOP
epochs = 5
print(f"Starting training on {X_train.shape[0]} samples...")
train_losses = []

for epoch in range(epochs):
    model.train()
    total_loss = 0
    for batch_x, batch_y in train_loader:
        optimizer.zero_grad()
        outputs = model(batch_x)
        loss = criterion(outputs, batch_y)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
    
    avg_loss = total_loss / len(train_loader)
    train_losses.append(avg_loss)
    print(f"Epoch [{epoch+1}/{epochs}], Loss: {avg_loss:.4f}")

# 6. EVALUATION AND PLOTS
model.eval()
with torch.no_grad():
    y_pred = model(X_test_t)
    predicted_classes = torch.argmax(y_pred, dim=1).numpy()

# Save Confusion Matrix
plt.figure(figsize=(12, 10))
cm = confusion_matrix(y_test, predicted_classes)
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=le.classes_, yticklabels=le.classes_)
plt.title('Initial Result 1: Confusion Matrix for Local GRU Agent')
plt.tight_layout()
plt.savefig(os.path.join(results_dir, 'confusion_matrix.png'))

# Save Loss Curve
plt.figure(figsize=(8, 5))
plt.plot(range(1, epochs+1), train_losses, marker='o', color='red')
plt.title('Initial Result 2: GRU Model Convergence (Loss Curve)')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.grid(True)
plt.savefig(os.path.join(results_dir, 'loss_curve.png'))

accuracy = np.mean(predicted_classes == y_test) * 100
print(f"\n--- Review 2 Results ---")
print(f"Final Accuracy: {accuracy:.2f}%")
print(f"All artifacts saved in: {results_dir}")
plt.show()