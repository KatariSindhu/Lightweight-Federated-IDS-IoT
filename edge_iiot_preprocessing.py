import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os
import numpy as np
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.model_selection import train_test_split

# --- FIXING PATHS FOR YOUR FOLDER STRUCTURE ---
# This looks for the file exactly where you have it in your explorer
script_dir = os.path.dirname(os.path.abspath(__file__))
dataset_path = os.path.join(script_dir, '..', 'Dataset_Edge_IIoT', 'ML-EdgeIIoT-dataset.csv')
results_dir = os.path.join(script_dir, '..', 'Results')

if not os.path.exists(results_dir):
    os.makedirs(results_dir)

print(f"Loading dataset from: {dataset_path}")

try:
    # 1. LOAD DATA
    df = pd.read_csv(dataset_path, low_memory=False)
    print("--- Dataset Loaded Successfully ---")
    
    # 2. GENERATE VISUALIZATION (PPT Slide 2 Requirement)
    plt.figure(figsize=(12, 6))
    sns.countplot(data=df, x='Attack_type', order=df['Attack_type'].value_counts().index, palette='viridis')
    plt.xticks(rotation=45)
    plt.title('Attack Type Distribution in Edge-IIoTset')
    plt.tight_layout()
    chart_file = os.path.join(results_dir, 'class_distribution.png')
    plt.savefig(chart_file)
    print(f"Chart saved to: {chart_file}")
    plt.show()

    # 3. DATA CLEANING (Review 2: 70% Completion Step)
    cols_to_drop = ['frame.time', 'ip.src_host', 'ip.dst_host', 'arp.src.proto_ipv4', 
                    'arp.dst.proto_ipv4', 'http.file_data', 'http.request.full_uri', 
                    'mqtt.msg', 'tcp.options', 'tcp.payload']
    
    df.drop(columns=[c for c in cols_to_drop if c in df.columns], inplace=True)
    df.dropna(inplace=True)

    # 4. ENCODING & SCALING
    le = LabelEncoder()
    df['target'] = le.fit_transform(df['Attack_type'])
    X = df.drop(['Attack_type', 'target'], axis=1)
    y = df['target']

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    # 5. TRAIN-TEST SPLIT (80/20)
    X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)

    print("\n--- Preprocessing Complete for Review 2 ---")
    print(f"Total Training Samples: {X_train.shape[0]}")
    print(f"Number of Features: {X_train.shape[1]}")

except Exception as e:
    print(f"An error occurred: {e}")