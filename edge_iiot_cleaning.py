import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.model_selection import train_test_split

# 1. Load Dataset
print("Loading dataset...")
df = pd.read_csv('../Dataset_Edge_IIoT/ML-EdgeIIoT-dataset.csv', low_memory=False)

# 2. Drop irrelevant columns (common in Edge-IIoTset)
# These are often strings or timestamps that the GRU cannot process
columns_to_drop = ['frame.time', 'ip.src_host', 'ip.dst_host', 'arp.src.proto_ipv4', 'arp.dst.proto_ipv4', 'http.file_data', 'http.request.full_uri', 'icmp.transmit_timestamp', 'http.request.uri.query', 'tcp.options', 'tcp.payload', 'tcp.srcport', 'tcp.dstport', 'udp.port', 'mqtt.msg']
df.drop(columns=[col for col in columns_to_drop if col in df.columns], inplace=True)

# 3. Handle Missing Values
df.dropna(inplace=True)

# 4. Encode the 'Attack_type' for training
le = LabelEncoder()
df['Attack_type_encoded'] = le.fit_transform(df['Attack_type'])

# 5. Feature Scaling (Mandatory for Neural Networks like GRU)
X = df.drop(['Attack_type', 'Attack_type_encoded'], axis=1)
y = df['Attack_type_encoded']

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# 6. Train-Test Split (80/20)
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)

print(f"Preprocessing Complete!")
print(f"Training samples: {X_train.shape[0]}")
print(f"Testing samples: {X_test.shape[0]}")