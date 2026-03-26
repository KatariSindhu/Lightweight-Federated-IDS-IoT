# Lightweight Federated IDS for Smart Home IoT 

## 📌 Project Overview
A lightweight Intrusion Detection System (IDS) developed using **Gated Recurrent Units (GRU)** for IoT environments. It achieves **96.5% accuracy** on the **Edge-IIoTset** while maintaining low latency and memory usage.

---

## 👤 Author Information
* **Name:** Sindhu  
* **Reg No:** 22BCE8991  
* **Institution:** VIT-AP University  
* **Project:** Senior Design Project (Review 2)

---

## 📁 Project Structure
```text
├── Code/
│   ├── edge_iiot_gru_model.py     # Model training & core results
│   ├── edge_iiot_extra_metrics.py # Benchmarking & 10 extra graphs
├── Dataset_Edge_IIoT/
│   └── ML-EdgeIIoT-dataset.csv    # (Add dataset here)
├── Results/                       # 12 Generated PNG graphs
└── README.md
```
---

## 🛠️ Setup & Execution
1. Requirements
```text
Bash
pip install torch torchvision torchaudio --index-url [https://download.pytorch.org/whl/cpu](https://download.pytorch.org/whl/cpu)
pip install pandas numpy matplotlib seaborn scikit-learn
```
3. Run the Project

**Train Model:** Run python Code/edge_iiot_gru_model.py to generate the Confusion Matrix and Loss Curve.

**Generate Metrics:** Run python Code/edge_iiot_extra_metrics.py to generate the remaining 10 comparative graphs.

---

## 📊 Summary of Results
```text
Accuracy: 96.5%
Detection Latency: 0.12ms/packet
Model Size: 1.2MB
Total Artifacts: 12 Technical Graphs (saved in /Results)
```
