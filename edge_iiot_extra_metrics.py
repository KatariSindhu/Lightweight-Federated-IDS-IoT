import matplotlib.pyplot as plt
import numpy as np
import os
import seaborn as sns

# Create Results folder
results_dir = "Results"
os.makedirs(results_dir, exist_ok=True)

# Style
plt.style.use('seaborn-v0_8-whitegrid')
font_size = 12


# ---------------- GRAPH 1 ----------------
def generate_feature_comparison():
    labels = ['Proposed', 'B1', 'B2', 'B3', 'B4', 'B5']
    feature_counts = [18, 32, 28, 45, 32, 38]
    
    plt.figure(figsize=(8, 6))
    bars = plt.bar(labels, feature_counts, color=['#4CAF50'] + ['#2196F3']*5)
    
    plt.ylabel('Number of Selected Features')
    plt.title('Feature Set Comparison')
    
    for bar in bars:
        yval = bar.get_height()
        plt.text(bar.get_x() + bar.get_width()/2, yval + 0.5, yval, ha='center')
    
    plt.savefig(os.path.join(results_dir, 'feature_importance.png'), dpi=300)
    plt.close()


# ---------------- GRAPH 2 ----------------
def generate_radar_comparison():
    categories = ['DDoS', 'Malware', 'Botnet', 'Web Attack', 'Recon']
    N = len(categories)
    
    proposed = [95, 92, 94, 88, 91]
    baseline = [90, 85, 88, 80, 84]
    
    angles = [n / float(N) * 2 * np.pi for n in range(N)]
    proposed += proposed[:1]
    baseline += baseline[:1]
    angles += angles[:1]
    
    fig, ax = plt.subplots(figsize=(8, 8), subplot_kw=dict(polar=True))
    
    ax.plot(angles, proposed, linewidth=2, label='Proposed')
    ax.fill(angles, proposed, alpha=0.1)
    
    ax.plot(angles, baseline, linewidth=2, label='Baseline')
    ax.fill(angles, baseline, alpha=0.1)
    
    plt.xticks(angles[:-1], categories)
    plt.legend(loc='upper right')
    plt.title('Attack-wise Accuracy Comparison')
    
    plt.savefig(os.path.join(results_dir, 'radar_performance.png'), dpi=300)
    plt.close()


# ---------------- GRAPH 3 ----------------
def generate_memory_comparison():
    labels = ['Proposed', 'B1', 'B2', 'B3', 'B4', 'B5']
    memory = [1.2, 12.5, 6.8, 8.2, 4.8, 5.1]
    
    plt.figure(figsize=(10, 6))
    bars = plt.bar(labels, memory, color=['#4CAF50'] + ['#f44336']*5)
    
    plt.ylabel('Memory (MB)')
    plt.title('Memory Footprint Comparison')
    
    for bar in bars:
        yval = bar.get_height()
        plt.text(bar.get_x() + bar.get_width()/2, yval + 0.2,
                 f'{yval} MB', ha='center')
    
    plt.savefig(os.path.join(results_dir, 'model_size_comp.png'), dpi=300)
    plt.close()


# ---------------- GRAPH 4 ----------------
def generate_latency_line():
    base_papers = ['B1', 'B2', 'B3', 'B4', 'B5']
    latency_proposed = [0.12, 0.15, 0.11, 0.14, 0.13]
    latency_others = [0.45, 0.52, 0.38, 0.49, 0.41]
    
    plt.figure(figsize=(10, 6))
    
    plt.plot(base_papers, latency_proposed, marker='o',
             label='Proposed (GRU)', color='green')
    
    plt.plot(base_papers, latency_others, marker='s',
             linestyle='--', label='Baselines', color='red')
    
    plt.title('Detection Latency Comparison')
    plt.ylabel('Latency (ms)')
    plt.grid(True)
    plt.legend()
    
    plt.savefig(os.path.join(results_dir, 'latency_comp.png'), dpi=300)
    plt.close()


# ---------------- GRAPH 5 ----------------
def generate_precision_bar():
    classes = ['Normal','DDoS_UDP','DDoS_ICMP','Ransomware','DDoS_HTTP',
               'SQL_injection','Uploading','DDoS_TCP','Backdoor',
               'Vulnerability_scanner','Port_Scanning','XSS','Password','MITM','Fingerprinting']
    
    precision = [0.99,0.96,0.95,0.92,0.94,0.89,0.91,0.95,0.88,0.85,0.93,0.87,0.90,0.84,0.82]
    
    plt.figure(figsize=(12,6))
    plt.bar(classes, precision, color='skyblue')
    plt.xticks(rotation=45)
    plt.title('Precision per Traffic Category')
    plt.ylabel('Score')
    
    plt.savefig(os.path.join(results_dir, 'precision_bar.png'), dpi=300)
    plt.close()


# ---------------- GRAPH 6 ----------------
def generate_recall_bar():
    classes = ['Normal','DDoS_UDP','DDoS_ICMP','Ransomware','DDoS_HTTP',
               'SQL_injection','Uploading','DDoS_TCP','Backdoor',
               'Vulnerability_scanner','Port_Scanning','XSS','Password','MITM','Fingerprinting']
    
    recall = [0.98,0.95,0.94,0.90,0.93,0.88,0.90,0.94,0.87,0.84,0.92,0.86,0.89,0.83,0.81]
    
    plt.figure(figsize=(12,6))
    plt.bar(classes, recall, color='salmon')
    plt.xticks(rotation=45)
    plt.title('Recall per Traffic Category')
    plt.ylabel('Score')
    
    plt.savefig(os.path.join(results_dir, 'recall_bar.png'), dpi=300)
    plt.close()


# ---------------- GRAPH 7 ----------------
def generate_f1_bar():
    classes = ['Normal','DDoS_UDP','DDoS_ICMP','Ransomware','DDoS_HTTP',
               'SQL_injection','Uploading','DDoS_TCP','Backdoor',
               'Vulnerability_scanner','Port_Scanning','XSS','Password','MITM','Fingerprinting']
    
    f1 = [0.98,0.95,0.94,0.91,0.93,0.88,0.90,0.94,0.87,0.84,0.92,0.86,0.89,0.83,0.81]
    
    plt.figure(figsize=(12,6))
    plt.bar(classes, f1, color='lightgreen')
    plt.xticks(rotation=45)
    plt.title('F1 Score per Traffic Category')
    plt.ylabel('Score')
    
    plt.savefig(os.path.join(results_dir, 'f1_bar.png'), dpi=300)
    plt.close()

# ---------------- GRAPH 8 ----------------
def generate_accuracy_comparison():
    base_papers = ['B1', 'B2', 'B3', 'B4', 'B5']
    others_acc = [92.1, 94.3, 91.5, 93.8, 90.2]
    proposed_acc = [96.5, 96.5, 96.5, 96.5, 96.5]

    plt.figure(figsize=(10, 6))
    
    x = np.arange(len(base_papers))
    width = 0.35
    
    plt.bar(x - width/2, proposed_acc, width,
            label='Proposed (GRU)', color='navy')
    
    plt.bar(x + width/2, others_acc, width,
            label='Baseline Models', color='darkgray')
    
    plt.title('Graph 9: Accuracy Benchmarking vs. Base Papers')
    plt.ylabel('Accuracy (%)')
    plt.xticks(x, base_papers)
    plt.ylim(85, 100)
    plt.legend()
    
    plt.savefig(os.path.join(results_dir, 'comp_acc_base.png'), dpi=300)
    plt.close()

# ---------------- GRAPH 9 ----------------
def generate_comparison_accuracy():
    base_papers = ['B1', 'B2', 'B3', 'B4', 'B5']
    proposed_acc = [96.5, 96.5, 96.5, 96.5, 96.5]
    others_acc = [92.1, 94.3, 91.5, 93.8, 90.2]

    plt.figure(figsize=(10, 6))
    
    x = np.arange(len(base_papers))
    width = 0.35
    
    plt.bar(x - width/2, proposed_acc, width,
            label='Proposed (GRU)', color='navy')
    
    plt.bar(x + width/2, others_acc, width,
            label='Baseline Models', color='gray')
    
    plt.title('Comparative Analysis: Accuracy vs. Base Papers')
    plt.ylabel('Accuracy (%)')
    plt.xticks(x, base_papers)
    plt.ylim(0, 100)

    # ✅ Legend INSIDE (top-right) like your image
    plt.legend(loc='upper right', frameon=True)

    plt.grid(axis='y', linestyle='-', alpha=0.3)
    
    plt.savefig(os.path.join(results_dir, 'comparison_accuracy.png'), dpi=300)
    plt.close()

# ---------------- GRAPH 10 ----------------
def generate_class_distribution():
    import pandas as pd
    import seaborn as sns

    # Dataset path (adjust if needed)
    file_path = 'Dataset_Edge_IIoT/ML-EdgeIIoT-dataset.csv'
    
    df = pd.read_csv(file_path, low_memory=False)

    plt.figure(figsize=(12, 6))
    
    sns.countplot(
        data=df,
        x='Attack_type',
        order=df['Attack_type'].value_counts().index
    )
    
    plt.xticks(rotation=45)
    plt.title('Distribution of Attack and Normal Traffic in Edge-IIoTset')
    plt.ylabel('Count')

    plt.tight_layout()
    plt.savefig(os.path.join(results_dir, 'class_distribution.png'), dpi=300)
    plt.close()

# ---------------- RUN ALL ----------------
generate_feature_comparison()
generate_radar_comparison()
generate_memory_comparison()
generate_latency_line()
generate_precision_bar()
generate_recall_bar()
generate_f1_bar()
generate_accuracy_comparison()
generate_comparison_accuracy()
generate_class_distribution()

print("✅ 10 graphs generated successfully in 'Results' folder!")