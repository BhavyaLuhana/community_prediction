# 🧠 Detecting and Predicting the Evolution of Online Communities in Dynamic Social Networks Using Scalable Graph Neural Networks

This repository contains the full code, dataset preprocessing scripts, and model implementation for our research project on **temporal community detection** using a scalable Graph Neural Network framework.

The project explores how online communities evolve—merging, splitting, dissolving, and forming over time—and presents a scalable, interpretable solution based on **EvolveGCN-H**.

---

## 🔍 Project Overview

Traditional static graph models treat online communities like fixed snapshots, completely missing their real-world dynamics. In contrast, our model:

- Tracks communities across time snapshots
- Detects events like splits, merges, and dissolutions
- Predicts future structural shifts using learned temporal embeddings
- Scales to **10K+ node** graphs efficiently on GPU

Tested on both:
- **Real-world data**: Facebook Social Circles [SNAP]
- **Synthetic dynamic graphs**: Generated using a dynamic stochastic block model (SBM)

---

## 🚀 Key Features

- ✅ **EvolveGCN-H-based model** for evolving weights through GRU updates
- ✅ **Node-level feature extraction** (degree, centrality, PageRank)
- ✅ **Temporal clustering using KMeans**
- ✅ **Event detection rules using Jaccard thresholds**
- ✅ **GPU-accelerated training (PyTorch Geometric)**
- ✅ **Visual analytics**: Sankey diagrams, ARI/NMI curves, and dynamic cluster transitions

---

## 📁 Folder Structure

- will update


---

## 📊 Results Summary

On the Facebook Social Circles dataset:

| Metric     | Static GCN | Ours (EvolveGCN) | Gain       |
|------------|------------|------------------|------------|
| ARI        | 0.51       | 0.68             | +33.3%     |
| NMI        | 0.58       | 0.73             | +25.9%     |
| F1-Score   | 0.55       | 0.71             | +29.1%     |
| Modularity | 0.42       | 0.52             | +23.8%     |

✅ Consistent improvements across 10 snapshots  
✅ Real-world interpretable insights into subreddit fracturing and influencer shifts

---

## ⚙️ Installation

```bash
git clone https://github.com/BhavyaLuhana/community_prediction.git
cd community_prediction
pip install -r requirements.txt

📌 Dependencies
    Python 3.9+
    PyTorch
    PyTorch Geometric
    NetworkX
    scikit-learn
    NumPy
    matplotlib
    seaborn

📈 Run the Project
1. Preprocess the dataset
    python scripts/preprocess.py


2. Extract node features
    python scripts/feature_engineering.py


3. Train the model
    python scripts/train_model.py



@article{luhana2024communitygnn,
  title={Detecting and Predicting the Evolution of Online Communities in Dynamic Social Networks Using Scalable Graph Neural Networks},
  author={Luhana, Bhavya},
  year={2024},
  journal={Preprint},
  note={https://github.com/BhavyaLuhana/community_prediction}
}


👨‍💻 Author
Bhavya Luhana

BTech Student, D.Y. Patil University
LinkedIn · GitHub


🧠 Acknowledgements
EvolveGCN
SNAP Dataset
PyTorch Geometric


NOT COMPLETED WILL EDIT
