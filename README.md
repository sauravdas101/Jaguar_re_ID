# Jaguar Re-Identification 🐆

This repository contains my solutions and experimental approaches for the [Kaggle Jaguar Re-ID Competition](https://www.kaggle.com/competitions/jaguar-re-id). Using this approach I obtained a rank of 12 among roughly 400 participants.

## 📖 Overview
Jaguars are the apex predators of the Americas, but tracking their populations across vast landscapes like the Pantanal is a monumental task. Traditionally, researchers have relied on manual identification by comparing unique spot patterns (rosettes), which act like a biological fingerprint. With the rise of eco-tourism and citizen science, thousands of photographs are captured each year, making manual identification a significant bottleneck for conservation efforts.

The **Jaguar Identification Project** aims to automate this process. The goal of this competition is to develop a computer vision model capable of identifying individual jaguars (e.g., "Medrosa," "Patricia," or "Ousado") from wildlife photographs, thereby advancing wildlife demographics research and supporting the survival of this near-threatened species.

## 🎯 The Challenge
Identifying individual animals in the wild presents unique obstacles compared to standard image classification:
* **Intra-class Variation:** The same jaguar can look drastically different depending on lighting, camera angle, and posture.
* **Inter-class Similarity:** Different jaguars can have surprisingly similar rosette patterns.
* **Extreme Data Imbalance:** Some jaguars have dozens of high-quality photos (frequent identities), while others have only a few (rare identities).
* **Spurious Correlations:** Models can "cheat" by learning the background (e.g., a specific riverbank) rather than the animal's unique physiological patterns.

## ⚖️ Evaluation Metric
Submissions are evaluated using **Identity-balanced Mean Average Precision (mAP)**. 

Standard mAP would naturally be dominated by frequently photographed individuals. Identity-balanced mAP addresses this by:
1. Computing the average precision for each query image.
2. Averaging the precision scores for all queries of the *same identity*.
3. Computing the final mAP as the mean across *all identities* (macro-average).

This aligns with conservation priorities: every individual jaguar has equal importance, regardless of how often it appears in camera trap footage.

## 📦 Modules & Dependencies
To run the notebooks in this repository, you will need the following core Python libraries:
* `torch` / `torchvision` (PyTorch for deep learning and neural networks)
* `timm` (PyTorch Image Models for the EVA-02 Large backbone)
* `pytorch-metric-learning` (For Batch-Hard Triplet Margin Loss)
* `pandas` & `numpy` (For data manipulation and dataset handling)
* `scikit-learn` (For splitting data and calculating evaluation metrics)
* `tqdm` (For training loop progress bars)

## 📂 Repository Structure & Approaches

This repository contains three main Jupyter Notebooks, each exploring a progressively complex architectural approach to solve the Re-ID problem using **EVA-02 Large** as the primary backbone.

### 1. `jaguarre-id-balancedarcface-weighted-cross-entropy.ipynb`
**Core Idea: Class-Aware Penalty**
* Implements a custom **Balanced ArcFace** module that dynamically adjusts the scale factor (`s`) based on the frequency of the jaguar in the training set. 
* Uses **Weighted Random Sampling** and **Weighted Cross-Entropy** to heavily penalize mistakes made on rare jaguars.
* Serves as a robust baseline that strictly aligns the training objective with the Identity-balanced mAP evaluation metric.

### 2. `jaguar-re-id-balancedarcface-triplet-loss.ipynb`
**Core Idea: Global-Local Metric Learning & Flat Minima**
* **Hybrid Loss:** Combines Balanced ArcFace (for classification) with **Batch-Hard Triplet Margin Loss** (via PyTorch Metric Learning) to strictly enforce intra-class compactness and inter-class separation based on rosette patterns.
* **Sharpness-Aware Minimization (SAM):** Implements a two-phase training loop (AdamW Warmup -> SAM Fine-Tuning). By minimizing both the loss value and the loss sharpness, the model learns a "flat minima," drastically reducing its reliance on spurious backgrounds and improving generalization to the unseen test cameras.

### 3. `balancedarcface-collaborative-learning.ipynb`
**Core Idea: Multi-Head Ensembles & Dynamic Sub-Centers**
* **Collaborative Learning:** Trains an `EVABoss` architecture with multiple classification heads. This "peer-learning" technique encourages the backbone to extract richer, more diverse feature representations.
* **Dynamic Sub-Center ArcFace:** A highly advanced metric learning approach that accommodates the high intra-class variance of jaguars in different poses. 
  * Frequent jaguars are assigned `K=3` sub-centers (e.g., left-side, right-side, frontal).
  * Rare jaguars are assigned `K=1` sub-center via a boolean masking trick to prevent unstable gradients.
* Features a "Warm-Start" mechanism that seamlessly copies weights from a pre-trained Balanced ArcFace head into the Dynamic Sub-Center head mid-training.
* Inference is combined using **Reciprocal Rank Fusion (RRF)** across similarity matrices rather than simple majority voting to preserve identity-balanced integrity.

## 🚀 How to Use
1. Clone this repository: `git clone https://github.com/sauravdas101/Jaguar_re_ID.git`
2. Download the dataset from the [Kaggle Competition Page](https://www.kaggle.com/competitions/jaguar-re-id/data) and place the `train/` and `test/` directories in your data path.
3. Install the required dependencies (e.g., `pip install torch torchvision timm pytorch-metric-learning pandas numpy scikit-learn tqdm`).
4. Update the `Config` class paths within the notebooks to point to your local directories.
5. Run the notebooks sequentially (starting from weighted cross-entropy as a baseline).
