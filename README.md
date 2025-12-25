# ⚖️ FairFace: Debiasing Facial Detection using DB-VAE

![Python](https://img.shields.io/badge/Python-3.8%2B-blue)
![Framework](https://img.shields.io/badge/Framework-TensorFlow%20%7C%20Keras-orange)
![Topic](https://img.shields.io/badge/Topic-AI%20Ethics%20%26%20Fairness-red)
![Model](https://img.shields.io/badge/Model-Variational%20Autoencoder%20(VAE)-green)

## 📌 Project Overview
**FairFace** is a Deep Learning project aimed at identifying and mitigating algorithmic bias in facial detection systems. Standard CNNs often perform poorly on underrepresented groups (e.g., darker skin tones or specific gender/age groups) due to dataset imbalance.

This project implements a **Debiasing Variational Autoencoder (DB-VAE)** to learn the latent structure of facial features and automatically re-weight training examples during learning, ensuring fair classification accuracy across all demographics.

## ⚙️ The Problem & Solution
* **The Bias:** Models trained on skewed datasets (like CelebA) tend to learn features correlated with the majority class (e.g., Light Skin / Female), leading to high error rates for minorities.
* **The Fix (DB-VAE):**
    1.  **Latent Learning:** A VAE learns a low-dimensional representation of faces (Latent Space).
    2.  **Automated Re-sampling:** The model identifies images in "sparse" regions of the latent space (underrepresented faces) and increases their sampling probability during training.
    3.  **Result:** The classifier sees "hard" or "rare" examples more often, reducing bias without needing manual labeling of attributes.

## 🛠️ Technical Implementation
* **Architecture:** Convolutional VAE (Encoder-Decoder) with a Classification Head.
* **Loss Function:**
    * **VAE Loss:** Reconstruction Loss + KL Divergence (for latent learning).
    * **Classification Loss:** Cross-Entropy (for face detection).
* **Debiasing Algorithm:** Adaptive re-sampling based on the estimated latent distribution $Q(z|x)$.

## 🚀 How to Run
1.  **Clone the repository:**
    ```bash
    git clone [https://github.com/mariamashraf731/FairFace-DBVAE.git](https://github.com/mariamashraf731/FairFace-DBVAE.git)
    ```
2.  **Install Dependencies:**
    ```bash
    pip install -r requirements.txt
    ```
3.  **Run the Analysis:**
    ```bash
    jupyter notebook notebooks/Debiasing_Facial_Detection.ipynb
    ```

## 👨‍💻 Tech Stack
* **Language:** Python
* **Deep Learning:** TensorFlow / Keras
* **Concepts:** Unsupervised Learning, VAEs, Latent Space Analysis, Algorithmic Fairness.