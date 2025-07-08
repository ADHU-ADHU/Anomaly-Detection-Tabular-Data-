# Predictive Maintenance with Deep Anomaly Detection

This work is course project of the standalone masters course Machine learning for Predictive Maintenance 

## Project Description
Implementation of deep learning models for anomaly detection in industrial equipment using the Aramis Challenge dataset. Developed Autoencoder, Variational Autoencoder (β-VAE), and GANomaly architectures to identify component degradation.

## Key Features
- Three distinct deep learning approaches for anomaly detection
- Specialized β-VAE variants addressing latent collapse
- Adaptation of GANomaly (image-based) to tabular sensor data
- Comprehensive evaluation with ROC/PRC

## Data Description and Task
The dataset used in this notebook is loaded from a MATLAB file (train.mat) and contains trajectory data labeled as normal or anomalous. Each entry in the dataset represents a system with multiple trajectories, where each trajectory is a 10-dimensional time series of length 1000. The labels (0 for normal and 1 for anomalous) indicate whether a given time step in the trajectory is anomalous. The dataset contains 200 such systems, each with 4 trajectories.

### Data Manipulation:
**Loading Data:** The data is loaded using scipy.io.loadmat, which reveals a structure with keys like Train_data.

**Extracting Trajectories and Labels:** For each system, the trajectories and corresponding labels are extracted. The trajectories are shaped as (10, 1000), and the labels are binary arrays of length 1000.

**Separating Normal and Anomalous Data:** The data is split into normal (label == 0) and anomalous (label == 1) samples. The normal data is transposed to shape (N_normal, 10), and the anomalous data is similarly processed.

**Train-Test Split:** The normal data is split into training (80%) and testing (20%) sets.

**Scaling:** The data is scaled using MinMaxScaler to the range [-1, 1] to ensure consistent input for the neural network.

### Task:
The task is anomaly detection, where the goal is to train a model to distinguish between normal and anomalous trajectories, while training only with the normal data. The project focuses on predicting the first entry time into abnormal states for each of the four components in 50 test systems, using sensor data converted into a tabular format.



## Models Implemented
1. **Autoencoder** - Baseline reconstruction-based anomaly detection
2. **β-VAE** - Two variants focusing on reconstruction vs KL divergence
3. **GANomaly** - Adversarial approach with latent space comparison

## Autoencoders
An autoencoder is a type of neural network that learns to compress (encode) input data into a lower-dimensional latent space and then reconstruct (decode) it back to the original form. It consists of two main parts:
1. **Encoder:** Reduces input data into a compact latent representation.
2. **Decoder:** Reconstructs the original data from the latent representation.

Autoencoders are trained to minimize reconstruction error (e.g., Mean Squared Error) between input and output, forcing them to learn efficient representations of normal data.
Since autoencoders are trained on normal data, they learn to reconstruct it well but perform poorly on anomalous data. Anomaly detection works as follows:
- Train the autoencoder only on normal data.
- Compute the reconstruction error for new data.
- If the error exceeds a threshold, classify the sample as an anomaly.

### Advantages:
   - Unsupervised (no labeled anomalies needed).
   - Works well when anomalies are rare.

### Limitations:
   - May struggle if anomalies are too similar to normal data.
   - Sensitive to the choice of threshold.

## Variational Autoencoder (VAE)
A Variational Autoencoder (VAE) is a generative model that combines neural networks with probabilistic Bayesian inference. Unlike a standard autoencoder (which learns a deterministic latent representation), a VAE learns a probability distribution over the latent space.
The [Beta-VAE](https://arxiv.org/abs/2112.14278) modifies the VAE by introducing a β hyperparameter to explicitly control the trade-off between latent disentanglement and reconstruction quality.

**Standard VAE:** Balances reconstruction loss and KL-divergence (which regularizes the latent space).
**Beta-VAE:** Strengthens the effect of KL-divergence by scaling it with β (β > 1 enforces stronger regularization).

Like standard autoencoders, anomalies have higher reconstruction errors. Anomalies may lie in low-probability regions of the latent distribution, further the latent space can be explored to detect anomalies. In this projects the latent space of the normal data where trained to fit GMM, KDE, mahanalobis distance and One class -SVM.

## [GANomaly](https://arxiv.org/abs/1805.06725): GAN-Based Anomaly Detection
GANomaly is a Generative Adversarial Network (GAN) variant designed for anomaly detection. It consists of:

**Generator:** This an typical autoencoder, where the encoder Encodes input into latent space and the decoder resonctructs reconstructs it.

**Discriminator:** Tries to distinguish between real and reconstructed data.

**Encoder:** This second encoder maps the reconstructed data back to latent space (used for anomaly scoring).

The model used in this project is addapted for training tabular data


## 📈 Model Evaluation
Evaluating anomaly detection models, especially in highly imbalanced settings (like predictive maintenance), requires more than just accuracy. This project uses:
- F1 Score
- ROC-AUC (Receiver Operating Characteristic - Area Under Curve)
- PRC-AUC (Precision-Recall Curve - Area Under Curve)

**F1 score:** The F1 score is the harmonic mean of precision and recall, giving a single performance value that balances both. In predictive maintenance, false negatives (missed failures) are very costly, and false positives (unnecessary repairs) are undesirable. F1 balances both.
 **ROC-AUC (Receiver Operating Characteristic):** The ROC curve plots the True Positive Rate (TPR) vs. False Positive Rate (FPR) at various threshold levels. It gives a threshold-independent measure of separability between normal and anomalous data. AUC close to 1 means good separability. However, ROC-AUC can be overly optimistic on imbalanced datasets.
**PRC-AUC (Precision-Recall Curve):** The Precision-Recall Curve focuses on the trade-off between precision and recall. In imbalanced datasets (many more normal than anomalous points), PRC is often more informative than ROC. It focuses on how well the model identifies rare events (anomalies), without being diluted by the majority class.

Anomaly detection often involves computing a reconstruction error or latent space distance, and setting a threshold to flag anomalies. The threshold can be tuned using:
- ROC or PRC curves
- F1-maximization
- Percentile of reconstruction errors from validation data
  
## Future Work
- Incorporation of temporal features
- Hyperparameter optimization
- Develop explainable AI methods to interpret model decisions
- Explore reinforcement learning for optimal maintenance scheduling
- Extend to larger systems with more complex component interactions
