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


## Models Implemented
1. **Autoencoder** - Baseline reconstruction-based anomaly detection
2. **β-VAE** - Two variants focusing on reconstruction vs KL divergence
3. **GANomaly** - Adversarial approach with latent space comparison

## Future Work
- Incorporation of temporal features
- Hyperparameter optimization
