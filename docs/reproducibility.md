# Reproducing the notebook experiments

## Data location and schema

The notebooks expect `train.mat` in their working directory and use `scipy.io.loadmat` to read its `Train_data` entry. The data file is external to this repository. See the [ARAMIS challenge brief](https://aramis3d.com/wp-content/uploads/2022/07/Aramis-Challenge.pdf) for its origin and task definition.

The committed preprocessing accesses:

```python
train_data = data['Train_data']
system = train_data[0, i]
trajectories = system['trajectory']  # four component trajectories
labels = system['Label']
traj = trajectories[0, j]           # 10 × 1000 in the notebook's expected data
label = labels[j]                  # one label per time step
```

Each time step becomes one ten-feature row. Labels select normal rows (`0`) and anomalous rows (`1`). The models train on normal rows; the anomaly labels support evaluation.

## Split, scaling and evaluation

Normal rows are split with `train_test_split(test_size=0.2, random_state=42)`. `MinMaxScaler(feature_range=(-1, 1))` is fitted on normal training rows and used to transform held-out normal rows and anomaly rows.

The split does not isolate systems, trajectories or time ranges. Related observations can be present on both sides. For a stronger generalization study, split by system or trajectory before pooling the rows.

The notebooks evaluate held-out normal rows against labeled anomalous rows from the development data. They do not load the challenge's separate 50-system test set or calculate its abnormal-onset timing metric.

Threshold selection maximizes F1 on the same labeled samples used to report classification performance. For a final assessment, select the threshold using separate calibration data, then freeze it before evaluating the held-out test data. ROC/PR curves, F1 and precision/recall answer different questions; retain the anomaly prevalence when comparing runs.

## Notebook environments

The [README](../README.md#run-locally) lists the imported packages and environment setup. No original dependency lockfile, package versions, trained weights or dataset are committed. The neural models select CUDA when available.

The scripts are exploratory notebooks with shared cell state. Run each notebook from the top in a fresh kernel and use `train.mat` as described above. Weight initialization, stochastic training and downsampling are not globally seeded, so repeated runs need not produce identical results.

## Checkpoints

The GANomaly notebook creates `checkpoints/ganomaly_checkpoint.pth` and automatically resumes from it if it exists. Its original loop saves periodically, every ten epoch indices, rather than selecting a verified best checkpoint.

Keep checkpoints from different experiments separate. If moving a checkpoint saved on a GPU to a CPU-only machine, review the notebook's `torch.load` call and configure its device mapping as needed. The checkpoint is not supplied with this repository.

## Model settings and saved evidence

| Experiment | Settings in the committed notebook | Saved outputs |
| --- | --- | --- |
| Autoencoder | MSE reconstruction; 8-dimensional latent space; Adam `0.0002`; batch `512`; up to `150` epochs | Training logs, visualizations and evaluation results |
| β-VAE | MSE reconstruction plus `100 × KL`; 8-dimensional mean/log-variance heads; Adam `0.001`; batch `512`; up to `5` epochs | None |
| Tabular GANomaly | Adversarial BCE, `50 × L1` reconstruction, separate E₂ latent MSE; Adam `0.0002`; batch `512`; up to `150` epochs | None |

The saved Autoencoder output reports ROC-AUC `0.9974`, PR curve AUC `0.9872`, F1 `0.9466`, precision `0.9294` and recall `0.9645`. These are historical, threshold-tuned results from that notebook. They do not establish GANomaly performance or a controlled cross-model comparison.

The VAE notebook explores reconstruction/KL scores, t-SNE, UMAP, a five-component Gaussian Mixture Model, Mahalanobis distance and One-Class SVM on the latent mean. Its reconstruction and KL plot-history assignments should be reviewed before using those curve labels to interpret a new run.

Repository presentation work inspected the source and saved outputs without executing model training. New results should report data grouping, seeds, environment versions, training budgets and threshold calibration alongside the scores.
