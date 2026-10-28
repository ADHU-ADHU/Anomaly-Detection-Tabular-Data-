# Tabular GANomaly: implementation notes

The source is [Pytorch  GANomaly_nooutput.ipynb](../Pytorch%20%20GANomaly_nooutput.ipynb). Cell indices below are zero-based notebook JSON indices; the notebook has no saved execution counts or outputs.

## Networks

All networks use fully connected `nn.Linear` layers. The input contains ten sensor features and the latent dimension is eight.

| Module | Linear widths | Activations and regularization | Definition |
| --- | --- | --- | --- |
| `Encoder` / E₁ | `10 → 64 → 32 → 16 → 8` | LeakyReLU(0.2) after each layer; BatchNorm after hidden activations at 64, 32 and 16 | Cell 19 |
| `Decoder` | `8 → 16 → 32 → 64 → 10` | Hidden LeakyReLU(0.2), then BatchNorm; final Tanh | Cell 19 |
| `Encoder2` / E₂ | `10 → 256 → 128 → 64 → 32 → 8` | Hidden LeakyReLU(0.2); linear latent output | Cell 23 |
| `Discriminator` | `10 → 32 → 16 → 1` | Hidden LeakyReLU(0.2) and Dropout(0.3); final Sigmoid | Cell 21 |

The `Encoder2` source comment says it has the same structure as the first encoder, but the actual layer definitions differ. The repository diagram follows those definitions.

## Three training steps

Cells 24–27 define the losses, optimizers and training loop. Let `x` be a normal training row, `z = E₁(x)`, `x_hat = Decoder(z)`, and `z_hat = E₂(x_hat)`.

1. **Discriminator:** distinguish real normal observations from reconstructed observations. The reconstruction is detached for this update.

   ```text
   L_D = [BCE(D(x), 1) + BCE(D(x_hat.detach()), 0)] / 2
   ```

2. **First encoder and decoder:** generate reconstructions that resemble normal input and are classified as real.

   ```text
   L_G = 1 × BCE(D(x_hat), 1) + 50 × L1(x_hat, x)
   ```

3. **Second encoder:** reproduce the first encoder's latent representation from the reconstruction.

   ```text
   L_E2 = 1 × MSE(z_hat, z)
   ```

   The first encoder and decoder run inside `torch.no_grad()` for this step. The latent loss therefore updates E₂ only.

All three Adam optimizers use learning rate `0.0002` and betas `(0.5, 0.999)`. The configured batch size is `512`, with a maximum of `150` epochs. The original loop stops early when the average discriminator loss falls below `0.3`.

## Inference scores

Cell 33 defines `compute_scores`. Each score is calculated per observation:

```text
discriminator_output = D(x)
reconstruction_score = mean(abs(x - x_hat))
latent_score         = mean((z - z_hat) ** 2)
combined_score       = reconstruction_score + latent_score
```

The reconstruction score averages over ten features; the latent score averages over eight latent coordinates. The combined score is an unweighted sum in the notebook, without an additional normalization step. It excludes the discriminator output.

Cells 39 and 41 evaluate latent-only and combined scores using ROC, precision–recall and an F1 threshold sweep. The E₂ latent visualizations later in the notebook encode input observations directly; the anomaly scorer above encodes their reconstructions.

## Relationship to the original GANomaly method

The [GANomaly paper](https://arxiv.org/abs/1805.06725) introduces an encoder–decoder–encoder approach for image anomaly detection. This project transfers that idea to tabular sensor observations using MLPs, asymmetric encoders, adversarial BCE and separate E₂ fitting. The paper's discriminator feature matching and joint latent regularization are useful reference points for further experiments.

The diagram and documentation describe the committed implementation. The original notebooks are preserved, and no retraining was performed while preparing this presentation.
