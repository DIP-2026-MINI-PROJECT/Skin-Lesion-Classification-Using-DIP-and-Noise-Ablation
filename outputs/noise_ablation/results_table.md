## Noise ablation results (same fixed model, 60 test images)

Precision, Recall, F1 and AUC are for the positive class "suspicious".

| Condition | Severity | Accuracy | Precision | Recall | F1 | AUC |
|---|---|---|---|---|---|---|
| Clean | None | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 1.0000 |
| Gaussian | Low | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 1.0000 |
| Gaussian | Medium | 0.9167 | 0.8571 | 1.0000 | 0.9231 | 0.9167 |
| Gaussian | High | 0.5500 | 0.5273 | 0.9667 | 0.6824 | 0.5500 |
| Salt-Pepper | Low | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 1.0000 |
| Salt-Pepper | Medium | 0.5167 | 0.5085 | 1.0000 | 0.6742 | 0.5167 |
| Salt-Pepper | High | 0.5000 | 0.5000 | 1.0000 | 0.6667 | 0.5000 |

### Drop vs clean (clean - noisy; positive = worse)

| Condition | Severity | Accuracy | Precision | Recall | F1 | AUC |
|---|---|---|---|---|---|---|
| Gaussian | Low | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 |
| Gaussian | Medium | 0.0833 | 0.1429 | 0.0000 | 0.0769 | 0.0833 |
| Gaussian | High | 0.4500 | 0.4727 | 0.0333 | 0.3176 | 0.4500 |
| Salt-Pepper | Low | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 |
| Salt-Pepper | Medium | 0.4833 | 0.4915 | 0.0000 | 0.3258 | 0.4833 |
| Salt-Pepper | High | 0.5000 | 0.5000 | 0.0000 | 0.3333 | 0.5000 |

### Percentage drop vs clean (100 * drop / clean)

| Condition | Severity | Accuracy | Precision | Recall | F1 | AUC |
|---|---|---|---|---|---|---|
| Gaussian | Low | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 |
| Gaussian | Medium | 8.3333 | 14.2857 | 0.0000 | 7.6923 | 8.3333 |
| Gaussian | High | 45.0000 | 47.2727 | 3.3333 | 31.7647 | 45.0000 |
| Salt-Pepper | Low | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 |
| Salt-Pepper | Medium | 48.3333 | 49.1525 | 0.0000 | 32.5843 | 48.3333 |
| Salt-Pepper | High | 50.0000 | 50.0000 | 0.0000 | 33.3333 | 50.0000 |
