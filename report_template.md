Report: CIFAR-10 Adversarial Attacks

This report documents the pipeline for training a CNN on CIFAR-10, evaluating its adversarial robustness under FGSM and PGD attacks, applying adversarial training as a defense, and analyzing model behavior using explainability techniques (Grad-CAM and Integrated Gradients).


1) Model Design
* Architecture: Convolutional Neural Network with BatchNorm, Dropout, and Global Average Pooling.
* Optimizers: Adam (clean training), SGD (adversarial training).
* Regularization: Weight decay, dropout.
* Training setup: Default 10–15 epochs, batch size 128.


2) Adversarial Attacks
* FGSM (Fast Gradient Sign Method): single-step attack.
* PGD (Projected Gradient Descent): iterative, stronger attack.
* Parameters: eps = 0.0314, alpha = 0.0078, steps = 4–10.
* Metrics: Clean accuracy vs adversarial accuracy under attacks.


3) Defense via Adversarial Training
* PGD-based adversarial training improves robustness at the cost of some clean accuracy.
* Key results to report:
o Best clean accuracy after adversarial training.
o Best adversarial (robust) accuracy.
* Discuss trade-off between standard accuracy and robustness.


4) Explainability
* Grad-CAM: highlights regions most important for classification.
* Integrated Gradients: shows pixel-level contribution to predictions.
* Results: Compare visualizations on clean vs adversarial inputs.


5) Discussion
* Trade-off observed between clean accuracy and adversarial robustness.
* Effect of hyperparameters (eps, alpha, steps) on robustness.
* Limitations: robustness gain is partial, larger models may perform better.
* Future work:
o Try other defenses (TRADES, randomized smoothing).
o Explore different architectures (ResNet, WideResNet).


6) Key Results (to fill with your numbers)
* Clean model:
o Clean accuracy: ~80%
o Adversarial accuracy (PGD, eps=0.0314): ~0%

* Adversarially trained model:
o Clean accuracy: ~61%
o Adversarial accuracy (PGD, eps=0.0314): ~30%


