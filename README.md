# Handwritten Digit Classifier with a Neural Network

A two-layer neural network implemented from scratch in NumPy and trained with gradient descent to classify MNIST digits. A web interface lets you train the network and draw digits for it to classify.

The best configuration (α = 0.5, 500 epochs, 256 hidden neurons) reaches **95.44% test accuracy**.

## Model

**Data.** MNIST has 70,000 28×28 grayscale images of handwritten digits (0–9). Pixels are scaled to [0, 1] and the set is split at random into 56,000 training and 14,000 test examples (80/20, seed 42). Labels are one-hot encoded (C = 10).

**Architecture.** 784 inputs, one hidden layer with *h* neurons, and 10 outputs. For a batch $X \in \mathbb{R}^{m \times 784}$:

$$Z_1 = XW_1 + b_1,\quad A_1 = \mathrm{relu}(Z_1) = \max(0, Z_1)$$

$$Z_2 = A_1W_2 + b_2,\quad A_2 = \mathrm{softmax}(Z_2)$$

ReLU avoids sigmoid saturation in the hidden layer, and softmax turns the output into a probability distribution over the 10 classes. Weights $W_1 \in \mathbb{R}^{784 \times h}$ and $W_2 \in \mathbb{R}^{h \times 10}$ are drawn from $\mathcal{N}(0, 1) \cdot 0.01$, which breaks the symmetry between neurons, and biases start at zero. The network has $795h + 10$ parameters (3,190 for h = 4, 203,530 for h = 256).

**Training.** Cross-entropy loss $L = -\frac{1}{m}\sum_i \sum_c y_{ic} \log \hat{y}_{ic}$, minimized with full-batch gradient descent: each epoch runs the forward pass over all 56,000 examples, computes $L$, backpropagates, and updates $w \leftarrow w - \alpha\,\partial L/\partial w$ for $W_1, b_1, W_2, b_2$. After each epoch, accuracy is measured on the test set as the fraction of examples where $\arg\max \hat{y}_i = \arg\max y_i$. All operations are vectorized, with no loops over examples.

## Results

The base configuration is α = 0.5, h = 64 and 500 epochs, and each experiment varies one hyperparameter. Training is deterministic, so the results for 50, 100, 200 and 500 epochs come from the same 500-epoch run. Times were measured on a 20-core CPU.

| Learning rate α | Final training loss | Test accuracy |
|---|---|---|
| 0.001 | 2.2997 | 14.91% |
| 0.01 | 2.0891 | 45.19% |
| 0.1 | 0.3417 | 90.21% |
| 0.5 | 0.1601 | 94.96% |

| Epochs | Final training loss | Test accuracy |
|---|---|---|
| 50 | 0.6849 | 78.84% |
| 100 | 0.3380 | 90.24% |
| 200 | 0.2585 | 92.43% |
| 500 | 0.1601 | 94.96% |

| Hidden neurons h | Final training loss | Test accuracy | ms/epoch |
|---|---|---|---|
| 4 | 1.2028 | 62.33% | 264 |
| 16 | 0.2102 | 93.72% | 262 |
| 64 | 0.1601 | 94.96% | 358 |
| 256 | 0.1462 | 95.44% | 819 |

### Best configuration

Combining the best value from each experiment (α = 0.5, 500 epochs, h = 256) gives **95.44% test accuracy** and a training loss of 0.1462. Accuracy passes 90% at epoch 84. Per-class accuracy ranges from 93.2% (digit 8) to 97.8% (digit 1). The most frequent confusions are 4→9 (37 cases), 9→4 (32), 5→3 (28), 8→3 (26) and 3→5 (23).

<p align="center">
  <img src="src/losses.png" width="45%" alt="Loss curve">
  <img src="src/accuracies.png" width="45%" alt="Accuracy curve">
</p>
<p align="center"><em>Training loss and test accuracy of the best configuration, sampled every 100 epochs (point k is epoch 100(k + 1)).</em></p>

### Digits drawn in the interface

With the best weights loaded, one example of each digit was drawn in the interface, plus a small "2" off in a corner. All ten centered digits were classified correctly. The network recognizes 1 and 8 with a wide margin, since their shapes are distinctive. The hard cases repeat the confusions seen on the test set: the 7 narrowly beat 2 and 3 (they share the top bar), the 9 was close to 3, the 4 had 9 as its second choice, and the 5 split probability with 3 (shared lower curve).

The off-center "2" was classified as 7. A fully connected network ties weights to fixed pixel positions, and MNIST digits are centered by center of mass, so strokes away from the center activate patterns learned for other classes. The interface's strokes are also thinner than MNIST's and pixel intensity is not capped at 255. The percentages the interface shows come from a normalized ReLU rather than softmax, so they look low: with softmax, the 7 had 51% confidence and the 9 had 75%, against ≥97% for the other centered digits.

## Discussion

**Learning rate: convergence vs. stability.** With initial weights around $10^{-2}$, activations and gradients are tiny at the start. With α ≤ 0.01 the network barely moves in 500 epochs: the loss stays near $\ln 10 \approx 2.303$, the value of a uniform prediction. With α = 0.1 it leaves this plateau between epochs 50 and 100, but only passes 90% at epoch 456. α = 0.5 converges much faster (90% at epoch 95) at the cost of transient instability: the loss spikes between epochs 30 and 90 (from 0.72 to 1.51 at epoch 33) when large steps overshoot the valley, then settles. Full-batch gradient descent without momentum needs a high α to make progress in few epochs, close to the stability limit.

**Epochs.** Returns diminish: going from 50 to 100 epochs adds 11.4 percentage points, from 200 to 500 only 2.5. Training loss and test accuracy keep improving together up to epoch 500, with no sign of overfitting, but each epoch is a full pass over 56,000 examples.

**Capacity vs. cost.** With h = 4 the hidden layer is a bottleneck: projecting 784 pixels onto 4 dimensions does not separate 10 classes well, and with α = 0.5 training oscillates for all 500 epochs, peaking at 83.7% but ending at 62.3% in a dip. With α = 0.1 the same network is stable and ends at 81.7%, so small networks are also more sensitive to a high α. From 16 to 256 neurons accuracy rises only 1.7 points, while the parameter count grows 16× and time per epoch about 3×. h = 64 is a good compromise (94.96%, 358 ms/epoch).

**Limitations.** Hyperparameters were chosen by accuracy on the test set itself, with no validation set, which makes the estimate optimistic. Each configuration ran with a single seed. Full-batch gradient descent is slow per epoch, and there is no regularization, mini-batching, momentum or learning-rate decay. Softmax does not subtract each row's maximum, which could overflow with large logits (it did not happen in these experiments).

**Possible improvements.** Mini-batch SGD with momentum (or Adam), learning-rate decay, He initialization, L2 regularization or dropout, a separate validation set, and convolutional networks, which exploit the spatial structure of images. In the interface, centering and thickening the drawn strokes the way MNIST is preprocessed would narrow the gap between training data and drawings.

## Running

From the repository root:

```bash
python -m venv .venv
source .venv/bin/activate   # Windows: .venv\Scripts\activate
pip install -r src/requirements.txt
```

Start the interface and open <http://localhost:5001>:

```bash
python src/server.py
```

To plot the curves from `src/model/` (saved to `src/losses.png` and `src/accuracies.png`):

```bash
python src/plot.py
```

`src/model/` already holds the weights of the best configuration. Click **Load Weights** in the interface to draw digits without training first.

## Structure

```
src/
├── learn.py        # network: initialization, ReLU, softmax, forward pass, loss, gradient step, accuracy
├── server.py       # Flask + Socket.IO server
├── plot.py         # loss and accuracy curves
├── model/          # weights (W1, b1, W2, b2) and loss/accuracy history
├── static/         # interface (p5.js)
└── templates/
```
