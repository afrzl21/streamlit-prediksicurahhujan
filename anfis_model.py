import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error
from sklearn.utils import resample
from sklearn.model_selection import train_test_split
import copy
import pickle

class ANFIS:
    def __init__(self, n_inputs=4, n_mf=2, alpha=0.01, epochs=300):
        self.n_inputs = n_inputs
        self.n_mf = n_mf
        self.alpha = alpha
        self.epochs = epochs

        # Inisialisasi parameter antecedent
        self.mu = np.random.rand(n_inputs, n_mf) * 2 - 1
        self.sigma = np.random.rand(n_inputs, n_mf) + 0.5

        # Parameter consequent
        self.p = np.random.randn(n_mf**n_inputs, n_inputs + 1) * 0.1

        # Inisialisasi gradien
        self.d_mu = np.zeros_like(self.mu)
        self.d_sigma = np.zeros_like(self.sigma)
        self.d_p = np.zeros_like(self.p)

        # Penyimpanan loss
        self.train_loss = []
        self.val_loss = []

    def gaussian_mf(self, x, mu, sigma):
        return np.exp(-((x - mu)**2) / (2 * sigma**2))

    def forward(self, x):
        self.x = x
        self.membership = np.zeros((self.n_inputs, self.n_mf))

        # Layer 1: Fuzzifikasi
        for i in range(self.n_inputs):
            for j in range(self.n_mf):
                self.membership[i,j] = self.gaussian_mf(x[i], self.mu[i,j], self.sigma[i,j])

        # Layer 2: Fire strength
        self.rule_activation = np.ones(self.n_mf**self.n_inputs)
        for r in range(self.n_mf**self.n_inputs):
            bits = np.unravel_index(r, [self.n_mf]*self.n_inputs)
            for i in range(self.n_inputs):
                self.rule_activation[r] *= self.membership[i, bits[i]]

        # Layer 3: Normalisasi
        self.norm_activation = self.rule_activation / (self.rule_activation.sum() + 1e-12)

        # Layer 4: Consequent
        self.consequent = np.zeros(self.n_mf**self.n_inputs)
        for r in range(self.n_mf**self.n_inputs):
            self.consequent[r] = np.dot(self.p[r], np.append(x, 1))

        # Layer 5: Aggregasi
        self.output = np.dot(self.norm_activation, self.consequent)
        return self.output

    def backward(self, y_pred, y_true):
        error = y_pred - y_true

        # Gradien consequent
        for r in range(self.n_mf**self.n_inputs):
            grad_p = error * self.norm_activation[r] * np.append(self.x, 1)
            self.d_p[r] += grad_p

        # Gradien antecedent
        total_activation = self.rule_activation.sum()
        d_norm = error * (self.consequent - self.output) / (total_activation + 1e-12)

        for i in range(self.n_inputs):
            for j in range(self.n_mf):
                d_mu = 0
                d_sigma = 0
                for r in range(self.n_mf**self.n_inputs):
                    bits = np.unravel_index(r, [self.n_mf]*self.n_inputs)
                    if bits[i] == j:
                        term = d_norm[r] * self.rule_activation[r]
                        d_mu += term * (self.x[i] - self.mu[i,j]) / (self.sigma[i,j]**2)
                        d_sigma += term * ((self.x[i] - self.mu[i,j])**2) / (self.sigma[i,j]**3)
                self.d_mu[i,j] += d_mu
                self.d_sigma[i,j] += d_sigma

    def update_params(self, batch_size):
        self.mu -= self.alpha * self.d_mu / batch_size
        self.sigma -= self.alpha * self.d_sigma / batch_size
        self.p -= self.alpha * self.d_p / batch_size

        # Reset gradien
        self.d_mu.fill(0)
        self.d_sigma.fill(0)
        self.d_p.fill(0)

    def fit(self, X, y, X_val=None, y_val=None):
        self.train_loss = []
        self.val_loss = []

        for epoch in range(self.epochs):
            epoch_loss = 0
            for xi, yi in zip(X, y):
                y_pred = self.forward(xi)
                self.backward(y_pred, yi)
                epoch_loss += (y_pred - yi)**2

            self.update_params(len(X))

            # Hitung training loss
            avg_epoch_loss = epoch_loss / len(X)
            self.train_loss.append(avg_epoch_loss)

            # Hitung validation loss
            if X_val is not None and y_val is not None:
                val_pred = np.array([self.forward(x) for x in X_val])
                self.val_loss.append(mean_squared_error(y_val, val_pred))

            if epoch % 10 == 0:
                val_log = f" | Val Loss: {self.val_loss[-1]:.4f}" if X_val is not None else ""
                print(f"Epoch {epoch:03d}/{self.epochs}, Train Loss: {avg_epoch_loss:.4f}{val_log}")

class ANFISBagging:
    def __init__(self, base_estimator, n_estimators=10, max_samples=1.0):
        self.base_estimator = base_estimator
        self.n_estimators = n_estimators
        self.max_samples = max_samples
        self.estimators = []

    def fit(self, X, y, X_val=None, y_val=None):
        self.estimators = []

        for _ in range(self.n_estimators):
            # 1. Bootstrap sampling
            estimator = copy.deepcopy(self.base_estimator)
            X_sample, y_sample = resample(X, y, n_samples=int(self.max_samples*len(X)))

            # 2. Train model individual
            estimator.fit(X_sample, y_sample, X_val, y_val)
            self.estimators.append(estimator)

    def predict(self, X):
        predictions = np.zeros((len(self.estimators), len(X)))
        for i, estimator in enumerate(self.estimators):
            predictions[i,:] = np.array([estimator.forward(x) for x in X])
        return np.mean(predictions, axis=0)