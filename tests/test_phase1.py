"""Tests for Phase 1: Foundations (numpy neural network components).

These tests verify that core building blocks work correctly:
- Activation functions produce expected outputs
- Linear layers have correct shapes
- Backpropagation gradients match numerical gradients
- XOR network converges
- Cross-entropy loss and softmax are correct
"""

import numpy as np
import pytest


# ============================================================
# Activation Functions
# ============================================================

def sigmoid(x):
    return 1.0 / (1.0 + np.exp(-x))

def relu(x):
    return np.maximum(0, x)

def softmax(x):
    exp_x = np.exp(x - np.max(x, axis=-1, keepdims=True))
    return exp_x / np.sum(exp_x, axis=-1, keepdims=True)


class TestActivations:
    def test_sigmoid_range(self):
        """Sigmoid output must be in (0, 1)."""
        x = np.linspace(-10, 10, 100)
        out = sigmoid(x)
        assert np.all(out > 0) and np.all(out < 1)

    def test_sigmoid_symmetry(self):
        """sigmoid(0) = 0.5, sigmoid(-x) = 1 - sigmoid(x)."""
        assert np.isclose(sigmoid(0), 0.5)
        x = np.array([1.0, 2.0, 3.0])
        assert np.allclose(sigmoid(-x), 1 - sigmoid(x))

    def test_relu_positive(self):
        """ReLU passes positive values unchanged."""
        x = np.array([1.0, 2.0, 3.0])
        assert np.allclose(relu(x), x)

    def test_relu_negative(self):
        """ReLU zeros out negative values."""
        x = np.array([-1.0, -2.0, -3.0])
        assert np.allclose(relu(x), 0)

    def test_relu_mixed(self):
        x = np.array([-2.0, -1.0, 0.0, 1.0, 2.0])
        expected = np.array([0.0, 0.0, 0.0, 1.0, 2.0])
        assert np.allclose(relu(x), expected)

    def test_softmax_sums_to_one(self):
        """Softmax output must sum to 1 for each sample."""
        x = np.random.randn(5, 10)
        probs = softmax(x)
        assert np.allclose(probs.sum(axis=-1), 1.0)

    def test_softmax_positive(self):
        """Softmax output must be positive."""
        x = np.random.randn(3, 5)
        assert np.all(softmax(x) > 0)

    def test_softmax_numerical_stability(self):
        """Softmax should handle large values without overflow."""
        x = np.array([[1000.0, 1001.0, 1002.0]])
        probs = softmax(x)
        assert np.allclose(probs.sum(), 1.0)
        assert not np.any(np.isnan(probs))
        assert not np.any(np.isinf(probs))


# ============================================================
# Linear Layer
# ============================================================

class Linear:
    def __init__(self, in_features, out_features):
        scale = np.sqrt(2.0 / in_features)
        self.W = np.random.randn(out_features, in_features) * scale
        self.b = np.zeros(out_features)
        self.grad_W = None
        self.grad_b = None

    def forward(self, x):
        self.x = x
        return x @ self.W.T + self.b

    def backward(self, grad_output):
        self.grad_W = grad_output.T @ self.x
        self.grad_b = np.sum(grad_output, axis=0)
        return grad_output @ self.W


class TestLinear:
    def test_output_shape(self):
        """Linear layer output shape must be (batch, out_features)."""
        np.random.seed(0)
        layer = Linear(3, 5)
        x = np.random.randn(4, 3)
        out = layer.forward(x)
        assert out.shape == (4, 5)

    def test_single_sample(self):
        """Verify output matches manual computation."""
        np.random.seed(0)
        layer = Linear(2, 1)
        x = np.array([[1.0, 2.0]])
        out = layer.forward(x)
        expected = x @ layer.W.T + layer.b
        assert np.allclose(out, expected)

    def test_backward_shapes(self):
        """Gradients must match parameter shapes."""
        np.random.seed(0)
        layer = Linear(3, 5)
        x = np.random.randn(4, 3)
        layer.forward(x)
        grad_out = np.random.randn(4, 5)
        grad_in = layer.backward(grad_out)
        assert layer.grad_W.shape == layer.W.shape
        assert layer.grad_b.shape == layer.b.shape
        assert grad_in.shape == x.shape

    def test_backward_gradient_check(self):
        """Analytical gradient must match numerical gradient."""
        np.random.seed(42)
        layer = Linear(2, 3)
        x = np.random.randn(4, 2)
        target = np.random.randn(4, 3)

        # Forward + analytical backward
        out = layer.forward(x)
        loss = np.mean((out - target) ** 2)
        grad_out = 2.0 * (out - target) / out.size  # divide by total elements (np.mean averages over all)
        layer.backward(grad_out)

        # Numerical gradient for W[0, 0]
        h = 1e-5
        layer.W[0, 0] += h
        loss_plus = np.mean((layer.forward(x) - target) ** 2)
        layer.W[0, 0] -= 2 * h
        loss_minus = np.mean((layer.forward(x) - target) ** 2)
        layer.W[0, 0] += h  # restore
        numerical = (loss_plus - loss_minus) / (2 * h)

        assert abs(layer.grad_W[0, 0] - numerical) < 1e-5, \
            f"Gradient mismatch: analytical={layer.grad_W[0, 0]:.8f}, numerical={numerical:.8f}"


# ============================================================
# XOR Network
# ============================================================

class TestXORNetwork:
    def test_xor_convergence(self):
        """XOR network must converge to correct predictions."""
        np.random.seed(42)
        X = np.array([[0, 0], [0, 1], [1, 0], [1, 1]], dtype=np.float64)
        y = np.array([[0], [1], [1], [0]], dtype=np.float64)

        hidden = 8
        W1 = np.random.randn(hidden, 2) * np.sqrt(2.0 / 2)
        b1 = np.zeros(hidden)
        W2 = np.random.randn(1, hidden) * np.sqrt(2.0 / hidden)
        b2 = np.zeros(1)

        lr = 1.0
        for _ in range(5000):
            z1 = X @ W1.T + b1
            a1 = np.maximum(0, z1)
            z2 = a1 @ W2.T + b2
            a2 = 1 / (1 + np.exp(-z2))
            bs = X.shape[0]
            dz2 = (2.0 / bs) * (a2 - y) * a2 * (1 - a2)
            dW2 = dz2.T @ a1
            db2 = np.sum(dz2, axis=0)
            da1 = dz2 @ W2
            dz1 = da1 * (z1 > 0).astype(float)
            dW1 = dz1.T @ X
            db1 = np.sum(dz1, axis=0)
            W2 -= lr * dW2; b2 -= lr * db2
            W1 -= lr * dW1; b1 -= lr * db1

        predictions = a2
        rounded = np.round(predictions)
        assert np.all(rounded == y), f"XOR failed: predictions={predictions.ravel()}"

    def test_xor_loss_decreases(self):
        """Loss must decrease during training."""
        np.random.seed(42)
        X = np.array([[0, 0], [0, 1], [1, 0], [1, 1]], dtype=np.float64)
        y = np.array([[0], [1], [1], [0]], dtype=np.float64)

        hidden = 8
        W1 = np.random.randn(hidden, 2) * np.sqrt(2.0 / 2)
        b1 = np.zeros(hidden)
        W2 = np.random.randn(1, hidden) * np.sqrt(2.0 / hidden)
        b2 = np.zeros(1)

        losses = []
        lr = 1.0
        for _ in range(1000):
            z1 = X @ W1.T + b1; a1 = np.maximum(0, z1)
            z2 = a1 @ W2.T + b2; a2 = 1 / (1 + np.exp(-z2))
            loss = np.mean((a2 - y) ** 2)
            losses.append(loss)
            bs = X.shape[0]
            dz2 = (2.0 / bs) * (a2 - y) * a2 * (1 - a2)
            W2 -= lr * (dz2.T @ a1); b2 -= lr * np.sum(dz2, axis=0)
            da1 = dz2 @ W2; dz1 = da1 * (z1 > 0).astype(float)
            W1 -= lr * (dz1.T @ X); b1 -= lr * np.sum(dz1, axis=0)

        assert losses[-1] < losses[0], "Loss should decrease during training"
        assert losses[-1] < 0.01, f"Final loss {losses[-1]} too high"


# ============================================================
# Cross-Entropy Loss
# ============================================================

class TestCrossEntropy:
    def test_perfect_prediction(self):
        """Loss should be near 0 for perfect predictions."""
        logits = np.array([[100.0, 0.0, 0.0]])  # very confident in class 0
        target = np.array([0])
        probs = softmax(logits)
        loss = -np.mean(np.log(probs[np.arange(1), target] + 1e-10))
        assert loss < 0.01

    def test_uniform_prediction(self):
        """Loss for uniform distribution = log(num_classes)."""
        n_classes = 5
        logits = np.zeros((1, n_classes))  # uniform
        target = np.array([0])
        probs = softmax(logits)
        loss = -np.mean(np.log(probs[np.arange(1), target] + 1e-10))
        expected = np.log(n_classes)
        assert np.isclose(loss, expected, atol=1e-5)

    def test_gradient_direction(self):
        """Gradient should push probability toward correct class."""
        logits = np.array([[1.0, 2.0, 3.0]])
        target = np.array([0])  # correct class has lowest logit
        probs = softmax(logits)
        grad = probs.copy()
        grad[np.arange(1), target] -= 1
        grad /= 1  # batch_size = 1
        # Gradient for correct class should be negative (decrease logit → increase prob)
        assert grad[0, 0] < 0, "Gradient should push correct class probability up"

    def test_cross_entropy_batch(self):
        """Cross-entropy should work with batches."""
        logits = np.random.randn(8, 10)
        targets = np.random.randint(0, 10, size=8)
        probs = softmax(logits)
        loss = -np.mean(np.log(probs[np.arange(8), targets] + 1e-10))
        assert np.isfinite(loss)
        assert loss > 0


# ============================================================
# Modern Components: RoPE, RMSNorm, SwiGLU, GQA
# ============================================================

def precompute_rope_frequencies(dim, max_seq_len, base=10000.0):
    freqs = 1.0 / (base ** (np.arange(0, dim, 2).astype(float) / dim))
    positions = np.arange(max_seq_len).astype(float)
    angles = np.outer(positions, freqs)
    return np.cos(angles), np.sin(angles)

def apply_rope(x, cos, sin):
    result = np.empty_like(x)
    result[:, 0::2] = x[:, 0::2] * cos - x[:, 1::2] * sin
    result[:, 1::2] = x[:, 0::2] * sin + x[:, 1::2] * cos
    return result

def rms_norm(x, gamma, eps=1e-5):
    rms = np.sqrt(np.mean(x ** 2, axis=-1, keepdims=True) + eps)
    return (x / rms) * gamma

def swish(x):
    return x * (1.0 / (1.0 + np.exp(-x)))


class TestRoPE:
    def test_preserves_magnitude(self):
        """RoPE rotation should preserve vector magnitude."""
        dim = 8
        cos, sin = precompute_rope_frequencies(dim, 20)
        v = np.random.randn(1, dim)
        v_rot = apply_rope(v, cos[5:6], sin[5:6])
        assert np.isclose(np.linalg.norm(v), np.linalg.norm(v_rot), atol=1e-10)

    def test_identity_at_position_zero(self):
        """At position 0, rotation angle is 0 → no change."""
        dim = 8
        cos, sin = precompute_rope_frequencies(dim, 10)
        v = np.random.randn(1, dim)
        v_rot = apply_rope(v, cos[0:1], sin[0:1])
        assert np.allclose(v, v_rot, atol=1e-10)

    def test_different_positions_different_outputs(self):
        """Same vector at different positions should produce different results."""
        dim = 8
        cos, sin = precompute_rope_frequencies(dim, 10)
        v = np.random.randn(1, dim)
        v_pos1 = apply_rope(v, cos[1:2], sin[1:2])
        v_pos5 = apply_rope(v, cos[5:6], sin[5:6])
        assert not np.allclose(v_pos1, v_pos5)

    def test_output_shape(self):
        """RoPE should preserve input shape."""
        dim = 16
        cos, sin = precompute_rope_frequencies(dim, 10)
        x = np.random.randn(5, dim)
        out = apply_rope(x, cos[:5], sin[:5])
        assert out.shape == x.shape


class TestRMSNorm:
    def test_output_rms_is_one(self):
        """After RMSNorm, RMS of output should be ~1.0."""
        dim = 32
        x = np.random.randn(4, dim) * 5
        gamma = np.ones(dim)
        out = rms_norm(x, gamma)
        rms = np.sqrt(np.mean(out ** 2, axis=-1))
        assert np.allclose(rms, 1.0, atol=0.01)

    def test_preserves_shape(self):
        x = np.random.randn(3, 16)
        out = rms_norm(x, np.ones(16))
        assert out.shape == x.shape

    def test_gamma_scaling(self):
        """Gamma should scale the output."""
        dim = 8
        x = np.random.randn(2, dim)
        gamma_1 = np.ones(dim)
        gamma_2 = np.ones(dim) * 2.0
        out1 = rms_norm(x, gamma_1)
        out2 = rms_norm(x, gamma_2)
        assert np.allclose(out2, out1 * 2.0)

    def test_handles_zero_input(self):
        """Should not crash on near-zero input."""
        x = np.zeros((2, 8)) + 1e-10
        out = rms_norm(x, np.ones(8))
        assert np.all(np.isfinite(out))


class TestSwiGLU:
    def test_swish_at_zero(self):
        """Swish(0) = 0."""
        assert np.isclose(swish(np.array([0.0]))[0], 0.0)

    def test_swish_positive_for_positive(self):
        """Swish(x) > 0 for x > 0."""
        x = np.array([0.1, 1.0, 5.0])
        assert np.all(swish(x) > 0)

    def test_swish_smooth(self):
        """Swish should be smooth (no discontinuities)."""
        x = np.linspace(-5, 5, 1000)
        y = swish(x)
        # Check no jumps: max difference between consecutive points should be small
        diffs = np.abs(np.diff(y))
        assert np.max(diffs) < 0.1  # smooth function


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
