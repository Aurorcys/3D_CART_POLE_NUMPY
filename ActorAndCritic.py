import numpy as np

#xaiver initialization
scale1 = np.sqrt(2.0 / 8)
scale2 = np.sqrt(2.0 / 128)


class Actor:
    def __init__(self):
        scale1 = np.sqrt(2.0 / 8)
        scale2 = np.sqrt(2.0 / 128)
        self.W1 = np.random.randn(8, 128) * scale1
        self.b1 = np.zeros((1, 128))
        self.W2 = np.random.randn(128, 128) * scale2
        self.b2 = np.zeros((1, 128))
        self.W3 = np.random.randn(128, 2) * 0.01

    def forward(self, state, std_fixed):
        if state.ndim == 1:
            state = state.reshape(1, -1)
        self.state = state
        self.z1 = state @ self.W1 + self.b1
        self.a1 = np.tanh(self.z1)
        self.z2 = self.a1 @ self.W2 + self.b2
        self.a2 = np.tanh(self.z2)
        self.mean = self.a2 @ self.W3
        return self.mean, std_fixed

    def backward(self, d_mean, lr):
        d_W3 = self.a2.T @ d_mean
        d_a2 = d_mean @ self.W3.T
        d_z2 = d_a2 * (1 - self.a2 ** 2)
        d_W2 = self.a1.T @ d_z2
        d_b2 = np.sum(d_z2, axis=0, keepdims=True)
        d_a1 = d_z2 @ self.W2.T
        d_z1 = d_a1 * (1 - self.a1 ** 2)
        d_W1 = self.state.T @ d_z1
        d_b1 = np.sum(d_z1, axis=0, keepdims=True)
        d_W1 = np.clip(d_W1, -1.0, 1.0)
        d_W2 = np.clip(d_W2, -1.0, 1.0)
        d_W3 = np.clip(d_W3, -1.0, 1.0)
        self.W1 -= lr * d_W1
        self.b1 -= lr * d_b1
        self.W2 -= lr * d_W2
        self.b2 -= lr * d_b2
        self.W3 -= lr * d_W3

class Critic:
    def __init__(self):
        scale1 = np.sqrt(2.0 / 8)
        scale2 = np.sqrt(2.0 / 128)
        self.W1 = np.random.randn(8, 128) * scale1
        self.b1 = np.zeros((1, 128))
        self.W2 = np.random.randn(128, 128) * scale2
        self.b2 = np.zeros((1, 128))
        self.W3 = np.random.randn(128, 1) * 0.01
        self.b3 = np.zeros((1, 1))

    def forward(self, state):
        if state.ndim == 1:
            state = state.reshape(1, -1)
        self.state = state
        self.z1 = state @ self.W1 + self.b1
        self.a1 = np.tanh(self.z1)
        self.z2 = self.a1 @ self.W2 + self.b2
        self.a2 = np.tanh(self.z2)
        return self.a2 @ self.W3 + self.b3

    def backward(self, states, returns, lr):
        values = self.forward(states)
        d_loss = 2 * (values.squeeze() - returns) / len(returns)
        d_loss = d_loss.reshape(-1, 1)
        d_W3 = self.a2.T @ d_loss
        d_b3 = np.sum(d_loss, axis=0, keepdims=True)
        d_a2 = d_loss @ self.W3.T
        d_z2 = d_a2 * (1 - self.a2 ** 2)
        d_W2 = self.a1.T @ d_z2
        d_b2 = np.sum(d_z2, axis=0, keepdims=True)
        d_a1 = d_z2 @ self.W2.T
        d_z1 = d_a1 * (1 - self.a1 ** 2)
        d_W1 = self.state.T @ d_z1
        d_b1 = np.sum(d_z1, axis=0, keepdims=True)
        self.W1 -= lr * d_W1
        self.b1 -= lr * d_b1
        self.W2 -= lr * d_W2
        self.b2 -= lr * d_b2
        self.W3 -= lr * d_W3
        self.b3 -= lr * d_b3