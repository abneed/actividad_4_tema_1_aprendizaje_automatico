import numpy as np

def perceptron_train(X, y, lr=0.2, epochs=200):
	Xb = np.c_[np.ones((X.shape[0], 1)), X]
	w = np.zeros(Xb.shape[1])

	for _ in range(epochs):
		for xi, target in zip(Xb, y):
			pred = 1 if np.dot(w, xi) >= 0 else 0
			w += lr * (target - pred) * xi
	return w

def perceptron_predict(X, w):
	Xb = np.c_[np.ones((X.shape[0], 1)), X]
	return np.array([1 if np.dot(w, xi) >= 0 else 0 for xi in Xb])

# OR
X_or = np.array([[0,0],[0,1],[1,0],[1,1]])
y_or = np.array([0,1,1,1])
w_or = perceptron_train(X_or, y_or)
print("Prediccion de OR:", perceptron_predict(X_or, w_or))

# XOR
X_xor = np.array([[0,0],[0,1],[1,0],[1,1]])
y_xor = np.array([0,1,1,0])
w_xor = perceptron_train(X_xor, y_xor)
print("Prediccion de XOR:", perceptron_predict(X_xor, w_xor))