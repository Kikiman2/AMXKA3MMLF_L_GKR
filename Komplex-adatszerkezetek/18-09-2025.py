import numpy as np

# Sigmoid function (squashes any number into 0-1 range)
def sigmoid(z):
    return 1 / (1 + np.exp(-z))

# --- Training data (8 samples) ---
X = np.array([0, 0.5, 1, 1.5, 2, 2.5, 3, 3.5])   # hours studied
y = np.array([0, 0, 0, 0, 1, 1, 1, 1])          # 0=fail, 1=pass

# --- Initialize weight and bias ---
w = np.random.randn()   # random starting weight
b = np.random.randn()   # random starting bias

#begginer friendly value
learning_rate = 0.1

# --- Training loop ---
for epoch in range(200):  # repeat many times so it learns well
    for i in range(len(X)):
        x_i = X[i]
        y_i = y[i]

        # Step 1: prediction
        z = w * x_i + b
        y_hat = sigmoid(z)

        # Step 2: error
        error = y_i - y_hat

        # Step 3: update weight and bias
        w += learning_rate * error * x_i
        b += learning_rate * error

# --- Test the model ---
test_hours = 6.3
z = w * test_hours + b
prediction = sigmoid(z)

print("Probability of passing:", prediction)
print("Prediction (pass=1/fail=0):", 1 if prediction >= 0.5 else 0)
