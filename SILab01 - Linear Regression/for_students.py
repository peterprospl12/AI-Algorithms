import numpy as np
import matplotlib.pyplot as plt
from data import get_data, inspect_data, split_data


def calc_theta(x_train1, y_train1):
    m = x_train1.shape[0]
    X = x_train1.reshape(m, 1)
    X = np.append(np.ones((m, 1)), X, axis=1)

    y = y_train1
    y = y.reshape(m, 1)

    return np.dot(np.linalg.inv(np.dot(X.T, X)), np.dot(X.T, y))


def calc_mse(x_train1, y_train1, theta_best1):
    return 1 / x_train1.shape[0] * (
        np.sum(np.square(theta_best1[0] + theta_best1[1] * x_train1 - y_train1)))


def plot_learning_curve(cost_history):
    plt.plot(np.arange(len(cost_history)), cost_history, label='Training Cost')
    plt.title('Learning Curve')
    plt.xlabel('Number of Iterations')
    plt.ylabel('Cost')
    plt.legend()
    plt.show()


def calc_theta_gradient_descent(x_train1, y_train1, theta, learning_rate=0.001, num_iterations=1000):
    m = x_train1.shape[0]
    X = np.column_stack((np.ones(m), x_train1))
    y = y_train1.reshape(m, 1)
    cost_history = []
    cost_before = 0

    for i in range(num_iterations):
        prediction = np.dot(X, theta)
        theta = theta - (2 / m) * learning_rate * (X.T.dot(prediction - y))
        cost = calc_mse(x_train1, y_train1, theta)
        cost_history.append(cost)
        if abs(cost - cost_before) < 0.00000001:
            break
        cost_before = cost
    plot_learning_curve(cost_history)
    return theta


data = get_data()
inspect_data(data)

train_data, test_data = split_data(data)

# Simple Linear Regression
# predict MPG (y, dependent variable) using Weight (x, independent variable) using closed-form solution
# y = theta_0 + theta_1 * x - we want to find theta_0 and theta_1 parameters that minimize the prediction error

# We can calculate the error using MSE metric:
# MSE = SUM (from i=1 to n) (actual_output - predicted_output) ** 2

# get the columns
y_train = train_data['MPG'].to_numpy()
x_train = train_data['Weight'].to_numpy()

y_test = test_data['MPG'].to_numpy()
x_test = test_data['Weight'].to_numpy()

# TODO: calculate closed-form solution
theta = calc_theta(x_train, y_train)
theta_best = [theta[0][0], theta[1][0]]
print("[Train] C-F solution theta: ", theta_best)
# TODO: calculate error

mse = calc_mse(x_train, y_train, theta_best)
print("[Train] C-F Solution MSE: ", mse)
print("[Test] C-F Solution MSE: ", calc_mse(x_test, y_test, theta_best))

# plot the regression line
x = np.linspace(min(x_test), max(x_test), 100)
y = float(theta_best[0]) + float(theta_best[1]) * x
plt.plot(x, y)
plt.scatter(x_test, y_test)
plt.xlabel('Weight')
plt.ylabel('MPG')
plt.show()

# TODO: standardization

train_stand_x = (x_train - np.mean(x_train)) / np.std(x_train)
test_stand_x = (x_test - np.mean(x_train)) / np.std(x_train)

train_stand_y = (y_train - np.mean(y_train)) / np.std(y_train)
test_stand_y = (y_test - np.mean(y_train)) / np.std(y_train)

theta = calc_theta(train_stand_x, train_stand_y)
theta_best = [theta[0][0], theta[1][0]]
print("[Train] Standardized theta: ", theta_best)
mse = calc_mse(train_stand_x, train_stand_y, theta_best)
#print("[Train] Standardized MSE: ", mse)
print("[Test] Standardized MSE: ", calc_mse(test_stand_x, test_stand_y, theta_best))

x = np.linspace(min(test_stand_x), max(test_stand_x), 100)
y = float(theta_best[0]) + float(theta_best[1]) * x
plt.plot(x, y)
plt.scatter(test_stand_x, test_stand_y)
plt.xlabel('Weight_stand')
plt.ylabel('MPG_stand')
plt.show()

# TODO: calculate theta using Batch Gradient Descent

theta = np.random.randn(2, 1)
theta = calc_theta_gradient_descent(train_stand_x, train_stand_y, theta, 0.001, 15000)
print("[Train] Gradient descent stand theta:", [theta[0][0], theta[1][0]], "LR: 0.001, ITR 15000")
#print("[Train] Gradient descent stand MSE: ", calc_mse(train_stand_x, train_stand_y, theta))
print("[Test] Gradient descent stand MSE: ", calc_mse(test_stand_x, test_stand_y, theta))

#scaled_theta = theta.copy()
#scaled_theta[1] = scaled_theta[1] * np.std(y_train) / np.std(x_train)
#scaled_theta[0] = np.mean(y_train) - scaled_theta[1] * np.mean(x_train)
#scaled_theta = scaled_theta.reshape(-1)

# TODO: calculate error
#theta_best = scaled_theta
print("[Train] Scaled theta", theta_best)
#print("[Train] Scaled theta MSE ", calc_mse(x_train, y_train, theta_best))
print("[Test] Scaled theta MSE ", calc_mse(test_stand_x, test_stand_y, theta_best))


# plot the regression line
x = np.linspace(min(test_stand_x), max(test_stand_x), 100)
y = float(theta_best[0]) + float(theta_best[1]) * x
plt.plot(x, y)
plt.scatter(test_stand_x, test_stand_y)
plt.xlabel('Weight')
plt.ylabel('MPG')
plt.show()
