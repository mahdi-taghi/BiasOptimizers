import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import seaborn as sns
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.optimizers import Adam


P = 850
E = 120e9
nu = 0.25
#creat dataset
sigma_max_values = np.linspace(10e6, 70e6, 4)
delta_u_values = np.linspace(0.2e-3, 0.7e-3, 4)
alpha_values = np.linspace(2, 7, 3)

def damage_non_linear(delta, delta_max, delta_u, alpha):
    if delta <= delta_max:
        return 0
    else:
        return 1 - ((delta_max / delta) * (1 - np.exp(-alpha * (delta - delta_max) / (delta_u - delta_max))))

def traction_linear(delta, sigma_max, delta_max, delta_u, D):
    if delta <= delta_max:
        return sigma_max * (delta / delta_max)
    else:
        return sigma_max * (1 - D)

data = []
for sigma_max in sigma_max_values:
    for delta_u in delta_u_values:
        delta_max = 0.01 * delta_u
        for alpha in alpha_values:
            for delta in np.linspace(0, delta_u, 50):
                D = damage_non_linear(delta, delta_max, delta_u, alpha)
                T = traction_linear(delta, sigma_max, delta_max, delta_u, D)
                data.append([sigma_max, delta_u, delta_max, alpha, delta, D, T])

columns = ['sigma_max', 'delta_u', 'delta_max', 'alpha', 'delta', 'damage', 'traction']
df = pd.DataFrame(data, columns=columns)

df.to_csv('data.csv', index=False)

x = df[['sigma_max', 'delta_u', 'delta_max', 'alpha', 'delta']]
y = df['damage']

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=42)
x_train.to_csv('X_train.csv', index=False)
x_test.to_csv('X_test.csv', index=False)
y_train.to_csv('y_train.csv', index=False)
y_test.to_csv('y_test.csv', index=False)
print("x_train shape:", x_train.shape)
print("y_train shape:", y_train.shape)
print("x_test shape:", x_test.shape)
print("y_test shape:", y_test.shape)
