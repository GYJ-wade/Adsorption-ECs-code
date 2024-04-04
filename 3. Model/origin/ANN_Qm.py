from sklearn.model_selection import GridSearchCV, train_test_split
from sklearn.metrics import mean_squared_error, r2_score
import pandas as pd
import numpy as np
from sklearn.neural_network import MLPRegressor
from sklearn.preprocessing import StandardScaler
import itertools

data_path = '..\\..\\1. Data\\Data-Qm.csv'
data = pd.read_csv(data_path, header=0)
x_tr, x_te, y_tr, y_te = train_test_split(data.iloc[:, 0:-2], data.iloc[:, [-2]],
                                          train_size=.80, random_state=168)

x_scaler, y_scaler = StandardScaler(), StandardScaler()
x_train_scaled, y_train_scaled = x_scaler.fit_transform(x_tr), y_scaler.fit_transform(y_tr)
x_test_scaled, y_test_scaled = x_scaler.transform(x_te), y_scaler.transform(y_te)

y_train_scaled = y_train_scaled.ravel()
y_test_scaled = y_test_scaled.ravel()

hidden_nodes = [50, 100, 200]
num_layers = 2
layer_combinations = []

for nodes in hidden_nodes:
    for _ in [1, 2]:
        combinations = itertools.product([nodes], repeat=_+1)
        layer_combinations.extend(combinations)

mlp_param_grid = {
    'hidden_layer_sizes': layer_combinations,
    'activation': ['relu', 'tanh', 'logistic'],
    'alpha': [0.001, 0.01, 0.02, 0.05],
    'learning_rate_init': [0.0005, 0.001, 0.005, 0.01, 0.05],
    'solver': ['lbfgs', 'sgd', 'adam'],
}

EPOCHS = 10000

mlp_model = MLPRegressor(max_iter=EPOCHS, early_stopping=True, n_iter_no_change=50)
mlp_grid_search = GridSearchCV(mlp_model, mlp_param_grid, scoring='r2', cv=5, verbose=3)
mlp_grid_search.fit(x_train_scaled, y_train_scaled)

print("Best Parameters: ", mlp_grid_search.best_params_)
print("Best Score: ", mlp_grid_search.best_score_)

best_rf_model = mlp_grid_search.best_estimator_
y_pred_scaled = best_rf_model.predict(x_test_scaled)
y_pred = y_scaler.inverse_transform(y_pred_scaled.reshape(-1, 1)).flatten()

rmse = np.sqrt(mean_squared_error(y_te, y_pred))
r2 = r2_score(y_te, y_pred)
print("Root Mean Squared Error: ", rmse)
print("R-squared: ", r2)

results = pd.DataFrame(mlp_grid_search.cv_results_)
results.to_csv('..\\score\\ANN_Qm.csv', index=False)
