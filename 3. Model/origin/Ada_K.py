from sklearn.model_selection import GridSearchCV, train_test_split
from sklearn.ensemble import AdaBoostRegressor
from sklearn.metrics import mean_squared_error, r2_score
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.tree import DecisionTreeRegressor

data_path = '..\\..\\1. Data\\Data-K.csv'
data = pd.read_csv(data_path, header=0)
x_tr, x_te, y_tr, y_te = train_test_split(data.iloc[:, 0:-2], data.iloc[:, [-2]],
                                          train_size=.80, random_state=216)

x_scaler, y_scaler = StandardScaler(), StandardScaler()

x_train_scaled, y_train_scaled = x_scaler.fit_transform(x_tr), y_scaler.fit_transform(y_tr)
x_test_scaled, y_test_scaled = x_scaler.transform(x_te), y_scaler.transform(y_te)

y_train_scaled = y_train_scaled.ravel()
y_test_scaled = y_test_scaled.ravel()

adaboost_param_grid = {
    'estimator__max_depth': range(10, 310, 60),
    'estimator__min_samples_split': range(2, 12, 2),
    'estimator__min_samples_leaf': [1, 2, 3, 4, 5],
    'n_estimators': range(10, 410, 80),
    'learning_rate': [0.001, 0.005, 0.01, 0.1, 0.5]
}

ada_model = AdaBoostRegressor(estimator=DecisionTreeRegressor())

ada_grid_search = GridSearchCV(ada_model, adaboost_param_grid, scoring='r2', cv=5, verbose=3)
ada_grid_search.fit(x_train_scaled, y_train_scaled)

print("Best Parameters: ", ada_grid_search.best_params_)
print("Best Score: ", ada_grid_search.best_score_)

best_rf_model = ada_grid_search.best_estimator_
y_pred_scaled = best_rf_model.predict(x_test_scaled)
y_pred = y_scaler.inverse_transform(y_pred_scaled.reshape(-1, 1)).flatten()

mse = mean_squared_error(y_te, y_pred)
r2 = r2_score(y_te, y_pred)
print("Mean Squared Error: ", mse)
print("R-squared: ", r2)

results = pd.DataFrame(ada_grid_search.cv_results_)
results.to_csv('..\\score\\Ada_K.csv', index=False)
