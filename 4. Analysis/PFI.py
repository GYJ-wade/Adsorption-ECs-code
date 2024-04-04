import joblib
from matplotlib import pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from eli5.sklearn import PermutationImportance
from sklearn.ensemble import AdaBoostRegressor
from sklearn.tree import DecisionTreeRegressor


def find_indices_in_first_array(first_array, second_array):
    indices = [first_array.index(value) for value in second_array]
    return indices


def main(path_name, name_label_use, name_label_use_plt, num_random, k):
    data_path = '..\\1. Data\\Data-{name}.csv'.format(name=path_name)

    data = pd.read_csv(data_path, header=0)

    x_tr, x_te, y_tr, y_te = train_test_split(data.iloc[:, 0:-2], data.iloc[:, [-1]],
                                              train_size=.80, random_state=num_random)

    feature_names = x_tr.columns.tolist()

    x_scaler, y_scaler = StandardScaler(), StandardScaler()
    x_train_scaled, y_train_scaled = x_scaler.fit_transform(x_tr), y_scaler.fit_transform(y_tr)
    x_test_scaled, y_test_scaled = x_scaler.transform(x_te), y_scaler.transform(y_te)

    x_all_scaled, y_all_scaled = np.concatenate([x_train_scaled, x_test_scaled], axis=0), np.concatenate([y_train_scaled, y_test_scaled], axis=0)

    model = joblib.load('..\\3. Model\\save\\model-log({name})-Ada.pkl'.format(name=path_name))

    y_pred_scaled = model.predict(x_test_scaled)
    y_pred = y_scaler.inverse_transform(y_pred_scaled.reshape(-1, 1)).flatten()

    rmse = np.sqrt(mean_squared_error(y_te, y_pred))
    r2 = r2_score(y_te, y_pred)
    print("Root Mean Squared Error: ", rmse)
    print("R-squared: ", r2)

    perm_importance = PermutationImportance(model, n_iter=10).fit(x_all_scaled, y_all_scaled)

    for feature, importance, importance_std in zip(feature_names, perm_importance.feature_importances_,
                                                   perm_importance.feature_importances_std_):
        print("{feature}: {importance}, {std}".format(feature=feature, importance=importance, std=importance_std))

    plt.figure(figsize=(12, 4), dpi=300)

    plt.rcParams['font.sans-serif'] = ['Times New Roman']
    for m in range(2):
        indices = find_indices_in_first_array(feature_names, name_label_use[m])
        importance_use = perm_importance.feature_importances_[indices]
        importance_std_use = perm_importance.feature_importances_std_[indices]
        importance_normalized = np.abs(importance_use) / np.sum(np.abs(importance_use))
        importance_std_normalized = importance_std_use * (importance_normalized[0] / np.abs(importance_use)[0])
        sorted_indices = np.argsort(importance_normalized)

        plt.subplot(1, 2, m + 1)
        ax = plt.gca()
        plt.grid(axis='x', linestyle='--', alpha=0.5)
        err_attr = {"elinewidth": 3, "ecolor": "cadetblue", "capsize": 4}
        plt.barh(range(len(name_label_use[m])), importance_normalized[sorted_indices], color='lightblue',
                 edgecolor='teal', xerr=importance_std_normalized[sorted_indices], error_kw=err_attr)
        for index, value in enumerate(importance_normalized[sorted_indices]):
            if index == 5 or (m == 1 and index == 4):
                ha, va = 'left', 0.02 + importance_std_normalized[sorted_indices[-1]]
            else:
                ha, va = 'left', 0.02 + importance_std_normalized[sorted_indices][index]
            plt.text(value + va, index, str(round(value, 3)), ha=ha, va='center', fontsize=15)
        plt.xticks(fontsize=15)
        plt.yticks(range(len(name_label_use[m])), np.array(name_label_use_plt[m])[sorted_indices], fontsize=15)
        if m == 0:
            plt.ylabel('Features', fontsize=20)
            plt.text(0.89, 0.05, ' ({}) '.format(num[m + k]), horizontalalignment="left",
                     fontsize='20', transform=ax.transAxes)
        x_max = importance_normalized[sorted_indices[-1]] + importance_std_normalized[sorted_indices[-1]] + 0.13
        plt.xlim([0, x_max])
    plt.tight_layout()
    plt.savefig('..\\5. Plot\\3. Interpretable analysis\\PFI_log({name})-0111.jpg'.format(name=path_name), dpi=300)
    plt.show()


name_label_plt = [[['C', 'Ash', '(O+N)/C', 'H/C', 'S$\mathregular{_{BET}}$', 'D$\mathregular{_{p}}$'],
                   ['D', 'MV', 'PSA', 'ST', 'FRB']],
                  [['C', 'Ash', '(O+N)/C', 'H/C', 'S$\mathregular{_{BET}}$', 'D$\mathregular{_{p}}$'],
                   ['D', 'MV', 'PSA', 'ST', 'FRB']]]

name_label = [
    [['C', 'Ash', '(O+N)/C', 'H/C', 'S_bet', 'Dp'], ['D', 'MV', 'PSA', 'ST', 'FRB']],
    [['C', 'Ash', '(O+N)/C', 'H/C', 'S_bet', 'Dp'], ['D', 'MV', 'PSA', 'ST', 'FRB']]]

num = ['a', 'b', 'c', 'd']

if __name__ == '__main__':
    main('K', name_label[0], name_label_plt[0], 216, 0)
    main('Qm', name_label[1], name_label_plt[1], 168, 1)
