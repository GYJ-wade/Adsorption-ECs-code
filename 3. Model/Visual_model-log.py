import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.ticker import MultipleLocator
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.preprocessing import StandardScaler
import joblib


def result_pred(x, y, name):
    rmse = np.sqrt(mean_squared_error(x, y))
    r2 = r2_score(y, x)
    print('{}, rmse:{:5.4f}, r2:{:5.4f}'.format(name, rmse, r2))


def plot_Test_Pred(x_tr, y_tr, x, y, r2_1, rmse_1, r2_2, rmse_2, num, unit, title, major, x_name):
    plt.rcParams['font.sans-serif'] = ['Times New Roman']
    ax = plt.gca()

    plt.grid(visible=True, color='silver', linewidth=1, alpha=0.5)
    plt.scatter(x_tr, y_tr, s=100, c='skyblue', alpha=0.6, marker='s')
    plt.scatter(x, y, s=100, c='teal', alpha=0.8, marker='o')
    plt.scatter([], [], c='w', alpha=0, marker='o')
    plt.scatter([], [], c='w', alpha=0, marker='o')

    plt.xticks(fontsize=20)
    plt.yticks(fontsize=20)
    plt.xlabel('Experimental log ' + x_name, fontsize='20', labelpad=8)
    plt.ylabel('Predicted log ' + x_name, fontsize='20', labelpad=8)
    x_major_locator = MultipleLocator(major)
    y_major_locator = MultipleLocator(major)
    ax.xaxis.set_major_locator(x_major_locator)
    ax.yaxis.set_major_locator(y_major_locator)

    plt.axis('square')
    xy_ratio_1 = 0.2
    xy_ratio_2 = 0.35
    x_min = min(x.min() - (x.max() - x.min()) * xy_ratio_1, x_tr.min() - (x_tr.max() - x_tr.min()) * xy_ratio_1)
    x_max = max(x.max() + (x.max() - x.min()) * xy_ratio_2, x_tr.max() + (x_tr.max() - x_tr.min()) * xy_ratio_2)
    y_min = min(y.min() - (y.max() - y.min()) * xy_ratio_1, y_tr.min() - (y_tr.max() - y_tr.min()) * xy_ratio_1)
    y_max = max(y.max() + (y.max() - y.min()) * xy_ratio_2, y_tr.max() + (y_tr.max() - y_tr.min()) * xy_ratio_2)
    xy_min = min(x_min, y_min)
    xy_max = max(x_max, y_max)
    plt.xlim([xy_min, xy_max])
    plt.ylim([xy_min, xy_max])

    plt.legend(['Train group',
                'Test group',
                'Test R$\mathregular{^2}$' + ':  {:4.3f}'.format(r2_2),
                'Test RMSE: {:4.3f}'.format(rmse_2)],
               loc='upper left', fontsize='13', labelspacing=0.2)
    _ = plt.plot([-10, 600], [-10, 600], color='grey')
    plt.text(0.85, 0.08, ' ({}) '.format(num), horizontalalignment="left",
             fontsize='20', transform=ax.transAxes)


plt.figure(figsize=(15, 10), dpi=100)
num_f = ['a', 'b', 'c', 'd', 'e', 'f']
unit = ['g/mmol/min', 'mmol/g']
name_title = ['log(K)', 'log(Qm)']
name_title_plt = ['log K', 'log Q$\mathregular{_{m}}$']
name_title_x = ['K', 'Q$\mathregular{_{m}}$']
name_model = ['RF', 'Ada', 'ANN']
num_Ada = [[70, 10, 3, 0.5, 330], [130, 4, 4, 0.001, 250]]
num_RF = [[25, 250, 170, 4, 1], [25, 170, 170, 2, 1]]
num_MLP = [[(50, 50), 'relu', 0.05, 0.01, 'adam'], [(50, 50), 'relu', 0.05, 0.01, 'adam']]
major_base = [1, 1.5]
num_lim = [[-3.5, 3.5], [-3, 1.5]]

for i in range(6):
    j = i % 3
    k = i // 3
    if k == 0:
        data_path = '..\\1. Data\\Data-K.csv'
        data = pd.read_csv(data_path, header=0)
        x_tr, x_te, y_tr, y_te = train_test_split(data.iloc[:, 0:-2], data.iloc[:, [-1]],
                                                  train_size=.80, random_state=216)
    elif k == 1:
        data_path = '..\\1. Data\\Data-Qm.csv'
        data = pd.read_csv(data_path, header=0)
        x_tr, x_te, y_tr, y_te = train_test_split(data.iloc[:, 0:-2], data.iloc[:, [-1]],
                                                  train_size=.80, random_state=168)

    x_scaler, y_scaler = StandardScaler(), StandardScaler()

    x_train_scaled, y_train_scaled = x_scaler.fit_transform(x_tr), y_scaler.fit_transform(y_tr)
    x_test_scaled, y_test_scaled = x_scaler.transform(x_te), y_scaler.transform(y_te)

    y_train_scaled = y_train_scaled.ravel()
    y_test_scaled = y_test_scaled.ravel()

    if j == 0:
        if k == 0:
            model = joblib.load('..\\3. Model\\save\\model-log(K)-RF.pkl')
        elif k == 1:
            model = joblib.load('..\\3. Model\\save\\model-log(Qm)-RF.pkl')
        y_tr_pred_scaled = model.predict(x_train_scaled)
        y_tr_pred = y_scaler.inverse_transform(y_tr_pred_scaled.reshape(-1, 1)).flatten()
        y_pred_scaled = model.predict(x_test_scaled)
        y_pred = y_scaler.inverse_transform(y_pred_scaled.reshape(-1, 1)).flatten()
        rmse_tr = np.sqrt(mean_squared_error(y_tr_pred, y_tr))
        r2_tr = r2_score(y_tr, y_tr_pred)
        rmse = np.sqrt(mean_squared_error(y_pred, y_te))
        r2 = r2_score(y_te, y_pred)
    elif j == 1:
        if k == 0:
            model = joblib.load('..\\3. Model\\save\\model-log(K)-Ada.pkl')
        elif k == 1:
            model = joblib.load('..\\3. Model\\save\\model-log(Qm)-Ada.pkl')
        y_tr_pred_scaled = model.predict(x_train_scaled)
        y_tr_pred = y_scaler.inverse_transform(y_tr_pred_scaled.reshape(-1, 1)).flatten()
        y_pred_scaled = model.predict(x_test_scaled)
        y_pred = y_scaler.inverse_transform(y_pred_scaled.reshape(-1, 1)).flatten()
        rmse_tr = np.sqrt(mean_squared_error(y_tr_pred, y_tr))
        r2_tr = r2_score(y_tr, y_tr_pred)
        rmse = np.sqrt(mean_squared_error(y_pred, y_te))
        r2 = r2_score(y_te, y_pred)
    elif j == 2:
        if k == 0:
            model = joblib.load('..\\3. Model\\save\\model-log(K)-ANN.pkl')
        elif k == 1:
            model = joblib.load('..\\3. Model\\save\\model-log(Qm)-ANN.pkl')
        y_tr_pred_scaled = model.predict(x_train_scaled)
        y_tr_pred = y_scaler.inverse_transform(y_tr_pred_scaled.reshape(-1, 1)).flatten()
        y_pred_scaled = model.predict(x_test_scaled)
        y_pred = y_scaler.inverse_transform(y_pred_scaled.reshape(-1, 1)).flatten()
        rmse_tr = np.sqrt(mean_squared_error(y_tr_pred, y_tr))
        r2_tr = r2_score(y_tr, y_tr_pred)
        rmse = np.sqrt(mean_squared_error(y_pred, y_te))
        r2 = r2_score(y_te, y_pred)

    plt.subplot(2, 3, i + 1)
    plt.gca().set_aspect('equal')
    plot_Test_Pred(y_tr.values.ravel(), y_tr_pred, y_te.values.ravel(), y_pred, r2_tr, rmse_tr, r2, rmse, num_f[i],
                   unit[k], name_model[j] + ' for ' + name_title_plt[k], major_base[k], name_title_x[k])
    plt.subplots_adjust(wspace=1.1, hspace=0.8)

plt.tight_layout(pad=1.3)
plt.savefig('..\\5. Plot\\2. Model\\model-plot-log.jpg', dpi=300)
plt.show()
