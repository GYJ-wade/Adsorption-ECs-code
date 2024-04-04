import joblib
import numpy as np
import pandas as pd
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.model_selection import train_test_split
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.inspection import partial_dependence
from scipy.interpolate import splev, splrep
from sklearn.preprocessing import StandardScaler

name_title = ['K', 'Qm']
unit_label = [['(wt.%)', '', '(m$\mathregular{^2}$/g)'], ['(m$\mathregular{^2}$/g)', '(nm)', '(wt.%)']]
name_label = [['Ash', '(O+N)/C', 'S$\mathregular{_{BET}}$'], ['S$\mathregular{_{BET}}$', 'D$\mathregular{_{p}}$', 'Ash']]
num = ['a', 'b', 'c', 'd', 'e', 'f']

plt.rcParams['font.sans-serif'] = ['Times New Roman']
plt.rcParams['font.size'] = '15'
name_x = [['Ash', '(O+N)/C', 'S_bet'], ['S_bet', 'Dp', 'Ash']]
index_num = [[1, 2, 4], [4, 5, 1]]


def PDP_plot(name, num_rs, name_num):
    data_path = '..\\1. Data\\Data-{name}.csv'.format(name=name)

    data = pd.read_csv(data_path, header=0)

    x_tr, x_te, y_tr, y_te = train_test_split(data.iloc[:, 0:-2], data.iloc[:, [-1]],
                                              train_size=.80, random_state=num_rs)

    x_scaler, y_scaler = StandardScaler(), StandardScaler()
    x_train_scaled, y_train_scaled = x_scaler.fit_transform(x_tr), y_scaler.fit_transform(y_tr)
    x_test_scaled, y_test_scaled = x_scaler.transform(x_te), y_scaler.transform(y_te)
    y_train_scaled = y_train_scaled.ravel()
    y_test_scaled = y_test_scaled.ravel()

    feature_names = x_tr.columns.tolist()
    feature_name = name_x[name_num]
    x_all_scaled, y_all_scaled = np.concatenate([x_train_scaled, x_test_scaled], axis=0), np.concatenate(
        [y_train_scaled, y_test_scaled], axis=0)
    x_pd = pd.DataFrame(x_all_scaled, columns=feature_names)

    model = joblib.load('..\\3. Model\\save\\model-log({name})-Ada.pkl'.format(name=name))

    y_pred_scaled = model.predict(x_test_scaled)
    y_pred = y_scaler.inverse_transform(y_pred_scaled.reshape(-1, 1)).flatten()

    rmse = np.sqrt(mean_squared_error(y_te, y_pred))
    r2 = r2_score(y_te, y_pred)
    print("Root Mean Squared Error: ", rmse)
    print("R-squared: ", r2)

    plt.figure(figsize=(14, 4), dpi=300, facecolor='white')
    plt.rcParams['font.sans-serif'] = ['Times New Roman']

    # 作图
    for j in range(len(feature_name)):
        plt.subplot(1, 3, j + 1)
        plt.rcParams['font.sans-serif'] = ['Times New Roman']
        ax = plt.gca()
        plt.grid(visible=True, color='silver', linewidth=1, alpha=0.5)

        sns.set_theme(style="ticks", palette="deep", font_scale=1.3)

        pdp = partial_dependence(model, x_pd, [feature_name[j]],
                                 kind="both",
                                 method='brute',
                                 grid_resolution=100)

        column_index = index_num[name_num][j]

        mean_value = x_scaler.mean_[column_index]
        std_dev = x_scaler.scale_[column_index]
        x_value = pdp['values'][0] * std_dev + mean_value

        mean_value = y_scaler.mean_
        std_dev = y_scaler.scale_
        y_value = pdp['average'][0] * std_dev + mean_value

        plot_x = x_value
        plot_y = y_value

        plt.plot(plot_x, plot_y, color='orangered', alpha=0.6, linewidth=4)
        sns.rugplot(data=x_tr, x=feature_name[j], height=.06, color='r', alpha=0.3, linewidth=4)

        x_min = plot_x.min() - (plot_x.max() - plot_x.min()) * 0.1
        x_max = plot_x.max() + (plot_x.max() - plot_x.min()) * 0.1
        y_min = plot_y.min() - (plot_y.max() - plot_y.min()) * 0.2
        y_max = plot_y.max() + (plot_y.max() - plot_y.min()) * 0.3
        plt.xlabel(name_label[name_num][j] + ' ' + unit_label[name_num][j])
        plt.ylabel('Partial dependence')
        plt.xlim(x_min, x_max)
        plt.ylim(y_min, y_max)
        ax = plt.gca()
        plt.text(0.02, 0.87, ' ({}) '.format(num[j + name_num * 3]), horizontalalignment="left",
                 fontsize='20', transform=ax.transAxes)
        plt.rcParams['font.sans-serif'] = ['Times New Roman']
        plt.tight_layout()
        plt.subplots_adjust(wspace=0.45, hspace=0.8)
    # plt.tight_layout()
    plt.rcParams['font.sans-serif'] = ['Times New Roman']
    plt.savefig('..\\5. Plot\\3. Interpretable analysis\\PDP_log({name}).jpg'.format(name=name), dpi=300)
    plt.show()


num_rs_all = [216, 168]

if __name__ == '__main__':
    for i in range(2):
        PDP_plot(name_title[i], num_rs_all[i], i)
