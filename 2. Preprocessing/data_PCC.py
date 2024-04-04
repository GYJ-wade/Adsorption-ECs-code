import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from scipy import stats

data = pd.read_csv("..\\1. Data\\Data-PCC.csv", header=0)
data_corr = data.corr()

num_row = [
    ['C', 'Ash', '(O+N)/C', 'H/C', 'S_bet', 'Dp', 'Qm', 'log(Qm)', 'K', 'log(K)'],
    ['MW', 'D', 'MV', 'PSA', 'P', 'ST', 'HBA', 'HBD', 'FRB']]
num_column = [
    ['C', 'Ash', '(O+N)/C', 'H/C', 'S_bet', 'Dp', 'Qm', 'log(Qm)', 'K', 'log(K)'],
    ['MW', 'D', 'MV', 'PSA', 'P', 'ST', 'HBA', 'HBD', 'FRB']]
name_row = [
    ['C', 'Ash', '(O+N)/C', 'H/C', 'S$\mathregular{_{BET}}$',
     'D$\mathregular{_{p}}$', 'Q$\mathregular{_{m}}$', 'log Q$\mathregular{_{m}}$', 'K', 'log K'],
    ['MW', 'D', 'MV', 'PSA', 'P', 'ST', 'HBA', 'HBD', 'FRB']]
name_column = [
    ['C', 'Ash', '(O+N)/C', 'H/C', 'S$\mathregular{_{BET}}$',
     'D$\mathregular{_{p}}$', 'Q$\mathregular{_{m}}$', 'log Q$\mathregular{_{m}}$', 'K', 'log K'],
    ['MW', 'D', 'MV', 'PSA', 'P', 'ST', 'HBA', 'HBD', 'FRB']]
fig_name = ['(a)', '(b)']


def add_stars(p_value, ax, i, j):
    if p_value == 'None':
        ax.text(j + 0.5, i + 0.5, "None", fontsize=10, color='black', ha='center', va='center')
    elif p_value < 0.01:
        ax.text(j + 0.65, i + 0.35, "**", fontsize=12, color='black', ha='left', va='center')
    elif p_value < 0.05:
        ax.text(j + 0.65, i + 0.35, "*", fontsize=12, color='black', ha='left', va='center')


for i in range(len(fig_name)):
    corr_use = data_corr.loc[num_row[i], num_column[i]]
    plt.figure(figsize=(8, 8), dpi=150)
    plt.rcParams['font.sans-serif'] = ['Times New Roman']
    ax = sns.heatmap(corr_use, xticklabels=name_row[i], yticklabels=name_column[i],
                     annot=True, cmap="coolwarm", fmt='.2f', annot_kws={"fontsize":13})
    cbar = ax.collections[0].colorbar
    cbar.ax.tick_params(labelsize=15)
    for k in range(len(num_row[i])):
        data_row = data[num_row[i][k]]
        for j in range(len(num_column[i])):
            data_column = data[num_column[i][j]]
            common_indices = data[[num_row[i][k], num_column[i][j]]].dropna().index
            if len(common_indices) < 2:
                p_value = 'None'
                add_stars(p_value, ax, k, j)
            else:
                temp_series1 = data.loc[common_indices, num_row[i][k]]
                temp_series2 = data.loc[common_indices, num_column[i][j]]
                p_value = round(stats.pearsonr(temp_series1, temp_series2)[1], 3)
                add_stars(p_value, ax, k, j)
    plt.xticks(rotation=45, fontsize=15)
    plt.yticks(rotation=45, fontsize=15)
    plt.title(fig_name[i], fontsize=20, pad=15)
    plt.tight_layout()
    plt.savefig('..\\5. Plot\\1. Data visualization\\PCC-{name}.jpg'.format(name=fig_name[i]), dpi=500)
    plt.show()
