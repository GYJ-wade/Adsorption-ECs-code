import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt


data_K = pd.read_csv("..\\1. Data\\Data-K.csv", header=0)
data_Qm = pd.read_csv("..\\1. Data\\Data-Qm.csv", header=0)


def plot_histogram_and_density(data_, column_name, x_name, num_plot, bins=None, unit=None, ax=None):
    plt.rcParams['font.sans-serif'] = ['Times New Roman']
    if ax is None:
        fig, ax = plt.subplots(figsize=(6, 5))
    sns.histplot(data=data_[column_name], color='lightblue', bins=bins, ax=ax)
    ax.set_xlabel('{column_name} {unit}'.format(column_name=x_name, unit=unit), fontsize=20, labelpad=15)
    ax.set_ylabel('Count', fontsize=20, labelpad=15)
    ax.tick_params(labelsize=20)
    ax2 = ax.twinx()
    sns.kdeplot(data=data_[column_name], fill=True, color='teal', ax=ax2)
    ax2.set_ylabel('', fontsize=20)
    ax2.tick_params(labelsize=20)
    ax2.set_yticks([])
    plt.tight_layout()
    plt.savefig('..\\5. Plot\\1. Data visualization\\PHD-{}.jpg'.format(column_name), dpi=250)
    plt.show()


plot_histogram_and_density(data_K, 'log(K)', 'log K', 'b', 15, '')  # (g/mmol/min)
plot_histogram_and_density(data_K, 'K', 'K', 'a', 25, '(g/mmol/min)')
plot_histogram_and_density(data_Qm, 'log(Qm)', 'log Q$\mathregular{_m}$', 'd', 10, '')  # (mmol/g)
plot_histogram_and_density(data_Qm, 'Qm', 'Q$\mathregular{_m}$', 'c', 15, '(mmol/g)')
