import pandas as pd

data_path = '..\\1. Data\\Data-All.csv'
data_read = pd.read_csv(data_path, header=3)
data_read['pH'].fillna(data_read['pH'].mode()[0], inplace=True)

data_PCC = data_read[['C', 'Ash', '(O+N)/C', 'H/C', 'S_bet', 'V_total', 'Dp',
                      'MW', 'D', 'MV', 'PSA', 'P', 'ST', 'HBA', 'HBD', 'FRB',
                      'T', 'pH', 'pH_i', 'C_0',
                      'Qm', 'log(Qm)', 'K', 'log(K)', 'R2']]

data_PCC_R2 = data_PCC[data_PCC['R2'] > 0.9]
data_PCC.drop(data_PCC.columns[-1], axis=1, inplace=True)
data_PCC_R2.drop(data_PCC_R2.columns[-1], axis=1, inplace=True)
data_PCC_R2.to_csv('..\\1. Data\\Data-PCC.csv', index=False)

data_read['pH'].fillna(data_read['pH'].mode()[0], inplace=True)
data_Qm = data_read[['C', 'Ash', '(O+N)/C', 'H/C', 'S_bet', 'Dp',
                     'D', 'MV', 'PSA', 'ST', 'FRB',
                     'T', 'pH', 'pH_i',
                     'Qm', 'log(Qm)', 'R2']]
data_Qm = data_Qm.dropna(subset=['Qm'])
data_Qm_R = data_Qm[data_Qm['R2'] > 0.9]
data_Qm_R = data_Qm_R.dropna(axis=0)
data_Qm_R.drop(data_Qm_R.columns[-1], axis=1, inplace=True)
data_Qm_R.to_csv('..\\1. Data\\Data-Qm.csv', index=False)

data_K = data_read[['C', 'Ash', '(O+N)/C', 'H/C', 'S_bet', 'Dp',
                    'D', 'MV', 'PSA', 'ST', 'FRB',
                    'T', 'pH', 'pH_i', 'C_0',
                    'K', 'log(K)', 'R2']]
data_K = data_K.dropna(subset=['K'])
data_K_R = data_K[data_K['R2'] > 0.9]
data_K_R = data_K_R.dropna(axis=0)
data_K_R.drop(data_K_R.columns[-1], axis=1, inplace=True)
data_K_R.to_csv('..\\1. Data\\Data-K.csv', index=False)
