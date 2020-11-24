import pandas as pd

df = pd.read_csv('results/ced_comp/evals.csv')
df.to_csv('results/ced_comp/evals.csv')
df2 = df.dropna()

print("FCN BEST MSE: ", df2['fcn_test_mse'].min())
print("EBM BEST MSE: ", df2['ebm_test_mse'].min())
print('FCN # params: ', df2.loc[df2['fcn_test_mse']==df2['fcn_test_mse'].min(),'fcn_num_params'].to_numpy())
print('EBM # params: ', df2.loc[df2['ebm_test_mse']==df2['ebm_test_mse'].min(),'ebm_num_params'].to_numpy())

