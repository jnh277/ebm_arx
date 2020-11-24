import pandas as pd
import matplotlib.pyplot as plt
import pickle5 as pickle
import argparse

parser = argparse.ArgumentParser(description='Estimate NARX model for different n features / n samples rate.')
parser.add_argument('-n', '--data_length', default=100, type=int,
                    help='data length')
parser.add_argument('-s', '--sigma', default=0.3, type=float,
                    help='noise level sigma within range 0.1 to 1.0')
args, unk = parser.parse_known_args()

N = args.data_length
sigma = max(min(args.sigma, 1.0), 0.1)

with open('results/chen_comparison/sigma' + str(int(sigma * 10)) + 'N' + str(N) + '/data.pkl', "rb") as f:
    data = pickle.load(f)
df = pd.read_csv('results/chen_comparison/sigma'+str(int(sigma*10))+'N'+str(N)+'/evals.csv')
df2 = df.dropna() # remove NaNs which sometimes occured due to training of EB-NARX model


print("FCN BEST MSE: ", df2['fcn_test_mse'].min())
print("EBM BEST MSE: ", df2['ebm_test_mse'].min())
print('FCN # params: ', df2.loc[df2['fcn_test_mse']==df2['fcn_test_mse'].min(),'fcn_num_params'].to_numpy())
print('EBM # params: ', df2.loc[df2['ebm_test_mse']==df2['ebm_test_mse'].min(),'ebm_num_params'].to_numpy())

