import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.pyplot import figure
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split


def get_synthetic_data(number_observations):
    x1 = np.random.uniform(0.0,1.0,(number_observations,1))
    x2 = np.random.uniform(0.0,1.0,(number_observations,1))
    d = np.random.binomial(1,0.1,(number_observations,1))
    y0 = x1 + x2 + np.random.normal(0,0.1*x1+0.15*x2,(number_observations,1))
    y1 = x1 + x2 + np.random.normal(0,0.1*x1+0.15*(1-x2),(number_observations,1))
    y = (1-d)*y0 + d*y1
    syn_data = pd.DataFrame(np.concatenate((x1,x2,d,y),axis=1), columns=['x1','x2','d','y'])

    D = syn_data['d'] 
    y = syn_data['y']
    syn_data = syn_data.drop(['d','y'], axis=1)

    train_features, test_features, D_train, D_test, y_train, y_test = train_test_split(syn_data, D, y, test_size=0.2, random_state = 0)
    
    return train_features, test_features, D_train, D_test, y_train, y_test


def get_mse_coverage(mse_df):
    mse_cov = []
    cov = []
    samples_cov = []
    for count, datapoint in enumerate(mse_df.index):
        samples_cov.append(datapoint)
        mse_cov.append(np.sum(mse_df['mse'][samples_cov])/len(samples_cov))
        cov.append((count+1)/mse_df.shape[0])

    mse_cov_new = []
    cov_new = []
    for i in range(len(cov)):
        if cov[i] > 0.1:
            cov_new.append(cov[i])
            mse_cov_new.append(mse_cov[i])

    return mse_cov_new, cov_new


def train_and_test(train_features_mean, train_features_var, y_train_mean, test_features_mean, test_features_var, y_test_mean, D_test):
    regressor_mean = LinearRegression()
    # Train the mean network
    regressor_mean.fit(train_features_mean, y_train_mean)

    # Obtain the outcome for the variance network
    y_train_var = np.square(regressor_mean.predict(train_features_mean) - y_train_mean)

    regressor_var = LinearRegression()
    # Train the variance network
    regressor_var.fit(train_features_var, y_train_var)

    # Test
    mse = np.square(regressor_mean.predict(test_features_mean) - y_test_mean)
    predictions_variance = regressor_var.predict(test_features_var)

    mse_df = pd.DataFrame(mse.to_numpy(), columns=['mse'])
    mse_df['variance'] = predictions_variance
    mse_df['D'] = D_test.to_numpy().reshape(-1,1)
    mse_df = mse_df.sort_values(['variance'], ascending=[True])
    
    return mse_df


def main():
    number_observations = 100000
    train_features, test_features, D_train, D_test, y_train_mean, y_test_mean = get_synthetic_data(number_observations)

    baseline = train_and_test(train_features, train_features, y_train_mean, test_features, test_features, y_test_mean, D_test)
    sufficiency = train_and_test(train_features, train_features[['x1']], y_train_mean, test_features, test_features[['x1']], y_test_mean, D_test)

    baseline_mse_0, baseline_cov_0 = get_mse_coverage(baseline[baseline['D']==0])
    sufficiency_mse_0, sufficiency_cov_0 = get_mse_coverage(sufficiency[sufficiency['D']==0])
    baseline_mse_1, baseline_cov_1 = get_mse_coverage(baseline[baseline['D']==1])
    sufficiency_mse_1, sufficiency_cov_1 = get_mse_coverage(sufficiency[sufficiency['D']==1])

    # Gaussian (one-stage) baseline control
    plt.figure(figsize=((8, 6)), dpi=80)
    plt.axes((.15, .2, .83, .75))

    line1 = plt.plot(baseline_cov_0, baseline_mse_0, ':', linewidth = 3.0, label = 'Majority', color = '#3E9A0A')
    line2 = plt.plot(baseline_cov_1, baseline_mse_1, linewidth = 3.0, label = 'Minority', color = '#FF0000')

    ax = plt.gca()
    ax.set_xlabel("coverage",fontsize = 28)
    ax.set_ylabel("MSE",fontsize = 28)
    plt.xticks(fontsize=28)
    plt.yticks(fontsize=28)

    ax.set_yticks([0.00,0.01,0.02,0.03], minor=False) 
    handles, labels = ax.get_legend_handles_labels()
    lgd = ax.legend(handles, labels, loc='lower right', fontsize=28)
    plt.show()

    # Gaussian (one-stage) sufficiency control
    plt.figure(figsize=((8, 6)), dpi=80)
    plt.axes((.15, .2, .83, .75))
    line1 = plt.plot(sufficiency_cov_0, sufficiency_mse_0, ':', linewidth = 3.0, label = 'Majority', color = '#3E9A0A')
    line2 = plt.plot(sufficiency_cov_1, sufficiency_mse_1, linewidth = 3.0, label = 'Minority', color = '#FF0000')

    ax = plt.gca()
    ax.set_xlabel("coverage",fontsize = 28)
    ax.set_ylabel("MSE",fontsize = 28)
    plt.xticks(fontsize=28)
    plt.yticks(fontsize=28)

    ax.set_yticks([0.00,0.01,0.02,0.03], minor=False) 
    handles, labels = ax.get_legend_handles_labels()
    lgd = ax.legend(handles, labels, loc='lower right', fontsize=28)
    plt.show()


if __name__ == "__main__":
    main()