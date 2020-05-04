import numpy as np
from scipy.special import softmax
import seaborn as sns
import matplotlib.pyplot as plt

if __name__ == '__main__':
    data_size = 100
    true_mu = 3
    true_sig = 1

    y_obs = np.random.normal(loc=true_mu, scale=true_sig, size=data_size)

    M = 2000000
    m = M//20  # M/m is usually around 20
    print(f'Generating {M} initial samples, and {m} re-samples')

    # sample M params from initial prior
    mu_prior = np.random.uniform(-10, 10, M) # mu ~ U(-10, 10)
    sig_prior = np.random.uniform(0.1, 20, M) # sig ~ U(0.1, 10)

    # calculate importance weights, assuming that we model y ~ N(mu, sig)
    log_weights = - data_size * np.log(sig_prior * np.sqrt(2 * np.pi)) \
                  - 1 / (2 * sig_prior ** 2) * np.sum([(y_obs[i] - mu_prior) ** 2 for i in range(data_size)], axis=0)
    weights = softmax(log_weights)

    # resample mu and sig using the above weights to get posterior
    mu_posterior = np.random.choice(mu_prior, m, p=weights)
    sig_posterior = np.random.choice(sig_prior, m, p=weights)

    # report summary stats
    print(f'True mu: {true_mu}')
    print(f'True sigma: {true_sig}')
    print(f'Mu posterior: mean={np.mean(mu_posterior):.3} - sd={np.std(mu_posterior):.3}')
    print(f'Sigma posterior: mean={np.mean(sig_posterior):.3} - sd={np.std(sig_posterior):.3}')

    # plot the new samples
    fig, axes = plt.subplots(1, 3, figsize=(9, 5))

    axes[0].set_title('Data')
    axes[1].set_title('Mu Posterior')
    axes[2].set_title('Sigma Posterior')

    sns.distplot(y_obs, ax=axes[0])
    sns.distplot(mu_posterior, ax=axes[1])
    sns.distplot(sig_posterior, ax=axes[2])
    plt.show()
