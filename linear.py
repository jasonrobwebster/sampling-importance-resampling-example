import numpy as np
from scipy.special import softmax
import seaborn as sns
import matplotlib.pyplot as plt

if __name__ == '__main__':
    data_size = 100
    true_grad = 3
    true_intercept = 1
    true_sig = 1

    x = np.linspace(0, 10, data_size)
    # y = m x + c
    y_obs = true_grad * x + true_intercept + np.random.normal(loc=0, scale=true_sig, size=data_size)

    M = 2000000
    m = M // 20  # M/m is usually around 20
    print(f'Generating {M} initial samples, and {m} re-samples')

    # sample M params from initial prior
    grad_prior = np.random.uniform(-10, 10, M)  # m ~ U(-10, 10)
    intercept_prior = np.random.uniform(-10, 10, M)  # c ~ U(-10, 10)
    sig_prior = np.random.uniform(0.1, 20, M)  # sig ~ U(0.1, 10)

    # calculate importance weights, assuming that we model y ~ N(mu, sig)
    exponent = 1 / (2 * sig_prior ** 2) \
               * np.sum([(y_obs[i] - (grad_prior * x[i] + intercept_prior)) ** 2 for i in range(data_size)], axis=0)

    log_weights = - data_size * np.log(sig_prior * np.sqrt(2 * np.pi)) - exponent

    weights = softmax(log_weights)

    # resample params using the above weights to get posterior
    grad_posterior = np.random.choice(grad_prior, m, p=weights)
    intercept_posterior = np.random.choice(intercept_prior, m, p=weights)
    sig_posterior = np.random.choice(sig_prior, m, p=weights)

    # report summary stats
    print(f'True gradient: {true_grad}')
    print(f'True intercept: {true_intercept}')
    print(f'True sigma: {true_sig}')
    print(f'Gradient posterior: mean={np.mean(grad_posterior):.3} - sd={np.std(grad_posterior):.3}')
    print(f'Intercept posterior: mean={np.mean(intercept_posterior):.3} - sd={np.std(intercept_posterior):.3}')
    print(f'Sigma posterior: mean={np.mean(sig_posterior):.3} - sd={np.std(sig_posterior):.3}')

    # plot the new samples
    fig, axes = plt.subplots(1, 4, figsize=(12, 3))

    axes[0].set_title('Data')
    axes[1].set_title('Gradient Posterior')
    axes[2].set_title('Intercept Posterior')
    axes[3].set_title('Sigma Posterior')

    axes[0].plot(x, y_obs, 'x')
    sns.distplot(grad_posterior, ax=axes[1])
    sns.distplot(intercept_posterior, ax=axes[2])
    sns.distplot(sig_posterior, ax=axes[3])
    plt.show()

    fig, ax = plt.subplots(figsize=(5, 5))
    ax.set_xlabel('Gradient')
    ax.set_ylabel('Intercept')
    ax.set_title('Joint distribution p(m, c)')
    sns.kdeplot(grad_posterior, intercept_posterior, shade=True, ax=ax)
    plt.show()
