import numpy as np
import pandas as pd

from scipy.integrate import odeint
from scipy.special import softmax

import matplotlib.pyplot as plt
import seaborn as sns


def ode(y, t, beta, alpha, gamma):
    s = y[0]
    e = y[1]
    i = y[2]

    ds = -beta * s * i
    de = beta * s * i - alpha * e
    di = alpha * e - gamma * i

    return [ds, de, di]


def ode_sample(y, t, beta, alpha, gamma):
    # assumes beta, alpha, and gamma are arrays of equal size
    s = y[:beta.size]
    e = y[beta.size:2*beta.size]
    i = y[2*beta.size:3*beta.size]

    ds = -beta * s * i
    de = beta * s * i - alpha * e
    di = alpha * e - gamma * i

    out = np.concatenate([ds, de, di])

    return out


if __name__ == '__main__':
    # collect randomised data
    true_beta = 3.5
    true_alpha = 1/5
    true_gamma = 1/2
    true_r0 = true_beta / true_gamma
    true_sigma = 0.1

    data_size = 30
    t = np.linspace(0, 30, data_size)
    y0 = [0.99, 0.01, 0]
    y = odeint(ode, y0, t, args=(true_beta, true_alpha, true_gamma))

    # we'll keep the infectious and recovered populations
    r = 1 - y[:, 0] - y[:, 1] - y[:, 2]
    i = y[:, 2]
    y = np.vstack([i, r]).T

    # remove the y0 value
    y = y[1:]

    # we'll add noise according to a log normal distribution
    log_yobs = np.random.normal(loc=np.log(y), scale=[true_sigma, true_sigma])

    M = 2000000
    m = M//20
    print(f"Calculating {M} samples and {m} resamples.")

    # give prior distributions
    beta_prior = np.random.uniform(0.1, 5, M)
    alpha_prior = np.random.uniform(0, 1, M)
    gamma_prior = np.random.uniform(0, 1, M)
    sig_prior = np.random.uniform(0.01, 2, M)

    # solve the ode with these samples
    y0 = np.zeros(3*M)
    y0[:M] = 0.99
    y0[M:2*M] = 0.01
    sol_sample = odeint(ode_sample, y0, t, args=(beta_prior, alpha_prior, gamma_prior))

    # calculate infectious and recovered
    r = 1 - sol_sample[:, :M] - sol_sample[:, M:2*M] - sol_sample[:, 2*M:]
    i = sol_sample[:, 2*M:]
    sol_sample = np.concatenate([[i], [r]]).transpose(1, 0, 2)
    print(f"solution shape: {sol_sample.shape}")

    # remove the y0 value
    sol_sample = sol_sample[1:]

    # sol_sample should now be a data_size x M matrix
    # we'll model the infected cases as a log normal distribution log I ~ N(log I, sigma)
    # we'll model recovered cases as a log normal distribution log R ~ N(log R, sigma)
    # we'll calculate the weights accordingly
    log_sample = np.log(sol_sample)
    log_weights_i = - data_size * np.log(sig_prior * np.sqrt(2 * np.pi)) \
                    - 1 / (2 * sig_prior ** 2) * np.sum((log_yobs[:, 0].reshape(-1, 1) - log_sample[:, 0]) ** 2, axis=0)
    log_weights_r = - data_size * np.log(sig_prior * np.sqrt(2 * np.pi)) \
                    - 1 / (2 * sig_prior ** 2) * np.sum((log_yobs[:, 1].reshape(-1, 1) - log_sample[:, 1]) ** 2, axis=0)
    log_weights = log_weights_i + log_weights_r
    weights = softmax(log_weights)

    # take m resamples according to weights to get posterior distribution
    beta_posterior = np.random.choice(beta_prior, m, p=weights)
    alpha_posterior = np.random.choice(alpha_prior, m, p=weights)
    gamma_posterior = np.random.choice(gamma_prior, m, p=weights)
    sig_posterior = np.random.choice(sig_prior, m, p=weights)

    # can also calculate r0
    r0_posterior = beta_posterior / gamma_posterior

    # report summary stats
    print(f"True beta: {true_beta}")
    print(f"True alpha: {true_alpha}")
    print(f"True gamma: {true_gamma}")
    print(f"True R0: {true_r0}")
    print(f"True sigma: {true_sigma}")
    print(f"Beta posterior: mean - {np.mean(beta_posterior):.3} std - {np.std(beta_posterior):.3}")
    print(f"Alpha posterior: mean - {np.mean(alpha_posterior):.3} std - {np.std(alpha_posterior):.3}")
    print(f"Gamma posterior: mean - {np.mean(gamma_posterior):.3} std - {np.std(gamma_posterior):.3}")
    print(f"R0 posterior: mean - {np.mean(r0_posterior):.3} std - {np.std(r0_posterior):.3}")
    print(f"Sigma posterior: mean - {np.mean(sig_posterior):.3} std - {np.std(sig_posterior):.3}")

    # construct pandas df of samples
    samples_df = pd.DataFrame({
        'Beta': beta_posterior,
        'Alpha': alpha_posterior,
        'Gamma': gamma_posterior,
        'R0': r0_posterior,
        'sigma': sig_posterior
    })

    # set seaborn for plotting
    sns.set(style='darkgrid')

    # plot the posterior samples
    fig, axes = plt.subplots(2, 3, figsize=(8, 6))

    axes = axes.flat

    axes[0].set_title('Data')
    axes[1].set_title('Beta Posterior')
    axes[2].set_title('Alpha Posterior')
    axes[3].set_title('Gamma Posterior')
    axes[4].set_title('R0 Posterior')
    axes[5].set_title('Sigma Posterior')

    axes[0].plot(t[1:], np.exp(log_yobs[:, 0]), 'x', c='C0')
    axes[0].plot(t[1:], y[:, 0], c='C0')
    axes[0].plot(t[1:], np.exp(log_yobs[:, 1]), 'x', c='C1')
    axes[0].plot(t[1:], y[:, 1], c='C1')
    sns.distplot(beta_posterior, ax=axes[1])
    sns.distplot(alpha_posterior, ax=axes[2])
    sns.distplot(gamma_posterior, ax=axes[3])
    sns.distplot(r0_posterior, ax=axes[4])
    sns.distplot(sig_posterior, ax=axes[5])

    plt.tight_layout()
    plt.show()

    # plot the pair grid
    plt.figure(figsize=(8, 8))
    g = sns.PairGrid(samples_df, corner=True)
    g.map_diag(sns.distplot)
    g.map_offdiag(sns.kdeplot)

    plt.show()
