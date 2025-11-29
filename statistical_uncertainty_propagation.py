#!/usr/bin/env python3

# Small miniscript to show how to extrapolate statistical error through a
# blackbox function for fsrs parameter calculation

import numpy as np
import itertools as it
import uncertainties
import matplotlib.pyplot as plt
from scipy.stats import norm, kstest, shapiro, probplot, chi2
from fsrs import ReviewLog, Optimizer
import random


CONSTANT_SEED = False
NUM_FITTED_PARAMS = 21
N_TRIALS = 5000
# 200 trials is enough to see the gaussian curve, albeit very messily
# 2000 trials is neough to see gaussian curve cleanly, albeit imperfectly
# 20000 is statistician's dream -- human eye to differentiate from theoretical gaussian curve becomes difficult

JIGGLES = 4

# DISTRIBUTION = "Poisson"
DISTRIBUTION = "Binomial"

# Runtime is O(NMP) where N and M and P are N_TRIALS and BINOM_JIGGLES and the number of reviews
# In actual deployment, after model is proven accurate, just O(MP)


if CONSTANT_SEED:
    np.random.seed(1729)  # In honor of fastest method to multiply 2 numbers
    random.seed(1729)


def jiggle_reviews(data, n_jiggles):
    # Assumes a sparse matrix of radiation detectors each with either zero or a single detection event
    # Might be beneficial to bin the dectors and then have integerial number of detections in each bin...
    # Will try that if first approach fails and/or binning is easily done.
    # For starters, each detector in its own bin
    for i in range(n_jiggles):
        if DISTRIBUTION == "Binomial":  # Faster, probably slightly more accurate
            values = np.random.binomial(n=1, p=0.5, size=len(data)) * 2  # * 2 does make variance math accurate but probably unncessary
        elif DISTRIBUTION == "Poisson":  # Easier to do math with, Binom converges to Poisson at high number of measurements
            values = np.random.poisson(lam=1, size=len(data))
        else:
            raise ValueError(f"DISTRIBUTION must be 'Binomial' or 'Poisson', not {DISTRIBUTION}")
        yield list(it.chain.from_iterable((datum, ) * num_detections for datum, num_detections in zip(data, values)))
        # yielded list is approx same length as len(data), but with various events duplicated and/or omitted, randomly


def fsrs_params_with_error(reviews) -> list[uncertainties.ufloat]:
    assert len(reviews) > NUM_FITTED_PARAMS**2  # This number is probably too big, but we avoid errors!
    assert JIGGLES >= 2  #  Can't propagate statistical variation from 1 run, also bessel correction div-by-zero
    statistically_jiggled_datasets = jiggle_reviews(reviews, n_jiggles=JIGGLES)
    # No idea if above is enough or too many samples, but will probably work.
    # Also, no idea what shape matrix is most efficient...
    # most_accurate_mu_calculation = calculate_fsrs_params(reviews)  # uses all fitting data
    jiggled_parameters = [fit_fsrs_params(dataset) for dataset in statistically_jiggled_datasets]

    # Below assumes correlated gaussian distribution. I bet it will be one. We need to test that though.
    means = np.mean(jiggled_parameters, axis=0)
    cov = np.cov(jiggled_parameters, rowvar=False)
    # Bessel correction for sampling bias
    cov *= ((JIGGLES / (JIGGLES - 1))) ** 4
    params_with_uncertainty = uncertainties.correlated_values(means, cov)
    return params_with_uncertainty


def fit_fsrs_params(reviews):
    """Calculate fitted FSRS parameters from reviews"""
    # Currently just toy data
    return np.random.normal(loc=0.0, scale=1.0, size=NUM_FITTED_PARAMS)


def get_reviews_somewhere():
    """Get list of reviews for testing. Ideally from experimental measurements from a single deck of a single user."""
    # Currently just toy data
    return list(range(5000))


def calculate_interval(review, fsrs_params):
    """Calculate the interval of a given review with the given fsrs parameters"""
    # Currently just toy data
    return sum(fsrs_params)


def evaluate_and_compare(real_data):
    synthetic_data = np.random.normal(0, 1, len(real_data))

    # === Statistical Tests ===
    print("=== Statistical Tests on Real Data ===")
    ks_stat, ks_p = kstest(real_data, "norm")
    shapiro_stat, shapiro_p = shapiro(real_data)

    print(f"Kolmogorov-Smirnov: statistic={ks_stat:.4f}, p={ks_p:.4f}")
    print(f"Shapiro-Wilk:       statistic={shapiro_stat:.4f}, p={shapiro_p:.4f}\n")

    # === Estimate Mean & Std with Confidence Intervals ===
    n = len(real_data)
    mu = np.mean(real_data)
    sigma = np.std(real_data, ddof=1)

    # Mean confidence interval
    sem = sigma / np.sqrt(n)
    mu_low = mu - 1.96 * sem
    mu_high = mu + 1.96 * sem

    # Variance confidence (Chi-square based) → converts to σ bounds
    dof = n - 1
    chi_low = chi2.ppf(0.025, dof)
    chi_high = chi2.ppf(0.975, dof)
    sigma_low = np.sqrt((dof * sigma**2) / chi_high)
    sigma_high = np.sqrt((dof * sigma**2) / chi_low)

    print("=== Fit Parameter Confidence ===")
    print(f"μ estimate: {mu:.4f}   (95% CI: {mu_low:.4f} → {mu_high:.4f})")
    print(f"σ estimate: {sigma:.4f} (95% CI: {sigma_low:.4f} → {sigma_high:.4f})")

    # === Plot ===
    plt.figure(figsize=(14, 6))

    # -------------------------------
    # Subplot 1: Histogram Comparison
    # -------------------------------
    plt.subplot(1, 2, 1)

    bins = np.linspace(min(real_data + synthetic_data),
                       max(real_data + synthetic_data), 30)

    plt.hist(real_data, bins=bins, alpha=0.5, density=True, label="Experimental")
    plt.hist(synthetic_data, bins=bins, alpha=0.5, density=True, label="Gaussian Sampling")

    x = np.linspace(bins[0], bins[-1], 300)

    # Best fit curve
    best_fit = norm.pdf(x, mu, sigma)
    plt.plot(x, best_fit, label=f"Gaussian Fit (μ={mu:.2f}, σ={sigma:.2f})", linewidth=2)

    # Confidence band using (μ_low..μ_high) AND (σ_low..σ_high)
    fit_low = norm.pdf(x, mu_low, sigma_high)
    fit_high = norm.pdf(x, mu_high, sigma_low)

    plt.fill_between(x, fit_low, fit_high, alpha=0.25, label="Fit 95% CI Band")

    # Ideal standard normal reference
    plt.plot(x, norm.pdf(x, 0, 1), "--", linewidth=2, label="Ideal N(0,1)")

    plt.title("Histogram: Measured data vs. Gaussian Sampling")
    plt.xlabel("Z-score")
    plt.ylabel("Density")
    plt.legend()
    plt.grid(alpha=0.3)

    # -------------------------------
    # Subplot 2: Q-Q Style Comparison
    # -------------------------------
    plt.subplot(1, 2, 2)

    sorted_real = np.sort(real_data)
    sorted_synth = np.sort(synthetic_data)

    theoretical = norm.ppf(np.linspace(0.001, 0.999, n))

    # Real observations
    plt.scatter(theoretical, sorted_real, s=20, alpha=0.7, label="Experimental")

    # Synthetic reference points
    plt.scatter(theoretical, sorted_synth, s=20, alpha=0.7, label="Gaussian Sampling")

    # 1:1 perfect normal line
    lo, hi = min(min(theoretical), min(sorted_real)), max(max(theoretical), max(sorted_real))
    plt.plot([lo, hi], [lo, hi], 'r--', label="Perfect Fit Line")

    plt.title("Q-Q Comparison: Measured data vs. Gaussian Sampling")
    plt.xlabel("Theoretical Quantiles")
    plt.ylabel("Observed Quantiles")
    plt.legend()
    plt.grid(alpha=0.3)

    plt.tight_layout()
    plt.show()


def main():
    reviews = get_reviews_somewhere()
    assert len(reviews) > 3
    z_scores = []
    for i in range(N_TRIALS):
        print(f"Running trial {i}")

        # Randomly assign fitting, checking, test review
        random.shuffle(reviews)
        halfway_split = (len(reviews)-1)//2  # 3, 4 -> 1
        fitting_reviews = reviews[:halfway_split]
        checking_reviews = reviews[halfway_split:2*halfway_split]  # to either 2nd- or 3rd-to-alst
        test_review = reviews[-1]

        fitted_params = fsrs_params_with_error(fitting_reviews)
        checking_params = fit_fsrs_params(checking_reviews)

        fitted_interval = calculate_interval(test_review, fitted_params)
        checking_interval = calculate_interval(test_review, checking_params)

        z_scores.append((checking_interval - fitted_interval.n)/fitted_interval.s)

    evaluate_and_compare(z_scores)


if __name__ == "__main__":
    main()
