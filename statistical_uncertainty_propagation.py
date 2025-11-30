#!/usr/bin/env python3

# Small miniscript to show how to extrapolate statistical error through a
# blackbox function for fsrs parameter calculation

import numpy as np
import itertools as it
import uncertainties
import matplotlib.pyplot as plt
from scipy.stats import norm, kstest, shapiro, probplot, chi2
from fsrs import ReviewLog, Optimizer, Scheduler
import random
import time
from math import inf
import pandas as pd
import json


CONSTANT_SEED = False
NUM_FITTED_PARAMS = 21
N_MEASUREMENTS = 1000000
# 200 trials is enough to see the gaussian curve, albeit very messily
# 2000 trials is neough to see gaussian curve cleanly, albeit imperfectly
# 20000 is statistician's dream -- human eye to differentiate from theoretical gaussian curve becomes difficult

JIGGLES = 4
# DISTRIBUTION = "poisson"
DISTRIBUTION = "binomial"

# Runtime is O(NM) where N and M N_MEASUREMENTS and M is JIGGLES
# In actual deployment, after model is proven accurate, just O(M), as
# only 1 measurement is necessary once model is proven


if CONSTANT_SEED:
    np.random.seed(1729)  # In honor of fastest method to multiply 2 numbers
    random.seed(1729)


def jiggle_reviews(reviews):
    # Assumes a sparse matrix of radiation detectors each with either zero or a single detection event
    # Might be beneficial to bin the dectors and then have integerial number of detections in each bin...
    # Will try that if first approach fails and/or binning is easily done.
    # For starters, each detection is in its own bin
    if DISTRIBUTION == "binomial":  # Faster, probably slightly more accurate
        values = np.random.binomial(n=1, p=0.5, size=len(reviews)) * 2  # * 2 does make variance math accurate but probably unncessary
    elif DISTRIBUTION == "poisson":  # Easier to do math with, Binom converges to poisson at high number of measurements
        values = np.random.poisson(lam=1, size=len(reviews))
    else:
        raise ValueError(f"DISTRIBUTION must be 'binomial' or 'poisson', not {DISTRIBUTION}")
    return list(it.chain.from_iterable((datum, ) * num_detections for datum, num_detections in zip(reviews, values)))
    # yielded list is approx same length as len(reviews), but with various events duplicated and/or omitted, randomly


def fit_fsrs_params(reviews):
    """Actually returns a Scheduler object with fitted params"""
    cards, revlogs = zip(*reviews)
    print(revlogs)
    optimizer = Optimizer(revlogs)
    params = optimizer.compute_optimal_parameters()
    scheduler = Scheduler(optimal_parameters, enable_fuzzing=False)
    return scheduler


def get_reviews():
    """Get list of reviews for testing. Ideally from experimental measurements from a single deck of a single user."""
    # Load the Parquet files
    revlogs = pd.read_parquet("user_data/revlogs/user_id=1/data.parquet")
    cards = pd.read_parquet("user_data/cards/user_id=1/data.parquet")

    print(revlogs.head(500))  # Check the first few rows
    print(cards.head(500))  # Check the first few rows

    return reviews


def calculate_intervals(reviews, scheduler):
    """Calculate the interval of given reviews with given scheduler"""
    intervals = []
    for card, review_log in reviews:
        next_state, updated_card = scheduler.review(card, review_log)
        next_interval = next_state.interval
        intervals.append(next_interval)
    return intervals


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


def display_occasionally(time_to_pause, msg):
    global last_display_time
    now = time.time()
    try:
        last_display_time
    except NameError:
        last_display_time = -inf
    if now > last_display_time + time_to_pause:
        print(msg)
        last_display_time = now


def main():
    reviews = get_reviews()
    assert len(reviews) > 3
    z_scores = []
    while len(z_scores) < N_MEASUREMENTS:
        pct_complete = len(z_scores) / N_MEASUREMENTS * 100
        display_occasionally(3, f"Running trials {pct_complete:-.2f}% done")

        # Randomly assign fitting, checking, test review
        random.shuffle(reviews)
        size = len(reviews)//3
        fitting_reviews = reviews[:size]
        checking_reviews = reviews[size:2*size]
        test_reviews = reviews[-size:]

        jiggled_reviews = [jiggle_reviews(fitting_reviews) for _ in range(JIGGLES)]

        jiggled_schedulers = [fit_fsrs_params(reviews) for reviews in jiggled_reviews]
        checking_scheduler = fit_fsrs_params(checking_reviews)

        jiggled_intervals = [calculate_intervals(test_reviews, params) for params in  jiggled_params]
        checking_intervals = calculate_intervals(test_reviews, checking_params)

        fitted_intervals_n = np.mean(jiggled_intervals, axis=0)
        fitted_intervals_s = np.std(jiggled_intervals, axis=0, ddof=1)

        z_scores.extend([(checking_interval - fitted_interval_n)/fitted_interval_s
                         for checking_interval, fitted_interval_n, fitted_interval_s
                         in zip(checking_intervals, fitted_intervals_n, fitted_intervals_s)])

    evaluate_and_compare(z_scores)


if __name__ == "__main__":
    main()
