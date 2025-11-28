VERSION FIRST DRAFT

The end-goal of this script is to demonstrate the feasibility of
calculating the statistical error associated with calculating FSRS
parameters and how to extrapolate that uncertainty through the
interval calculation process, such that users can only be given
intervals which are not too long due to poor FSRS calibration.

As of the time of writing, it assumes that the statistics associated
with a single review are equivalent to the statistics of a radiaition
detection event in a radiation detector, and thus follows a Poisson
distribution with λ=1.

This is the equivalent of an n-dimensional detector array which is
sparse where virtually all detectors have 0 events.

-----------------------------

Feed into the script N reviews ( > 2 * N_PARAMS^2 ).  It then runs
N_TRIALS over the review (only 1 is necessary in deployment for end
users if we can accurately calculate the distribution exactly, which we probably can)
Below N_TRIALS < 20, it is hard to see Gaussian features.
At N_TRIALS = 200, Gaussian features should be somewhat visible.
At N_TRIALS = 2000, Gaussian features should be very clear.
At N_TRIALS = 20k, Gaussian features should be imperceptably
different to human eye, or close to it.

For each trial, it splits the data into 2 random datasets of equal
size, one for fitting, and one for testing.

With the fitting data set, it will do a bunch of statistical jiggling
on the data (hopefully accurately) and then estimate (hopefully
accurately) the statistical fluctuations associated with the
calculated FSRS parameters, and then (hoepfully accurately)
propagates that statistical uncertainty through the interval
calculation.  What you are left with is a prediction of an interval,
as well as a stastical error boudns for it.

Then, on the testing data set, it will do the standard calculation of
FSRS parameters to calculate an interval for the calculated parameters.

Then, a Z-score for how far apart the test value differs from the
fitting value (divided by statistical uncertainty) is calculated.

It does this N_TRIALS times, each time calculating a Z-score,
building up an array of calculated Z-scores for each random splitting of the data.

Then, data is shown to the user showing how accurate the calculated
Z-scores match that of a simulated perfect Gaussian distribution. 
The perect Gaussian distribution samples N_TRIALS datapoints so it
will have identical statistical fluctuations (but not inherent error)
to the test data.

That is, the two distributions should look similar. And you should be
able to turn N_TRIALS up to get clearer and clearer comparisons.

----------------

As of right now, all data is semi-deterministic toy data so
everything looks Gaussian no matter what you do.


------------------

To execute (probably, not tested)

    git clone <repo-url>
    cd repo
    python -m venv .venv
    source .venv/bin/activate  # Windows: .venv\Scripts\activate
    pip install -r requirements.txt
    ./statistical_uncertainty_propagation.py
