"""
Simulate Genshin's Gacha Probabilities.

NB: Using numba's jit to speed up calcs!
"""
from numba import jit
import matplotlib.pyplot as plt
from scipy.stats import geom
import numpy as np
import plotly.graph_objects as go

# Global - Number of 5 stars we wish to obtain from the simulations. Actual nr of wishes is more than this!!
N = 10000000


@jit(nopython=True, parallel=True)
def simulate(cutoff=1):
    """
    Simulates getting "a" five star, i.e. no specific 5 star.
    :param cutoff: The cut-off range for when we start wishing
    :return: Numpy array of counts from when we got the 5 star, i.e. the number of wishes associated with that 5 star
    """

    # Pre-allocate the array
    five_star = np.zeros((N,))

    # Simulate random uniform numbers
    random_numbers = np.random.uniform(0, 1, N * 90)  # We use N * 90 since hard pity is at 90 pulls, i.e. upper bound

    for i in range(N):
        for j in range(cutoff, 91):
            if random_numbers[i * 90 + j] <= get_rate(j):
                five_star[i] = j
                break

    return five_star


@jit(nopython=True, parallel=True)
def simulate_joint():
    # Pre-allocate the array
    five_star = np.zeros((N, 2))

    # Simulate random uniform numbers
    random_numbers = np.random.uniform(0, 1, N * 90 * 2)  # We use N * 90 since hard pity is at 90 pulls, i.e. upper bound
    random_numbers = random_numbers.reshape((N * 90, 2))

    for i in range(N):
        for k in range(2):
            for j in range(1, 91):
                if random_numbers[i * 90 + j, k] <= get_rate(j):
                    five_star[i, k] = j
                    break

    return five_star


@jit(nopython=True, parallel=True)
def simulate_specific_5star():
    # Pre-allocate the array
    five_star = np.zeros((N,))

    # Simulate random uniform numbers
    random_numbers = np.random.uniform(0, 1, N * 180)  # We use N * 90 since hard pity is at 90 pulls, i.e. upper bound
    random_numbers = random_numbers.reshape((N * 180))

    # Has previous 5 star variable - i.e. guarantee condition for losing 50/50 on next successful attempt.
    guarantee = False
    for i in range(N):
        count = 1
        for j in range(1, 180):
            # If this triggers, we got a 5 star
            if random_numbers[i * 180 + j] <= get_rate(count):

                # 50/50, if we lose we assign a "5 star" and update guarantee rule.
                cond_1 = np.random.rand() >= 0.5

                if cond_1 and (not guarantee):
                    guarantee = True

                    # Reset the wish count
                    count = 0
                else:
                    # Won 50/50 or have previous non-event exclusive 5 star
                    five_star[i] = j

                    # Reset the loop & guarantee rule
                    guarantee = False
                    break

            # Update the wish count
            count += 1

    return five_star


@jit(nopython=True, parallel=True)
def simulate_5050():
    """
    Simulate the new "50/50" for Capturing Radiance. Code is written to work with jit, not to be "neat" xD.
    For example, don't change things to:
        cond_2 = cond_3 = False
    Since then it will break the jit compilation, even though it is more readable and "Pythonic" (whatever that means)

    :return: Tuple of ints showing the various counts.
    """

    # Pre-allocate the array - the number of 5 stars we wish to pull for
    five_star = np.zeros((N,))

    # Simulate random uniform numbers
    random_numbers = np.random.uniform(0, 1, N * 180)  # We use N * 90 since hard pity is at 90 pulls, i.e. upper bound
    random_numbers = random_numbers.reshape((N * 180))

    # Counters for the 50/50 chances, SUM MUST EQUAL N
    counter_no_5050 = 0
    counter_5050 = 0
    counter_7525 = 0
    counter_100_zero = 0

    # Has previous 5 star variable
    guarantee = False
    failed_first_5050 = False
    failed_second_5050 = False
    failed_third_5050 = False
    for i in range(N):
        count = 1
        for j in range(1, 180):
            # If this triggers, we got a 5 star
            if random_numbers[i * 180 + j] <= get_rate(count):

                # Custom logic for Radiance Capture
                test_val = np.random.rand()  # Random number to compare with after winning "a" 5 star.
                cond_2 = False
                cond_3 = False
                cond_4 = False  # Allocate conditions for changes in 50/50 rate

                if failed_third_5050 and (not guarantee):
                    # Super unlucky, failed three 50/50 before, so now capturing radiance is 100%.
                    cond_4 = True
                    cond_1 = False

                    # Tally
                    counter_100_zero += 1

                elif failed_second_5050 and (not guarantee):
                    # Failed the first two 50/50, so we increase the odds!
                    cond_3 = test_val >= 0.75
                    cond_1 = False

                    # Tally
                    if cond_3:
                        counter_7525 += 1
                    else:
                        # We won 75/25 & reset
                        failed_third_5050 = False
                elif failed_first_5050 and (not guarantee):
                    # Failed first 50/50 - no changes to the rate!
                    cond_1 = cond_2 = test_val >= 0.5
                    if cond_2:
                        counter_5050 += 1
                    else:
                        # We won 50/50
                        failed_second_5050 = False
                else:
                    # Normal 50/50, i.e. we lost it!
                    cond_1 = test_val >= 0.5
                    if not cond_1:
                        failed_first_5050 = False

                # 50/50, if we lose we assign a "5 star" and update guarantee rule.
                if cond_1 and (not guarantee) and not (cond_2 or cond_3 or cond_4):
                    """
                    LOST FIRST 50/50
                    """
                    guarantee = True

                    # Reset the wish count
                    count = 0

                    # Record
                    failed_first_5050 = True

                    # Tally
                    counter_5050 += 1

                elif cond_2 and (not guarantee):
                    """
                    LOST SECOND 50/50 in a row
                    """
                    guarantee = True

                    # Reset the wish count
                    count = 0

                    # Record
                    failed_second_5050 = True
                    failed_first_5050 = False

                elif cond_3 and (not guarantee):
                    """
                    LOST Third 75/25
                    """
                    guarantee = True

                    # Reset the wish count
                    count = 0

                    # Record
                    failed_third_5050 = True
                    failed_second_5050 = False
                    failed_first_5050 = False

                elif cond_4 and (not guarantee):
                    """
                    GUARANTEE FROM CAPTURING RADIANCE - 100%
                    """
                    guarantee = True  # Don't change this line

                    # Record the 5 event-star count
                    five_star[i] = j

                    # Reset all rules
                    failed_first_5050 = False
                    failed_second_5050 = False
                    failed_third_5050 = False

                else:
                    # Won 50/50 or have previous non-event exclusive 5 star
                    five_star[i] = j

                    # Reset the loop & guarantee rule
                    if not guarantee:
                        counter_no_5050 += 1

                    guarantee = False

                    break

            # Update the wish count
            count += 1

    return counter_5050, counter_7525, counter_100_zero, counter_no_5050


# Define the pull rate function
@jit(nopython=True)
def get_rate(x: np.float):
    # Normal pulls
    if x < 74:
        return 0.006
    # Soft pity -> Hard Pity Ramp: Cumulatively add 6% per wish past 74.
    elif 74 <= x < 90:
        return (x - 73) * 0.06 + 0.006
    else:
        return 1.0


# For the soft pity, we need to add this to the probabilities - recursive function since it depends on the previous
# rates
def get_rate_mult(x: np.float):
    if x == 75:
        return 1 - get_rate(74)
    else:
        return (1 - get_rate(x - 1)) * get_rate_mult(x - 1)


# Define a probability function
def prob(wish_nr: np.ndarray, cumulative=False, rate=False):
    """
    Calculates the probability density/cumulative density function as a function of the number of wishes between
    1 and 90.
    :param wish_nr: The amount of wishes since the start, e.g. 40 means your 40th wish since last 5 star
    :param cumulative: Whether the output should be a CDF (True) or PDF (False)
    :param rate: Whether the output should be the rate associated with the xth pull
    :return: Numpy array of probabilities OR rates
    """

    # Pre-allocate the array
    output = np.zeros((wish_nr.size,))

    # Cumulative Probabilities above 75 need to be calculated differently using recursion
    plus_75 = [(1 - geom.cdf(73, 0.006)) * get_rate_mult(i) * get_rate(i) for i in range(75, 90)]
    plus_75 = np.cumsum(plus_75) + geom.cdf(73, 0.006) + (1 - geom.cdf(73, 0.006)) * get_rate(74)

    # Calculate cumulative probabilities
    for ii in range(wish_nr.size):
        if wish_nr[ii] < 74:
            # In simple english: 1 - (1 - 0.006) ^ ii, or 1 - probability of not getting a 5 star in ii pulls
            output[ii] = geom.cdf(wish_nr[ii], 0.006)
        elif wish_nr[ii] == 74:
            # In simple english: Probability of getting it in 73 pulls, then getting it on the 74th pull which is
            # the rate of getting it multiplied by the probability of not getting it in 73 pulls
            output[ii] = geom.cdf(73, 0.006) + (1 - geom.cdf(73, 0.006)) * get_rate(74)
        elif 75 <= wish_nr[ii] < 90:
            # Some recursive magic xD.
            output[ii] = plus_75[int(wish_nr[ii]) - 75]
        else:
            output[ii] = 1

        if rate:
            output[ii] = get_rate(wish_nr[ii])

    if not cumulative:
        output = np.insert(np.diff(output), 0, 0.006)

    return output


if __name__ == "__main__":
    """
    Fun stuff happens here ^^
    """

    # Working out new consolidated probability for "Capturing Radiance"
    counter_5050, counter_7525, counter_100_zero, counter_no_5050 = simulate_5050()
    sum_counters = counter_5050 + counter_7525 + counter_100_zero + counter_no_5050
    c_prob = (counter_5050 + counter_7525 + counter_100_zero) / sum_counters  # Sum_counters = N!
    print(f"Consolidated new '50/50' probability: {1-c_prob:.2f}:{c_prob:.2f}")

    # Simulate a specific 5 star
    results = simulate_specific_5star()
    print("The mean nr of pulls for a specific 5 star: {}".format(results.mean()))
    print("The median nr of pulls for a specific 5 star: {}".format(np.median(results)))

    # Do a gacha simulation, assuming we are starting from 40th pull & plot its probability distribution
    results = simulate(cutoff=40)

    # Work out the probility density function from the results
    x, y = np.unique(results, return_counts=True)
    y = y / np.sum(y)

    # Plot the y vs x using scatter plot in matplotlib
    plt.scatter(x, y)
    plt.show()

    """
    Simulating account pulls to determine if one account is more lucky than another - Graphical output
    """

    # Further work...
    results = results.reshape((int(N / 10000), 10000))

    # Calculate a cumulative mean using numpy
    cum_mean = np.cumsum(results, axis=1) / (np.arange(len(results[0, :])) + 1).T
    range = np.arange(500).tolist()
    upper_95 = np.percentile(cum_mean, 95, axis=0).tolist()
    lower_95 = np.percentile(cum_mean, 5, axis=0).tolist()
    fig = go.Figure([
        go.Scatter(
            name='Random Account 1 - Total Pulls: {}'.format(np.sum(results[:500, 0])),
            x=range,
            y=cum_mean[0, :500].tolist(),
            mode='lines',
            line=dict(color='rgb(31, 119, 180)'),
        ),go.Scatter(
            name='Random Account 2 - Total Pulls: {}'.format(np.sum(results[:500, 1])),
            x=range,
            y=cum_mean[1, :500].tolist(),
            mode='lines',
            line=dict(color='#dc2346'),
        ),
        go.Scatter(
            name='Upper Bound',
            x=range,
            y=upper_95,
            mode='lines',
            marker=dict(color="#444"),
            line=dict(width=0),
            showlegend=False
        ),
        go.Scatter(
            name='Lower Bound',
            x=range,
            y=lower_95,
            marker=dict(color="#444"),
            line=dict(width=0),
            mode='lines',
            fillcolor='rgba(68, 68, 68, 0.3)',
            fill='tonexty',
            showlegend=False
        )
    ])
    fig.update_layout(
        yaxis_title='Cumulative Rolling Average number of pulls to get a 5 star',
        xaxis_title='Nr of times a 5 star was pulled',
        title='Cumulative Rolling Average number of pulls to get a 5 star with an average of {} pulls per 500 five stars'.format(np.sum(results[:500, :], axis=0).mean()),
        hovermode="x"
    )
    fig.show()



