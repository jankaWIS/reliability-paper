import os
import time
import warnings

import numpy as np
import pandas as pd
from scipy import special

### Functions for fitting

def linear(x, a, b):
    return a * x + b


def pdf_beta(x, a, b):
    return special.gamma(a+b) * (x**(a-1)) * ((1-x)**(b-1)) / (special.gamma(a)*special.gamma(b))


def hyperbolic_fit(x, a):
    return x / (x + a)


def beta_fit_using_moments(data):
    """
    Given a distribution of values (data), return alpha and beta coefficients that would correspond to this distribution
    if it was beta. This fit is done using moments of the data -- mean and variance corrected for finite samples (ddof=1)

    References:
    https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.beta.html
    https://stackoverflow.com/questions/23329331/how-to-properly-fit-a-beta-distribution-in-python

    Parameters
    ----------
    data: array or list, distribution that we want to check

    Returns
    -------
    alpha, beta: floats, coefficients of the corresponding beta distribution
    mean, var: floats, mean and sample variance

    """
    mean = np.mean(data)
    var = np.var(data, ddof=1)
    alpha = mean ** 2 * (1 - mean) / var - mean
    beta = alpha * (1 - mean) / mean

    return alpha, beta, mean, var


def analytical_C(mu, var):
    """

    Parameters
    ----------
    mu: float, mean of a given distribution of probabilities (accuracies)
    var:  float, (sample) variance of a given distribution of probabilities (accuracies)

    Returns
    -------
    coefficient C

    """
    return (mu-mu**2)/var - 1


def analytical_C_unbiased(mu, var, L2):
    """
    Formula corrected for limited sample size using the law of total variance

    Parameters
    ----------
    mu: float, mean of a given distribution of probabilities (accuracies)
    var:  float, (sample) variance of a given distribution of probabilities (accuracies)
    L2: int, number of trials that was used, NOTE -- the formula in appendix has 2L because L is half of the trials

    Returns
    -------
    Unbiased estimation of coefficient C

    """
    return (mu-mu**2-var)/(var - (mu-mu**2)/L2)


def bin_samples_rand4(rng, N, total_n_trials, mu_throws, ravel=True):
    """
    Given N subjects each having a probability p of success stored in mu_throws, simulate total_n_trials of flip coins

    check the testing part in simulate_task_distributions4reliability.ipynb and my question on slack for details
    https://stackoverflow.com/questions/72487233/binomial-distributions-bernoulli-trials-with-different-probabilities

    Parameters
    ----------
    rng: np random state, for reproducibility
    N: int, number of subjects
    total_n_trials: int, length of sequence of 1 and 0 that will be generated for each subject
    mu_throws: array of floats <0,1> of len N, array of probabilities of success (p) for each subject
    ravel: bool, default True, what shape to return, whether to return an array of shape (1, N*total_n_trials)

    Returns
    -------
    Coin flips (1,0) array of shape (N, total_n_trials) or (1, N*total_n_trials)
    """
    if ravel:
        return (rng.random(size=(N, total_n_trials)) <= mu_throws[:, None]).astype(np.uint8).ravel()

    return (rng.random(size=(N, total_n_trials)) <= mu_throws[:, None]).astype(np.uint8)


### Functions for reliability and analysis/preprocessing

def create_random_reliability_curve(total_n_trials=150, N=20, n_repeats=10 ** 3, step=None,
                                    rng=np.random.default_rng(0), rng_split=np.random.default_rng(0),
                                    print_time=True, verbose=True):
    """

    Parameters
    ----------
    total_n_trials: int, default 150, how many trials to create for each subject
    N: int, default 20, number of subjects in the simulation
    n_repeats: int, default 1000, define repeats for reliability sampling -- how many times to calculate the reliability,
                                  how many times to sample/shuffle
    step: default None and depends on total_n_trials -- either 2 if there is less than 100 trials, or 5, how refined is the sampling
    rng: numpy random generator, default is with seed 0, random generator for defining data
    rng_split: numpy random generator, default is with seed 0, random generator for defining sampling (splits)
    print_time: bool, default True, whether to count and print how long it took
    verbose: bool, default True, whether to print n trials list

    Returns
    -------
    array_corr_trials_psychofit: numpy array of reliability per repeat per L (n trials in the sequence from step to total n trials/2)
    n_trials_list: list of Ls (sample sizes) that the reliability was calculated for

    """

    # define step
    if step is None:
        # have smaller steps for less trials
        if total_n_trials <= 100:
            step = 2
        else:
            step = 5

    n_trials_list = np.arange(step, (total_n_trials + step) // 2, step)
    if verbose:
        print(f"Going over {*n_trials_list,} trials per task, ie. {len(n_trials_list)} items")

    # define corr array
    array_corr_trials_psychofit = np.zeros((len(n_trials_list), n_repeats))

    if print_time:
        # start the timer
        start = time.time()

    # generate trials, array of ones and zeros with mean of 0.5
    all_trials_reshaped = (rng.random(size=(N, total_n_trials)) <= 0.5).astype(np.uint8).reshape(N, -1)

    for j, n_trials in enumerate(n_trials_list):

        # check that it's possible
        assert n_trials <= total_n_trials // 2

        # go over iterations
        for i in range(n_repeats):
            # instead of splitting it, I will sample = shuffle the array and then take first and second half
            arr = None
            arr = rng_split.choice(all_trials_reshaped, size=(n_trials * 2), replace=False, axis=1)

            # calculate correlation
            array_corr_trials_psychofit[j, i] = \
            np.corrcoef(arr[:, :n_trials].mean(axis=1), arr[:, n_trials:].mean(axis=1))[0, 1]

    if print_time:
        print(f"Process took: {time.time() - start:.2f} s which is {(time.time() - start) / 60:.2f} min.")

    return array_corr_trials_psychofit, n_trials_list


def check_double_access(df):
    """
    df: DataFrame with "userID" and "observation" fields

    Checks if there is the same number of user IDs as observations
    """
    if df['observation'].unique().size != df['userID'].dropna().unique().size:
        print(f"There are {df['observation'].unique().size} participants (observations) and {df['userID'].dropna().unique().size} unique IDs! Check!\n")
        warnings.warn(f"There are {df['observation'].unique().size} participants (observations) and {df['userID'].dropna().unique().size} unique IDs! Check!\n")
    else:
        print("All is ok.")
        print(f"There are now {df['observation'].unique().size} participants (observations) and {df['userID'].dropna().unique().size} unique IDs.\n")


def split_dataframes_faster(df, total_n_trials, n_trials, N, n_levels=1):
    """
    Takes data and splits them in two halves

    Parameters
    ----------
    df: DataFrame, the real data which we want to split,
    N: int, number of subjects
    total_n_trials: int, how many trials is the maximum,
    n_trials: int, how many trials are we taking,
    n_levels: int, how many levels (perception/memory) there are, this is used for pseudoslopes where we want
                    to take the same trials from each of the levels, so we want to tile (repeat) the same indices
                    we had in one category (level) to all the others
    """
    # select n_trials random indexes
    random_inx = np.random.choice(range(total_n_trials), n_trials, replace=False)

    # create boolean indexing
    half_A = np.ones(total_n_trials)
    half_A[random_inx] = 0

    # select the same trials from all three difficulties
    half_A = np.tile(half_A, n_levels)

    # create boolean indexing for the entire array
    A_indexes = np.tile(half_A, N).astype(bool)

    # split the data into two halves, sorted by userID and task so that we get the exact same trials for every subject
    df_first = df[A_indexes].copy()
    df_second = df[(1 - A_indexes).astype(bool)].copy()

    return df_first, df_second


def split_dataframes_faster_chunks(df, total_n_trials, n_trials, N, n_levels=1, random_indexing=True):
    """
    Takes data and splits them in two halves.

    To have it adapted for splits to chunks and not only halves, there is a new addition. We first choose indices
    at random as before. Then we take the same number of indices from the remaining, not yet selected, indices. That
    assures that there will be the same number of ones, ie. the same number of elements in both halves. Note that
    this yields equivalent result for splitting halves.

    Parameters
    ----------
    df: DataFrame, the real data which we want to split,
    N: int, number of subjects
    total_n_trials: int, how many trials is the maximum,
    n_trials: int, how many trials are we taking,
    n_levels: int, how many levels (perception/memory) there are, this is used for pseudoslopes where we want
                    to take the same trials from each of the levels, so we want to tile (repeat) the same indices
                    we had in one category (level) to all the others
    random_indexing: bool, default True, if to sample and split the df/array randomly or if to take the first 2*n_trials
                    elements and split them in half (for checking learning)
    """

    if random_indexing:
        # select n_trials random indexes
        random_inx = np.random.choice(range(total_n_trials), n_trials, replace=False)
        # if we run samples and not halves, we need to select again a sample which is of the same size and not sampled
        # yet, for that take a random selection of not yet chosen indices (note that for halves this is the same as
        # getting a complement)
        random_inx_B = np.random.choice(list(set(range(total_n_trials)) - set(random_inx)), n_trials, replace=False)

    else:
        # create indices from 0 to n_trials
        random_inx = np.arange(n_trials)
        # and from n_trials to twice as much
        random_inx_B = np.arange(n_trials, 2*n_trials)

    # create boolean indexing, select those indices
    half_A = np.zeros(total_n_trials)
    half_A[random_inx] = 1

    half_B = np.zeros(total_n_trials)
    half_B[random_inx_B] = 1

    # select the same trials from all three difficulties
    half_A = np.tile(half_A, n_levels)
    half_B = np.tile(half_B, n_levels)

    # create boolean indexing for the entire array
    A_indexes = np.tile(half_A, N).astype(bool)
    B_indexes = np.tile(half_B, N).astype(bool)

    # split the data into two halves, sorted by userID and task so that we get the exact same trials for every subject
    df_first = df[A_indexes].copy()
    df_second = df[B_indexes].copy()

    return df_first, df_second


def check_df_get_numbers(df, N, col='userID'):
    """
    Returns the maximum number of trials.
    Checks if those number fit. Checks if the num of subjects N is the same as in the df

    Parameters
    ----------
    df: dataFrame, data where to check
    N: int, expected number of subject
    col: sting, default 'userID', which column to check
    """

    # check that all people have the same number of trials
    assert df[col].value_counts().unique().size == 1
    # get the num of trials
    total_n_trials = df[col].value_counts().unique()[0]
    print(f"Total number of trials is {total_n_trials}.")

    # check for discrepancy
    assert df[col].unique().size == N

    return total_n_trials


def run_reliability_estimate4many_trials(total_n_trials, df, N, test,
                                         save=True, step=None,
                                         sort_cols=["userID"], cols=["userID", "correct"], measure='correct',
                                         n_repeats=10 ** 3, num_forms=None,
                                         take_trials=None, exp_name='longitudinal',
                                         path_curve_fit="/Users/jan/Documents/GitHub/UCLA/UCLA_Weizmann_Project/Analysis/Reliability/curve_fits",
                                         verbose=True):
    """
    Computes reliability for split halves in a given dataframe (df) for several number of trials which are generated
    as a sequence from (step, total_n_trials) with a step of step.

    Parameters
    ----------
    total_n_trials: int, number of trials to take
    df: dataframe, data which to use, must contain userID and correct columns
    N: int, number of participants
    test: str, name of the test which is being run
    save: bool, default True, if to save the data
    step: int, default None, step to take when creating a sequence of num_trials for which this function will run,
                by default if None, it will take either step=2 if the total_n_trials is <=100 or step=5
    sort_cols: str, list , default 'userID', what to use for sorting the data
    cols: list, default ['userID', 'correct'], list of columns to be used to sort the dataframe by
    measure: str, default 'correct', measure to be used to calculate correlations between
    n_repeats: int, default 1000, how many times to compute correlation
    num_forms: int, how many forms were used to get the data
    take_trials: int, default None, limit the max num of trials which we sample
    exp_name: string, default 'longitudinal', name of the subexperiment under which it will be saved
    path_curve_fit: str, default my folder in reliability, path where to store the data
    verbose: bool, default True, whether to print out comments

    """
    # define name label, if we take first n trials, this label will capture it
    label = ''

    # define step
    if step is None:
        # have smaller steps for less trials
        if total_n_trials <= 100:
            step = 2
        else:
            step = 5

    n_trials_list = np.arange(step, (total_n_trials + step) // 2, step)
    # limit the number of trials if set
    if take_trials is not None:
        n_trials_list = n_trials_list[n_trials_list < take_trials + 1]
        # update file name label
        label = f'_trials{take_trials}'

    print(f"Going over {*n_trials_list,} trials per task, ie. {len(n_trials_list)} items")

    # define corr array
    array_corr_trials_psychofit = np.zeros((len(n_trials_list), n_repeats))
    trial_time = np.zeros(len(n_trials_list))

    # define the df
    df_sort = df.sort_values(by=sort_cols)[cols]

    # start the timer
    start = time.time()

    for j, n_trials in enumerate(n_trials_list):

        start_trial = time.time()

        # check that it's possible
        assert n_trials <= total_n_trials // 2

        # go over iterations
        for i in range(n_repeats):
            df_first = None
            df_second = None

            # split the data into two halves, sort it by userID and task so that we get the exact same trials for
            # every subject
            df_first, df_second = split_dataframes_faster_chunks(df_sort, total_n_trials, n_trials, N)

            # calculate correlation
            array_corr_trials_psychofit[j, i] = df_first.groupby(sort_cols)[measure].mean().corr(
                df_second.groupby(sort_cols)[measure].mean(),
                method='pearson')

        trial_time[j] = time.time() - start_trial
        if verbose:
            print(f"Time per {n_trials} trials: {trial_time[j]:.2f} s which is {trial_time[j] / 60:.2f} min.")

    end = time.time()
    print(f"Process took: {end - start:.2f} s which is {(end - start) / 60:.2f} min.")

    if save:
        # save
        pd.DataFrame(array_corr_trials_psychofit.T, columns=[f"n_trials_{x}" for x in n_trials_list]
                     ).to_csv(os.path.join(path_curve_fit,
                                           f"{test}_{exp_name}_reliability_{num_forms}{label}_chunks_psychofit_step{step}.csv"),
                              index=False)

        pd.DataFrame({"reliability": array_corr_trials_psychofit.mean(axis=1),
                      "n_trials": n_trials_list}).to_csv(os.path.join(path_curve_fit,
                                                                      f"{test}_{exp_name}_reliability_{num_forms}{label}_chunks_psychofit_step{step}-fit.csv"),
                                                         index=False)

    return array_corr_trials_psychofit, trial_time, n_trials_list


def get_statistics4many_trials(df, n_trials_list, sort_cols=["userID"], cols=["userID", "correct"], random_state=0,
                               measure="correct", save_merge=False, path="./", file=""):
    """

    Parameters
    ----------
    df: dataframe, data
    n_trials_list: list or np array, number of trials to take in each iteration
    sort_cols: str, list , default 'userID', what to use for sorting the data
    cols: list, default ['userID', 'correct'], list of columns to be used to sort the dataframe by
    measure: str, default 'correct', measure to be used to calculate the values for
    random_state: int or none, default 0, random state for reproducibility
    save_merge: bool, whether to update the csv file with reliability
    path: str, if save_merge, then from where to load and to where to save the reliability data
    file: str, name of the file with reliability

    Returns
    -------
    df with measures (SEM, mean, SD, var, sum of subjects' SD) per n_trials
    var_p and mean_p returns the (sample) variance and mean respectively of the distribution of accuracies (p) in that task
    """

    # define the dataframe
    df_measures = pd.DataFrame()
    # select only relevant cols
    df_sort = df.sort_values(by=sort_cols)[cols]

    for j, n_trials in enumerate(n_trials_list):
        # randomly sample number of trials per subject
        df_tmp = df_sort.groupby(sort_cols).sample(n_trials, random_state=random_state)
        # create statistics
        df_measures = df_measures.append(pd.DataFrame({
            "n_trials": [n_trials],
            "mean": df_tmp[measure].mean(),
            "sd": df_tmp[measure].std(),
            "var": df_tmp[measure].var(),
            "sum_subj_sd": df_tmp.groupby(sort_cols)[measure].std().sum(),
            "sem": df_tmp[measure].std() / np.sqrt(n_trials),
            "mean_p": df_tmp.groupby(sort_cols).mean().mean(),
            "var_p": df_tmp.groupby(sort_cols).mean().var(),
        }))

    # if wanted, merge this df with the corresponding reliability and save to csv
    if save_merge:
        df_measures.merge(pd.read_csv(os.path.join(path, file)), on="n_trials").to_csv(os.path.join(path, file),
                                                                                       index=False)

    return df_measures


def extract_data(df, total_n_trials, measure='correct', user_col='userID'):
    """
    Extracts and reshapes data from a DataFrame to a numpy array based on unique user IDs and a specified measure.
    This is used to speed up reliability calculations. It also executes a check that the values per subject are indeed
    the same as in the df. To do that, it sorts the values by the "user_col" value, meaning it can possibly
    reorder the trials.

    The function first sorts the DataFrame based on the specified user column, then extracts the specified measure
    for each user, reshapes the data into a 2D array, and ensures the integrity of the extracted data.

    Parameters:
    -----------
    df: pd dataframe, DataFrame containing participants' data, expects at least 2 columns - trial outcomes per subject
    total_n_trials : int, total number of trials per participant
    measure : str, optional (default='correct'), the measure to extract from the DataFrame (trial outcomes). Default is 'correct'.
    user_col : str, optional (default='userID'), the column in the DataFrame representing user IDs. Default is 'userID'.

    Returns:
    --------
    2D numpy.ndarray containing reshaped data for each unique user. The shape of the array is (number of unique users, total_n_trials).

    Raises:
    ------
    AssertionError if the extracted data and reshaped data are not equal for any user. Takes nans into account.

    """

    # sort values
    df.sort_values(by=user_col, ignore_index=True, inplace=True)

    # extract data
    all_trials_reshaped = df.loc[:, measure].values.reshape(df[user_col].unique().size, total_n_trials)

    # check that indeed they are the same
    for i, user in enumerate(df[user_col].unique()):
        if np.isnan(all_trials_reshaped[i]).any():
            # deal with nans
            assert np.allclose(df.loc[df[user_col] == user, measure].values, all_trials_reshaped[i], equal_nan=True)
        else:
            assert (df.loc[df[user_col] == user, measure].values == all_trials_reshaped[i]).all()

    return all_trials_reshaped


def count_consecutive(s):
    """
    Some inspiration for how to apply it on the df:
    https://stackoverflow.com/questions/29640588/pandas-calculate-length-of-consecutive-equal-values-from-a-grouped-dataframe

    The algorithm is then adapted from:
    https://stackoverflow.com/questions/24342047/count-consecutive-occurences-of-values-varying-in-length-in-a-numpy-array

    Principle
    ----------
    Convert the dataframe/series into numpy
    Generate list of bool arrays for each of the correct responses. Then for an array c in this list, do the fancy
    count of consecutive values and take max of such an array. Finally, take and return max of those three values.


    Parameters
    ----------
    s: pandas series, data that we want to check
    """
    # convert into numpy
    condition = s.values

    return max([max(
        np.diff(np.where(np.concatenate(([c[0]],
                                         c[:-1] != c[1:],
                                         [True]))
                         )[0])[::2]
    ) for c in [condition == x for x in np.unique(condition)]])


def flag_participants(df, measure='correct', n_score_std=0.5, n_RT_std=2, n_seq_std=2, n_RT_robot_std=2,
                      RT_thr=10000, correct_response='correct_response', plot_distribution=False):
    """
    Follow the following algorithm to flag potentially problematic participants:
    1. No exclusion happens if participant's score is greater than $\mu-\sigma$ even if this participants is flagged several times.
    2. Flags are given if:
        1. RT outlier -> the person is 2 (or 1.5) SD faster than the group.
        2. Too narrow RT -> it is likely a script/robot. Cut too long RT (maybe task specific?), then flag people whose SD of RT is below 2 SD of the task/group.
        3. Sequence length -> if they press too many times the same key (either do SD from group or SD from max length of such a sequence in the experiment or over some number that can rarely happen at chance --> done as 2 SD from mean of max length across participants).
        4. (not implemented here) Low accuracy on sanity checks (3 SD bellow the average) -- only for tasks where we have it.


    Parameters
    ----------
    df: pandas dataframe
    measure: str, default 'correct', what measure we want to use for preselection - accuracy
    n_score_std: float, default 0.5, number of standard deviations to use to suspect subjects if their score is below
            mean minus this std
    n_RT_std: float, default 2, number of std that a suspected subject must be faster to be flagged
    n_seq_std: float, default 2, number of std that a suspected subject must be above the averaged maximal
            consequtive asnwer length (button press) to be flagged
    n_RT_robot_std: float, default 2, number of std that a suspected subject's SD of their RT must be smaller to be flagged
    RT_thr: int/float, default 10000 (ms), what to cut off when checking for robot outliers
    correct_response: str, default 'correct_response', what column to use to count the longest naturally occuring sequence of answers
    plot_distribution: bool, default false, if to plot a distribution of scores if there are no suspects in the task

    Returns
    ----------
    flag_all: list, of all flagged subjects
    suspects: list, of suspects
    """

    # define output
    flag_all = []

    # convert types and check that we have it
    for m in [measure, "RT"]:
        if m in df.columns:
            df[m] = df[m].astype(float)
        else:
            print(f'Measure {m} is not in the dataframe.')
            raise KeyError(f'Measure {m} is not in the dataframe.')

    mu_score = df[measure].mean()
    std_score = df[measure].std()
    # alt. take mean of participant's std
    # std_score = df.groupby(['userID'])[measure].std()

    suspects = df.groupby(['userID'])[measure].mean()[
        df.groupby(['userID'])[measure].mean() < (mu_score - n_score_std * std_score)].index.tolist()

    ### check for robots

    df_RT_bot = df[df["RT"] < RT_thr].groupby(['userID'])["RT"].std()[
        df[df["RT"] < RT_thr].groupby(['userID'])["RT"].std() < \
        (df[df["RT"] < RT_thr].groupby(['userID'])["RT"].std().median() - n_RT_robot_std *
         df[df["RT"] < RT_thr].groupby(['userID'])["RT"].std().std())]
    print(df_RT_bot)
    flag_RT_bot = df_RT_bot.index.tolist()
    if len(flag_RT_bot):
        print(f'\n--> Flagging {len(flag_RT_bot)} subjects for too narrow RT (bots/scripts).\n')

    # concat bots and suspects
    #     suspects = list(set(suspects)|set(flag_RT_bot))
    print(f'In the task, there are {len(suspects)} suspects.')

    if len(suspects) + len(flag_RT_bot) == 0:
        if plot_distribution:
            fig, ax = plt.subplots(1, 1)
            sns.histplot(df.groupby(['userID'])[measure].mean(), ax=ax)
            ax.axvline(mu_score, c='k', label=f'$\mu=${mu_score:.2f}\n$\sigma=${std_score:.2f}')
            plt.legend()
            plt.show()
        print('No exclusions')

    else:
        df_suspects = df.loc[df["userID"].isin(suspects)]

        ### check for RT outliers
        mu_RT = df["RT"].mean()
        std_RT = df["RT"].std()

        flag_RT = df_suspects.groupby(['userID'])["RT"].mean()[
            df_suspects.groupby(['userID'])["RT"].mean() < (mu_RT - n_RT_std * std_RT)].index
        print(f'\n--> Flagging {len(flag_RT)} subjects for RT.\n')

        ### check for sequence length

        # get maximum consecutive responses that DOES happen naturally in the experiment
        # select all the trials for one participants
        # either use iloc[:n_trials_per_form] or like this with the first user - NOTE!!! Must be cleaned of non-finished
        max_consecutive = count_consecutive(df[df['userID'] == df['userID'].unique()[0]][correct_response].dropna())

        try:
            # get the maximum per participant
            tmp = df_suspects.groupby(['userID']).response.apply(count_consecutive)

        # if answers aren't integers but strings, do a quick fix
        except TypeError:
            # create mapping
            resp_dict = {}
            for i, item in enumerate(df['response'].unique()):
                resp_dict[item] = i

            df_suspects["response_conv"] = df_suspects["response"].map(resp_dict)
            tmp = df_suspects.groupby(['userID']).response_conv.apply(count_consecutive)

        print(
            f'Max naturally occurring answers is {max_consecutive} and mean of max per subject is {tmp.mean():.2f} +/- {tmp.std():.2f}.')

        # this is probably wrong but flag participants who have at some point longer sequence than avg+max in the set
        print(tmp[tmp > tmp.mean() + n_seq_std * tmp.std()])
        flag_seq = tmp[tmp > tmp.mean() + n_seq_std * tmp.std()].index

        print(f'\n--> Flagging {len(flag_seq)} subjects for too long consecutive answers.\n')

        # combine the two flags
        flag_all = sorted(list(set(flag_RT) | set(flag_RT_bot) | set(flag_seq)))
        print(f'In total flagging {len(flag_all)} participants:\n{*flag_all,}')

    return flag_all, sorted(suspects)
