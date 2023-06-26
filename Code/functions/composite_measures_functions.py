import numpy as np
import pandas as pd
from scipy import special
from scipy.stats import norm

def calculate_LDI_REC(df_MST_concat, num_forms=1, n_trials=320):
    """
    Given a MST dataset with the following columns, compute LDI and REC measures. It is not adapted for reliability
    split use.

    Parameters
    ----------
    df_MST_concat: df, data, needs to have 'userID', 'trial_type', 'repetition', 'correct', 'response' columns
    num_forms: int, default 1, how many forms=multiples of the data we have
    n_trials: int, default 320, how many trials are there per 1 form in this experiment

    Returns
    -------
    LDI: series of userID and LDI index
    REC: series of userID and REC index

    """

    # clean the dataframe to have only full sets
    df_MST_concat = df_MST_concat[
        df_MST_concat['userID'].isin(df_MST_concat['userID'].value_counts()[df_MST_concat['userID'].value_counts()==num_forms*n_trials].keys())]

    # set userID to be a category to easier deal with NANs and zeros later -- no sure if needed, can be done as in
    # the code above but this eliminates the use to create a df
    df_MST_concat.userID = df_MST_concat.userID.astype('category')
    # https://stackoverflow.com/questions/46752347/why-does-value-count-method-in-pandas-returns-a-zero-count
    # https://pandas.pydata.org/pandas-docs/stable/user_guide/categorical.html

    # define a dict of the subtasks and the num trials per each of the subtasks
    MST_categories_num_trials = {
        "targets": num_forms * 64,
        "foils":   num_forms * 64,
        "lures":   num_forms * 64,
    }

    # num of subjects
    N = len(df_MST_concat.userID.unique())

    # pre-define the data -- note the naming, must correspond to the keys in the dict above
    df_targets = df_MST_concat[(df_MST_concat["trial_type"] == "repeat") & (df_MST_concat["repetition"] == "b")].copy()
    df_foils   = df_MST_concat[(df_MST_concat["trial_type"] == "foil")].copy()
    df_lures   = df_MST_concat[(df_MST_concat["trial_type"] == "lure") & (df_MST_concat["repetition"] == "b")].copy()

    # checks for num of trials per user
    assert (df_targets["userID"].value_counts() == MST_categories_num_trials["targets"]).all()
    assert (df_foils["userID"].value_counts() == MST_categories_num_trials["foils"]).all()
    assert (df_lures["userID"].value_counts() == MST_categories_num_trials["lures"]).all()

    # LDI
    LDI = (df_lures[df_lures["response"] == "similar"].userID.value_counts() / MST_categories_num_trials["lures"]) - \
          (df_foils[df_foils.response == 'similar'].userID.value_counts() / MST_categories_num_trials["foils"])

    # REC
    REC = (df_targets[df_targets["response"] == "old"].userID.value_counts() / MST_categories_num_trials["targets"]) - \
          (df_foils[df_foils.response == 'old'].userID.value_counts() / MST_categories_num_trials["foils"])

    # check that n(userID) == N to verify that we have all subjects and no NANs/missing
    assert N == LDI.size
    assert N == REC.size

    return LDI, REC


def calculate_SCAP_Cowan_k(df_SCAP_concat, num_forms=1, SCAP_n_trials=[25, 23, 24]):
    """
    Given raw SCAP data, return max Cowan's k

    Parameters
    ----------
    df_SCAP_concat: df, data, needs to have 'userID', 'set_size', 'correct' columns
    num_forms: int, default 1, how many forms=multiples of the data we have
    SCAP_n_trials: array, list, default [25,23,24], how many trials there are per levels 3,5,7

    Returns
    -------
    DataFrame of max Cowan's k measure per participant

    """
    # check that it's correct
    assert len(SCAP_n_trials) == 3

    # create arrays for Cowan's k
    cowan_arr = np.zeros((3, df_SCAP_concat['userID'].unique().size))

    # iterate over the levels with their n_trials
    for (j, lvl), n_trials in zip(enumerate([3, 5, 7]), num_forms * np.asarray(SCAP_n_trials)):
        # get load 1 only
        df_scap_shuffle = df_SCAP_concat.loc[df_SCAP_concat["set_size"] == lvl, ["userID", "correct"]].sort_values(
            by=["userID"]).copy()

        # get the Cowan measure
        cowan_arr[j, :] = lvl * (df_scap_shuffle.groupby("userID").correct.mean() * 2 - 1).values

    # take only the max Cowan's k per category
    df = pd.DataFrame({
        "userID": df_SCAP_concat['userID'].sort_values().unique(),
        "SCAP_Cowan_k": cowan_arr.max(axis=0),
    })

    return df


def calculate_all_PGNG_measures(df_PGNG, num_forms=1, blocks=[2, 3, 4], PCTT_n_trials=[33, 26, 32], PCIT_n_trials=[12, 13]):
    """
    Given raw PGNG data, return in total of 9 columns -- three levels for PCTT measure, two levels for PCIT, one overall
    accuracy, one overall PCIT, one overall PCTT (these two are across levels) and userID. It computes all the measures
    based on the definitions in literature. Overall score is accuracy, for Percent Correct Target Trials it takes the
    number of target trials that are not inhibitory and divide the score on those trials by it. For Percent Correct
    Inhibitory Trials, it does the same on inhibitory trials.

    Parameters
    ----------
    df_PGNG: df, data, needs to have 'userID', 'inhibitory', 'block', 'correct' columns
    num_forms: int, default 1, how many forms=multiples of the data we have
    blocks: list, default [2, 3, 4], labels of blocks to include
    PCTT_n_trials: array, list, default [33,26,32], how many non-inhibitory trials there are per block
    PCIT_n_trials: array, list, default [12,13], how many inhibitory trials there are per block

    Returns
    -------
    DataFrame of 9 columns of three levels for PCTT measure, two for PCIT, overall accuracy, PCIT and PCTT, and userID

    """
    # define order of variables that go out
    col_order = ["userID", "PGNG_PCTT", "PGNG_PCIT", "PGNG_L1_PCTT", "PGNG_L2_PCTT", "PGNG_L3_PCTT", "PGNG_L1_PCIT",
                 "PGNG_L2_PCIT", "PGNG_overall_acc", "PGNG_overall_score"]

    # sort all values
    df_PGNG.sort_values(by=["userID"], inplace=True)
    # take only block that we care about
    df_PGNG = df_PGNG[df_PGNG["block"].isin(blocks)]
    # # define the num of trials based on the block, only responses to r,s,t
    # num_targets = df_PGNG.groupby(["userID"]).block.value_counts().unstack().reset_index()

    # filter all the ones which are not inhibitory, then group by block and participant, count the number of correct responses and also have the num of trials in this case
    # if they don't have any correct response, give them 0 (line 116)

    # PCTT - num trials which are correct but not inhibitory
    corr_targets = df_PGNG[~df_PGNG["inhibitory"].astype(bool)].groupby(["userID", "block"]).correct.sum().unstack().reset_index()
    # num of targets in non-inhibitory condition, we will divide by this (and not num_target)
    num_non_inhib = df_PGNG[~df_PGNG["inhibitory"].astype(bool)].groupby(["userID"]).block.value_counts().unstack().reset_index()
    # compute across levels
    corr_targets_overall = df_PGNG[~df_PGNG["inhibitory"].astype(bool)].groupby(["userID"]).correct.sum()
    num_non_inhib_overall = df_PGNG.loc[~df_PGNG["inhibitory"].astype(bool), "userID"].value_counts().reset_index()

    # check that we have the right number of trials
    assert (num_non_inhib[[2, 3, 4]] == num_forms * np.asarray(PCTT_n_trials)).all().all()
    assert (num_non_inhib_overall["userID"] == num_forms * np.asarray(PCTT_n_trials).sum()).all()

    # PCIT - it's the same strategy as for the one above, just that is for all the non-inhibitory and this is for all
    # num trials which are correct and inhibitory
    corr_inhibitory = df_PGNG[(df_PGNG["inhibitory"].astype(bool)) & (df_PGNG["block"].isin([2, 4]))].groupby(["userID", "block"]).correct.sum().unstack().reset_index()
    # the inhibitory, also select the inhibitory blocks only
    num_inhibitory = df_PGNG[(df_PGNG["inhibitory"].astype(bool)) & (df_PGNG["block"].isin([2, 4]))].groupby(["userID"]).block.value_counts().unstack().reset_index()
    # compute across levels
    corr_inhibitory_overall = df_PGNG[(df_PGNG["inhibitory"].astype(bool)) & (df_PGNG["block"].isin([2, 4]))].groupby(["userID"]).correct.sum()
    num_inhibitory_overall = df_PGNG.loc[(df_PGNG["inhibitory"].astype(bool)) & (df_PGNG["block"].isin([2, 4])), "userID"].value_counts().reset_index()

    # check that we have the right number of trials
    assert (num_inhibitory[[2, 4]] == num_forms * np.asarray(PCIT_n_trials)).all().all()
    assert (num_inhibitory_overall["userID"] == num_forms * np.asarray(PCIT_n_trials).sum()).all()

    # divide them, https://stackoverflow.com/questions/49412694/divide-two-pandas-dataframes-and-keep-non-numeric-columns
    df_scores = corr_targets.select_dtypes(exclude='object').div(
        num_non_inhib.select_dtypes(exclude='object')).combine_first(num_non_inhib)
    df_scores_inhib = corr_inhibitory.select_dtypes(exclude='object').div(
        num_inhibitory.select_dtypes(exclude='object')).combine_first(num_inhibitory)

    # renaming
    df_scores_inhib.columns = ["PGNG_L1_PCIT", "PGNG_L2_PCIT", "userID", ]
    if len(blocks) == 3:
        df_scores.columns = ["PGNG_L2_PCTT", "PGNG_L1_PCTT", "PGNG_L3_PCTT", "userID"]

    elif len(blocks) == 4:
        df_scores.columns = ["PGNG_L0_PCTT", "PGNG_L2_PCTT", "PGNG_L1_PCTT", "PGNG_L3_PCTT", "userID"]
        col_order.insert(3, "PGNG_L0_PCTT")
    else:
        raise ValueError(f"Levels are only defined for 3 or 4 blocks, there is {len(blocks)}.")

    # add the measures across levels
    df_scores = df_scores.merge(
        (corr_targets_overall / (num_forms*np.asarray(PCTT_n_trials).sum())).reset_index().rename(columns={"correct": "PGNG_PCTT"}),
        on="userID")
    df_scores_inhib = df_scores_inhib.merge(
        (corr_inhibitory_overall / (num_forms*np.asarray(PCIT_n_trials).sum())).reset_index().rename(columns={"correct": "PGNG_PCIT"}),
        on="userID")

    # overall correct score -- mean number of correct trials, irrespective of block/condition/whether it's inhibitory or not
    df_scores["PGNG_overall_acc"] = df_scores["userID"].map(df_PGNG.groupby(["userID"]).correct.mean())
    # and total sum (score)
    df_scores["PGNG_overall_score"] = df_scores["userID"].map(df_PGNG.groupby(["userID"]).correct.sum()).astype(int)

    # merge and make nice
    df_merge_results = df_scores.merge(df_scores_inhib, on="userID")
    return df_merge_results[col_order]


def calculate_d_prime(df, signal, noise, n_signal, n_noise, col='correct_response', beta=False, c=False, Ad=False):
    """
    This function calculates d prime and uses the third correction for extreme values from this list:
    https://stats.stackexchange.com/questions/134779/d-prime-with-100-hit-rate-probability-and-0-false-alarm-probability
    To quote:
    add 0.5 to both the number of hits and the number of false alarms, and add 1 to both the number of signal trials and
    the number of noise trials; dubbed the loglinear approach (Hautus, 1995)
    Note: the loglinear method calls for adding 0.5 to all cells under the assumption that there are an equal number of
    signal and noise trials. If this is not the case, then the numbers will be different. If there are, say, 60% signal
    trials and 40% noise trials, then you would add 0.6 to the number of Hits, and 2x0.6 = 1.2 to the number of signal
    trials, and then 0.4 to the number of false alarms, and 2x0.4 = 0.8 to the number of noise trials, etc.

    The implementation of other measures follows https://lindeloev.net/calculating-d-in-python-and-php/

    The function requires (beside numpy and pandas):
    from scipy import special
    from scipy.stats import norm

    Parameters
    ----------
    df: dataframe, assumes columns with response_type labels ['hit', 'CR', 'FA', 'miss'] and a column with correct
        response per userID
    signal: float, adjusted ratio of n_signal/(2*n_trials_per_form_all) where n_signal is number of signal trials
    noise: float, adjusted ratio of n_noise/(2*n_trials_per_form_all) where n_noise is number of noise trials,
                 it must hold that: n_signal+n_noise == 2*n_trials_per_form_all
    n_signal: int, number of signal trials
    n_noise: int, number of noise trials

    Returns
    -------
    Series of d prime per userID calculated as df_d['hit'] - df_d['FA'] where these are ppfs of the adjusted rates

    """

    # select only relevant trials -- this is slow but I don't know how to make it faster
    df = df[df["response_type"].isin(['hit', 'FA'])]

    # get counts of hits and FAs (and all response types)
    # https://datascience.stackexchange.com/questions/94436/pandas-groupby-count-doesnt-count-zero-occurrences
    #     df_d = pd.crosstab(df["response_type"], df["userID"]).T
    # this is faster: https://stackoverflow.com/questions/37003100/pandas-groupby-for-zero-values/
    df_d = df.groupby(['response_type', 'userID'])[col].count().unstack(1, fill_value=0).T

    # make sure it works even if there is no hit or FA
    for col in ['hit', 'FA']:
        if col not in df_d.keys():
            df_d[col] = 0

    # the norm.ppf is slowing down the calculations, https://stackoverflow.com/questions/48552133/why-is-the-scipy-stats-norm-ppf-implementation-so-slow
    # df_d['FA'] = ((df_d['FA']+noise)/(n_noise+2*noise)).apply(lambda x: scipy.special.ndtri(x))
    # df_d['hit'] =((df_d['hit']+signal)/(n_signal+2*noise)).apply(lambda x: scipy.special.ndtri(x))

    # and list of comprehensions is also faster
    df_d['FA'] = [special.ndtri(x) for x in (df_d['FA'] + noise) / (n_noise + 2 * noise)]
    df_d['hit'] = [special.ndtri(x) for x in (df_d['hit'] + signal) / (n_signal + 2 * signal)]

    # check if we asked for any of the measures, if yes, return dictionary, otherwise only return d prime to keep it fast
    if beta or c or Ad:
        # define dict
        out = {'d_prime': df_d['hit'] - df_d['FA']}
        # check one by one if we want those measures
        if beta:
            out['beta'] = np.exp((df_d['FA']**2 - df_d['hit']**2)/2) #math.exp((Z(fa_rate) ** 2 - Z(hit_rate) ** 2) / 2),
        if c:
            out['c'] = -(df_d['hit'] + df_d['FA']) / 2  # -(Z(hit_rate) + Z(fa_rate)) / 2,
        if Ad:
            out['Ad'] = (out['d_prime']/np.sqrt(2)).apply(lambda x: norm.cdf(x)) #norm.cdf(out['d'] / math.sqrt(2)),

        return out

    return df_d['hit'] - df_d['FA']
