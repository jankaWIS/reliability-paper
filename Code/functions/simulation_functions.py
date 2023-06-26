import os, sys
import time
import numpy as np
import pandas as pd
from lmfit import Model

path_results = "../../Data/results"
path_simulation_general = os.path.join(path_results, "curve_fits", "compare_fitting_methods")
path_output = os.path.join(path_simulation_general, "simulations")
functions_modul = "../functions"

# Importing module for functions
sys.path.insert(0, functions_modul)

from reliability_functions import check_df_get_numbers, split_dataframes_faster_chunks, hyperbolic_fit, linear, \
    analytical_C, analytical_C_unbiased, bin_samples_rand4


def run_estimate_C_with_unbiased(params, N, total_n_trials, n_trials_list, n_simulations=10 ** 2, n_repeats=10 ** 2,
                    distribution='gaussian', fit_functions=[linear, hyperbolic_fit], rng=None, rng_mu=None,
                    update_rng_per_simulation=False, rng_C=None, unconstrained_N=10**7, save=True,
                    mu_subjects=None, vary='N', path_output=path_output):
    """
    Run simulations of experiments and return values of C (previously named BA) fits using four different methods
    (added unbiased C)

    Parameters
    ----------
    params: tuple of floats, mean and sigma of the distribution we want to sample
    N: int, number of participants
    total_n_trials: int, length of sequence of 1 and 0 that will be generated for each subject
    n_trials_list: list/array of int, which L (chunks/samples of data) to go over when calculating reliability to create
                    a good data for the fit
    n_simulations: int, default 100, how many simulations we want to run
    n_repeats: int, default 100, repeats for reliability sampling -- how many times to calculate the reliability
    distribution: str, default gaussian, only lognormal or gaussian are implemented, which distribution to sample
    fit_functions: list of functions, default [linear, hyperbolic_fit], functions we want to use for fitting using lmfit
    rng: np random state, default np.random.default_rng(0), for reproducibility in generating the sample distribution
    rng_mu: rng: np random state, default np.random.default_rng(0), for reproducibility in generating distribution of mu
    update_rng_per_simulation: bool, default False, if True, it will generate a new seed (rng) for every iteration of
                    the simulations in creating the data
    rng_C: np random state, default np.random.default_rng(0), for reproducibility, generates the unconstrained true C distribution
    unconstrained_N: int, default 10**7, size (N) of the unconstrained distribution that will be used to determine true C
    save: bool, default True, whether to save the results
    mu_subjects: array of floats <0,1> of len N, default None, array of probabilities of success (p) for each subject,
                    if None, it will be computed from given sigma and variance (params)
    vary: str, default 'N', label, whether we are varying N or n_trials
    path_output: str, where to save the results, default should be the same as stated above

    Returns
    -------
    6 arrays:
    * three are the three fits of size n_simulations (one of them has a shape (2, n_simulations) since it has intercept
      and slope)
    * mean and variance of the generated distribution, each of shape n_simulations
    * reliability array of shape (n_simulations, len(n_trials_list)) providing reliability for each simulation and each
      value of L

    Can save csv with pandas dataframe.
    True C or true unbiased C: computed using either original or corrected formula, calculates the C coefficient on
        large sample (of size unconstrained_N) and reflects the true value of C coefficient that we should find. Note
        that for large L, these two should be identical. The unbiased C does not make much sense in this particular
        design because we generate the P distribution (actual proficiencies) and not its approximation Z, so it's
        disabled at the moment.
    Expected C or expected unbiased C: computed using the distribution definitions. This can be not precise and should
        be used with caution as if used with distributions that need clipping (aren't between 0 and 1, e.g. some
        gaussian), these will give wrong and erroneous results. On the other hand, this can serve as a QC for the
        sampled distribution.

    """
    # define saving
    if vary == "N":
        path_output = os.path.join(path_output, str(N))
    elif vary == "n_trials":
        path_output = os.path.join(path_output, str(total_n_trials))

    # define seeds, it is done this way to avoid different and inconsistent seeds if run in parallel
    if rng is None:
        rng = np.random.default_rng(0)
    if rng_C is None:
        rng_C = np.random.default_rng(0)
    if rng_mu is None:
        rng_mu = np.random.default_rng(0)

    # start the timer
    start = time.time()

    # save logs
    logs = []

    # unpack params
    mu_subjects_mu, mu_subjects_std = params
    logs.append(f'Processing mean/a {mu_subjects_mu} and std/b {mu_subjects_std} for {distribution} distribution.\n')

    # save log name
    log_name = f'log_{distribution}_N{N}_{mu_subjects_mu}_{mu_subjects_std}.txt'

    generate_means = False
    if mu_subjects is None:
        # generate means
        generate_means = True
        # mu_subjects = get_subject_means(mu_subjects_mu, mu_subjects_std, N, distribution, rng)

        # create a sample that would determine the true C, it must be a big one to estimate the true C because because
        # of the clipping, the distribution that we create does not need to be the same as we define
        real_distribution = get_subject_means(mu_subjects_mu, mu_subjects_std, unconstrained_N, distribution, rng_C)

        # compute and save true C based on created distribution (what we sample from but bigger)
        true_C = analytical_C(real_distribution.mean(), real_distribution.var())
        # true_C_unbiased = analytical_C_unbiased(real_distribution.mean(), real_distribution.var(), total_n_trials)

        # note
        logs.append('Generating distribution because none was provided.\n\n')

    # define arrays
    simulated_C_teor = np.zeros(n_simulations)
    simulated_C_teor_unbiased = np.zeros(n_simulations)
    simulated_C_fit_A = np.zeros(n_simulations)
    simulated_C_fit_lin = np.zeros((n_simulations, 2))

    simulated_task_mu = np.zeros(n_simulations)
    simulated_task_var = np.zeros(n_simulations)
    simulated_reliability = np.zeros((n_simulations, len(n_trials_list)))

    ###
    # run the simulation
    for s in range(n_simulations):
        if generate_means:
            # generate means (proficiencies)
            mu_subjects = get_subject_means(mu_subjects_mu, mu_subjects_std, N, distribution, rng_mu)

        # generate matrix of trials, total_n_trials per participants
        tmp = bin_samples_rand4(rng, N, total_n_trials, mu_subjects, ravel=False)

        # update=redefine generators
        if update_rng_per_simulation:
            rng_mu = np.random.default_rng(s + 444)
            rng = np.random.default_rng(s + 444)

        # define random state for the simulation, used for splitting data in half
        rng_sim = np.random.default_rng(s)

        # save the group mean and variance
        simulated_task_mu[s] = tmp.mean(axis=1).mean()
        simulated_task_var[s] = tmp.mean(axis=1).var()

        #### Calculate reliability
        # define corr array
        array_corr_trials_psychofit = np.zeros((len(n_trials_list), n_repeats))

        for j, n_trials in enumerate(n_trials_list):

            # check that it's possible
            assert n_trials <= total_n_trials // 2

            # go over iterations
            for i in range(n_repeats):
                # instead of splitting it, I will sample = shuffle the array and then take first and second half
                arr_u = None
                arr_u = rng_sim.choice(tmp, size=(n_trials * 2), replace=False, axis=1)

                # calculate correlation
                array_corr_trials_psychofit[j, i] = np.corrcoef(arr_u[:, :n_trials].mean(axis=1),
                                                                arr_u[:, n_trials:].mean(axis=1))[0, 1]

        # save the reliability
        simulated_reliability[s, :] = array_corr_trials_psychofit.mean(axis=1)  #np.nanmean(array_corr_trials_psychofit, axis=1)

        ### Do the fit
        # go over all the functions
        for (k, fx) in enumerate(fit_functions):
            result = None
            gmodel = None

            # initiate the model
            gmodel = Model(fx, nan_policy='omit')

            # define variables
            if gmodel.name == 'Model(hyperbolic_fit)':
                x = n_trials_list
                y = simulated_reliability[s]
            else:
                # deal with undefined values
                simulated_reliability[s, simulated_reliability[s] == 0] = np.nan
                # skip if fit is not possible
                if np.isnan(simulated_reliability).all():
                    simulated_C_fit_lin[s] = np.array([np.nan, np.nan])
                    continue

                else:
                    x = 1 / n_trials_list
                    y = 1 / simulated_reliability[s]

            # set params, it differs for different functions
            if len(gmodel.param_names) == 1:
                params = gmodel.make_params(a=3)
            #             print(f"For model {gmodel.name} taking 1 param.")

            elif len(gmodel.param_names) == 2:
                if 'cdf_lognormal' in gmodel.name or 'cdf_normal' in gmodel.name:
                    params = gmodel.make_params(mu=0.3, sigma=0.3)
                else:
                    params = gmodel.make_params(a=3, b=1)
            #             print(f"For model {gmodel.name} taking 2 params.")

            # perform the fit
            try:
                result = gmodel.fit(y, params, x=x)
                # check if we have b
                if 'b' in result.best_values.keys():
                    simulated_C_fit_lin[s] = np.array([result.best_values["a"], result.best_values["b"]])
                else:
                    simulated_C_fit_A[s] = result.best_values["a"]

            except ValueError:
                print(f'Failed to fit {gmodel.name}')
                logs.append(f'Failed to fit {gmodel.name}')

            except TypeError as e:
                print(f'There was an error {e}, failed to fit {gmodel.name}\nParameters were:\nx={x}\ny={y}\nn trials list={n_trials_list}\nntrials={n_trials}\ns={s}\nmean {mu_subjects_mu} and std {mu_subjects_std} for {distribution} distribution.')
                logs.append(f'There was an error {e}, failed to fit {gmodel.name}\nParameters were:\nx={x}\ny={y}\nn trials list={n_trials_list}\nntrials={n_trials}\ns={s}\nmean {mu_subjects_mu} and std {mu_subjects_std} for {distribution} distribution.')

        ## get analytical
        simulated_C_teor[s] = analytical_C(simulated_task_mu[s], simulated_task_var[s])
        simulated_C_teor_unbiased[s] = analytical_C_unbiased(simulated_task_mu[s], simulated_task_var[s], total_n_trials)

    end = time.time()
    # print(f"Process took: {end - start:.2f} s which is {(end - start) / 60:.2f} min.")
    logs.append(f"Process took: {end - start:.2f} s which is {(end - start) / 60:.2f} min.\n")
    logs.append('\n\n-----------------------------------\n')

    # save the logs
    write_log(os.path.join(path_output, log_name), logs)

    if distribution == 'lognormal':
        # calculate lognormal mean and variance
        m = np.exp(mu_subjects_mu + mu_subjects_std**2 / 2.0)
        v = np.exp(2 * mu_subjects_mu + mu_subjects_std**2) * (np.exp(mu_subjects_std**2) - 1)
        expected_C = analytical_C(m, v)
        # expected_C_unbiased = analytical_C_unbiased(m, v, total_n_trials)
    elif distribution == 'gaussian':
        m = np.nan
        v = np.nan
        expected_C = analytical_C(mu_subjects_mu, mu_subjects_std ** 2)
        # expected_C_unbiased = analytical_C_unbiased(mu_subjects_mu, mu_subjects_std ** 2, total_n_trials)
    elif distribution == 'beta':
        # note that for beta distribution, the coefficients are labelled as alpha=mu_subjects_mu and beta=mu_subjects_std
        m = mu_subjects_mu/(mu_subjects_mu+mu_subjects_std)
        v = mu_subjects_mu*mu_subjects_std / ((mu_subjects_mu+mu_subjects_std+1) * (mu_subjects_mu+mu_subjects_std)**2)
        expected_C = mu_subjects_mu + mu_subjects_std #analytical_C(m, v) --> See the derivation
        # expected_C_unbiased = analytical_C_unbiased(m, v, total_n_trials)
    else:
        m, v, expected_C, expected_C_unbiased = None, None, None, None

    # save
    if save:
        pd.DataFrame({
                       'hyperbolic_fit': simulated_C_fit_A,
                       'fit_linear_slope': simulated_C_fit_lin[:, 0],
                       'fit_linear_intercept': simulated_C_fit_lin[:, 1],
                       'fit_theoretical': simulated_C_teor,
                       'fit_theoretical_unbiased': simulated_C_teor_unbiased,
                       'distribution': distribution,
                       'true_C': true_C,  # C coefficient
                       # 'true_C_unbiased': true_C_unbiased,
                       'expected_C': expected_C,
                       # 'expected_C_unbiased': expected_C_unbiased,
                       'mu_or_alpha': mu_subjects_mu,
                       'sigma_or_beta': mu_subjects_std,
                       'mu_lognorm_or_beta': m,
                       'var_lognorm_or_beta': v,
                       'N': N,
                   }).to_csv(os.path.join(path_output, f'C_coefficients_{distribution}_{n_simulations}sim_{N}_{mu_subjects_mu}_{mu_subjects_std}.csv'), index=False)

    return simulated_C_fit_A, simulated_C_fit_lin, simulated_C_teor, simulated_C_teor_unbiased, simulated_task_mu, simulated_task_var, simulated_reliability


def run_estimate_C_with_unbiased_explosions(params, N, total_n_trials, n_trials_list, n_simulations=10**2, n_repeats=10**2,
                    distribution='gaussian', fit_functions=[linear, hyperbolic_fit], rng=None, rng_mu=None,
                    update_rng_per_simulation=False, rng_C=None, unconstrained_N=10**7, save=True,
                    mu_subjects=None, vary='N', path_output=path_output, atol=1*10**-3):
    """
    Run simulations of experiments and return values of C (previously named BA) fits using four different methods
    (added unbiased C), checks for explosions -- times when the denominator of unbiased C comes within "atol" of
    distance to zero which makes the formula spit out crazy high and low values (explode).

    Parameters
    ----------
    params: tuple of floats, mean and sigma of the distribution we want to sample
    N: int, number of participants
    total_n_trials: int, length of sequence of 1 and 0 that will be generated for each subject
    n_trials_list: list/array of int, which L (chunks/samples of data) to go over when calculating reliability to create
                    a good data for the fit
    n_simulations: int, default 100, how many simulations we want to run
    n_repeats: int, default 100, repeats for reliability sampling -- how many times to calculate the reliability
    distribution: str, default gaussian, only lognormal or gaussian are implemented, which distribution to sample
    fit_functions: list of functions, default [linear, hyperbolic_fit], functions we want to use for fitting using lmfit
    rng: np random state, default np.random.default_rng(0), for reproducibility in generating the sample distribution
    rng_mu: rng: np random state, default np.random.default_rng(0), for reproducibility in generating distribution of mu
    update_rng_per_simulation: bool, default False, if True, it will generate a new seed (rng) for every iteration of
                    the simulations in creating the data
    rng_C: np random state, default np.random.default_rng(0), for reproducibility, generates the unconstrained true C distribution
    unconstrained_N: int, default 10**7, size (N) of the unconstrained distribution that will be used to determine true C
    save: bool, default True, whether to save the results
    mu_subjects: array of floats <0,1> of len N, default None, array of probabilities of success (p) for each subject,
                    if None, it will be computed from given sigma and variance (params)
    vary: str, default 'N', label, whether we are varying N or n_trials
    path_output: str, where to save the results, default should be the same as stated above
    atol: float, default 1*10 ** -3, how close should the denominator be to 0 to count it as an explosion and discard

    Returns
    -------
    7 arrays:
    * three are the three fits of size n_simulations (one of them has a shape (2, n_simulations) since it has intercept
      and slope)
    * mean and variance of the generated distribution, each of shape n_simulations
    * reliability array of shape (n_simulations, len(n_trials_list)) providing reliability for each simulation and each
      value of L
    * array of number of explosions of size n_simulations

    Can save csv with pandas dataframe.
    True C: computed using either original or corrected formula, calculates the C coefficient on
        large sample (of size unconstrained_N) and reflects the true value of C coefficient that we should find. Note
        that for large L, these two should be identical. The unbiased C does not make much sense in this particular
        design because we generate the P distribution (actual proficiencies) and not its approximation Z, so it's
        disabled at the moment.
    Expected C or expected unbiased C: computed using the distribution definitions. This can be not precise and should
        be used with caution as if used with distributions that need clipping (aren't between 0 and 1, e.g. some
        gaussian), these will give wrong and erroneous results. On the other hand, this can serve as a QC for the
        sampled distribution.

    """
    # define saving
    if vary == "N":
        path_output = os.path.join(path_output, str(N))
    elif vary == "n_trials":
        path_output = os.path.join(path_output, str(total_n_trials))

    # define seeds, it is done this way to avoid different and inconsistent seeds if run in parallel
    if rng is None:
        rng = np.random.default_rng(0)
    if rng_C is None:
        rng_C = np.random.default_rng(0)
    if rng_mu is None:
        rng_mu = np.random.default_rng(0)

    # start the timer
    start = time.time()

    # save logs
    logs = []

    # unpack params
    mu_subjects_mu, mu_subjects_std = params
    logs.append(f'Processing mean/a {mu_subjects_mu} and std/b {mu_subjects_std} for {distribution} distribution.\n')

    # save log name
    log_name = f'log_{distribution}_N{N}_{mu_subjects_mu}_{mu_subjects_std}.txt'

    generate_means = False
    if mu_subjects is None:
        # generate means
        generate_means = True
        # mu_subjects = get_subject_means(mu_subjects_mu, mu_subjects_std, N, distribution, rng)

        # create a sample that would determine the true C, it must be a big one to estimate the true C because because
        # of the clipping, the distribution that we create does not need to be the same as we define
        real_distribution = get_subject_means(mu_subjects_mu, mu_subjects_std, unconstrained_N, distribution, rng_C)

        # compute and save true C based on created distribution (what we sample from but bigger)
        true_C = analytical_C(real_distribution.mean(), real_distribution.var())

        # note
        logs.append('Generating distribution because none was provided.\n\n')

    # define arrays
    simulated_C_teor = np.zeros(n_simulations)
    simulated_C_teor_unbiased = np.zeros(n_simulations)
    simulated_C_fit_A = np.zeros(n_simulations)
    simulated_C_fit_lin = np.zeros((n_simulations, 2))

    simulated_task_mu = np.zeros(n_simulations)
    simulated_task_var = np.zeros(n_simulations)
    simulated_reliability = np.zeros((n_simulations, len(n_trials_list)))
    count_explosions_arr = np.zeros(n_simulations)

    ###
    # run the simulation
    s = 0
    while s < n_simulations:
        if generate_means:
            # generate means (proficiencies)
            mu_subjects = get_subject_means(mu_subjects_mu, mu_subjects_std, N, distribution, rng_mu)

        # generate matrix of trials, total_n_trials per participants
        tmp = bin_samples_rand4(rng, N, total_n_trials, mu_subjects, ravel=False)

        # update=redefine generators
        if update_rng_per_simulation:
            rng_mu = np.random.default_rng(s+444)
            rng = np.random.default_rng(s+444)

        # define random state for the simulation, used for splitting data in half
        rng_sim = np.random.default_rng(s)

        # get the stats
        mu = tmp.mean()
        var = tmp.mean(axis=1).var()
        L = tmp.shape[1]
        # at the moment checking only if it's close to 0
        if np.isclose(var - (mu - mu**2) / L, 0, atol=atol):
            count_explosions_arr[s] += 1
            # continue # TODO uncomment if needed

        # save the group mean and variance
        simulated_task_mu[s] = mu #tmp.mean(axis=1).mean()
        simulated_task_var[s] = var #tmp.mean(axis=1).var()


        #### Calculate reliability
        # define corr array
        array_corr_trials_psychofit = np.zeros((len(n_trials_list), n_repeats))

        for j, n_trials in enumerate(n_trials_list):

            # check that it's possible
            assert n_trials <= total_n_trials // 2

            # go over iterations
            for i in range(n_repeats):
                # instead of splitting it, I will sample = shuffle the array and then take first and second half
                arr_u = None
                arr_u = rng_sim.choice(tmp, size=(n_trials * 2), replace=False, axis=1)

                # calculate correlation
                array_corr_trials_psychofit[j, i] = np.corrcoef(arr_u[:, :n_trials].mean(axis=1),
                                                                arr_u[:, n_trials:].mean(axis=1))[0, 1]

        # save the reliability
        simulated_reliability[s, :] = array_corr_trials_psychofit.mean(axis=1)  #np.nanmean(array_corr_trials_psychofit, axis=1)

        ### Do the fit
        # go over all the functions
        for (k, fx) in enumerate(fit_functions):
            result = None
            gmodel = None

            # initiate the model
            gmodel = Model(fx, nan_policy='omit')

            # define variables
            if gmodel.name == 'Model(hyperbolic_fit)':
                x = n_trials_list
                y = simulated_reliability[s]
            else:
                # deal with undefined values
                simulated_reliability[s, simulated_reliability[s] == 0] = np.nan
                # skip if fit is not possible
                if np.isnan(simulated_reliability).all():
                    simulated_C_fit_lin[s] = np.array([np.nan, np.nan])
                    continue

                else:
                    x = 1 / n_trials_list
                    y = 1 / simulated_reliability[s]

            # set params, it differs for different functions
            if len(gmodel.param_names) == 1:
                params = gmodel.make_params(a=3)
            #             print(f"For model {gmodel.name} taking 1 param.")

            elif len(gmodel.param_names) == 2:
                if 'cdf_lognormal' in gmodel.name or 'cdf_normal' in gmodel.name:
                    params = gmodel.make_params(mu=0.3, sigma=0.3)
                else:
                    params = gmodel.make_params(a=3, b=1)
            #             print(f"For model {gmodel.name} taking 2 params.")

            # perform the fit
            try:
                result = gmodel.fit(y, params, x=x)
                # check if we have b
                if 'b' in result.best_values.keys():
                    simulated_C_fit_lin[s] = np.array([result.best_values["a"], result.best_values["b"]])
                else:
                    simulated_C_fit_A[s] = result.best_values["a"]

            except ValueError:
                print(f'Failed to fit {gmodel.name}')
                logs.append(f'Failed to fit {gmodel.name}')

            except TypeError as e:
                print(f'There was an error {e}, failed to fit {gmodel.name}\nParameters were:\nx={x}\ny={y}\nn trials list={n_trials_list}\nntrials={n_trials}\ns={s}\nmean {mu_subjects_mu} and std {mu_subjects_std} for {distribution} distribution.')
                logs.append(f'There was an error {e}, failed to fit {gmodel.name}\nParameters were:\nx={x}\ny={y}\nn trials list={n_trials_list}\nntrials={n_trials}\ns={s}\nmean {mu_subjects_mu} and std {mu_subjects_std} for {distribution} distribution.')

        ## get analytical
        simulated_C_teor[s] = analytical_C(simulated_task_mu[s], simulated_task_var[s])
        simulated_C_teor_unbiased[s] = analytical_C_unbiased(simulated_task_mu[s], simulated_task_var[s], total_n_trials)

        # increase count
        s += 1

    end = time.time()
    # print(f"Process took: {end - start:.2f} s which is {(end - start) / 60:.2f} min.")
    logs.append(f"Process took: {end - start:.2f} s which is {(end - start) / 60:.2f} min.\n")
    logs.append('\n\n-----------------------------------\n')

    # save the logs
    write_log(os.path.join(path_output, log_name), logs)

    if distribution == 'lognormal':
        # calculate lognormal mean and variance
        m = np.exp(mu_subjects_mu + mu_subjects_std**2 / 2.0)
        v = np.exp(2 * mu_subjects_mu + mu_subjects_std**2) * (np.exp(mu_subjects_std**2) - 1)
        expected_C = analytical_C(m, v)
        # expected_C_unbiased = analytical_C_unbiased(m, v, total_n_trials)
    elif distribution == 'gaussian':
        m = np.nan
        v = np.nan
        expected_C = analytical_C(mu_subjects_mu, mu_subjects_std ** 2)
        # expected_C_unbiased = analytical_C_unbiased(mu_subjects_mu, mu_subjects_std ** 2, total_n_trials)
    elif distribution == 'beta':
        # note that for beta distribution, the coefficients are labelled as alpha=a=mu_subjects_mu and beta=b=mu_subjects_std
        m = mu_subjects_mu/(mu_subjects_mu+mu_subjects_std)
        v = mu_subjects_mu*mu_subjects_std / ((mu_subjects_mu+mu_subjects_std+1) * (mu_subjects_mu+mu_subjects_std)**2)
        expected_C = mu_subjects_mu + mu_subjects_std #analytical_C(m, v) --> See the derivation
        # expected_C_unbiased = analytical_C_unbiased(m, v, total_n_trials)
    else:
        m, v, expected_C, expected_C_unbiased = None, None, None, None

    # save
    if save:
        pd.DataFrame({
                       'hyperbolic_fit': simulated_C_fit_A,
                       'fit_linear_slope': simulated_C_fit_lin[:, 0],
                       'fit_linear_intercept': simulated_C_fit_lin[:, 1],
                       'fit_theoretical': simulated_C_teor,
                       'fit_theoretical_unbiased': simulated_C_teor_unbiased,
                       'distribution': distribution,
                       'true_C': true_C,  # C coefficient
                       'expected_C': expected_C,
                       # 'expected_C_unbiased': expected_C_unbiased,
                       'mu_or_a': mu_subjects_mu,
                       'sigma_or_b': mu_subjects_std,
                       'mu_lognorm_or_beta': m,
                       'var_lognorm_or_beta': v,
                       'N': N,
                       'exploded': count_explosions_arr,
                   }).to_csv(os.path.join(path_output, f'C_coefficients_{distribution}_{n_simulations}sim_{N}_{mu_subjects_mu}_{mu_subjects_std}.csv'), index=False)

    return simulated_C_fit_A, simulated_C_fit_lin, simulated_C_teor, simulated_C_teor_unbiased, simulated_task_mu, simulated_task_var, simulated_reliability, count_explosions_arr


def run_simulation_unbiasedC_space_explosions(all_param_combinations_clean, Ns_array, n_trials_array, total_N, total_n_trials,
                   n_simulations=10**2, unconstrained_N=10**7, path_output='./', save_name='simulation_C_error',
                   rng=None, rng_mu=None, rng_C=None, rng_k=None, rng_l=None, run_multiprocess=True, update_rng_per_simulation=False, atol=1*10**-3):
    """

    Run simulations of experiments and return values of C (previously named BA) of the MV (analytical) fit, checks for
    explosions -- times when the denominator of unbiased C comes within "atol" of distance to zero which makes the
    formula spit out crazy high and low values (explode). Return also median and mean distances from the true C that
    is determined using "uncontrained_N" number of subjects. The function offers options for specifying seeds for
    reproducibility that account for running the script in parallel. Every sampling has its own generator that can be
    provided as a parameter.

    Parameters
    ----------
    all_param_combinations_clean: tuple of str and floats; expected C, distribution type, its mean and sigma we want to sample
    Ns_array: array/list, number of participants
    n_trials_array: list/array of int,  stating length of sequence of 1 and 0 that will be generated for each subject
    total_N: int, define total number of participants, this must be >= the max we sample to (given by max(Ns_array))
    total_n_trials: int, define total number of trials, this must be >= the max we sample to (given by max(n_trials_array))
    n_simulations: int, default 100, how many simulations we want to run
    unconstrained_N: int, default 10**7, size (N) of the unconstrained distribution that will be used to determine true C
    path_output: str, where to save the results, default should be the same as stated above
    save_name: str, default simulation_C_error, output name
    rng: np random state, default np.random.default_rng(0), for reproducibility in generating the sample distribution
    rng_mu: rng: np random state, default np.random.default_rng(0), for reproducibility in generating distribution of mu
    rng_C: np random state, default np.random.default_rng(0), for reproducibility, generates the unconstrained true C distribution
    rng_k: np random state, default np.random.default_rng(0), for reproducibility when sampling N
    rng_l: np random state, default np.random.default_rng(0), for reproducibility when sampling L
    run_multiprocess: bool, default True, whether to run it in multiprocess
    update_rng_per_simulation: bool, default False, if True, it will generate a new seed (rng) for every iteration of
                    the simulations in creating the data
    atol: float, default 1*10 ** -3, how close should the denominator be to 0 to count it as an explosion and discard


    Returns
    -------
    Nothing but it saves all the computations both in csv files as well as numpy arrays. These contain all the simulation
    variables (distribution, mean, var, expectred C, N, L) as well as the actual data (computed C and distance from the
    real C).

    """

    # start the timer
    start = time.time()

    # define df for all the results
    df_all_results = pd.DataFrame()

    # define random state for generating the distributions
    if rng is None:
        rng = np.random.default_rng(0)
    if rng_C is None:
        rng_C = np.random.default_rng(0)
    if rng_mu is None:
        rng_mu = np.random.default_rng(0)

    if rng_l is None:
        rng_l = np.random.default_rng(0)
    if rng_k is None:
        rng_k = np.random.default_rng(0)

    ###
    # iterate over all the combinations of mean and std per distribution
    print('going over: ', all_param_combinations_clean)
    if run_multiprocess:
        # define arrays
        difference_C = np.zeros((n_simulations, len(Ns_array), len(n_trials_array)))
        median_difference_C = np.zeros((len(Ns_array), len(n_trials_array)))
        mean_difference_C = np.zeros((len(Ns_array), len(n_trials_array)))
        #
        simulated_C_teor_unbiased = np.zeros((n_simulations, len(Ns_array), len(n_trials_array)))
        simulated_task_mu = np.zeros((n_simulations, len(Ns_array), len(n_trials_array)))
        simulated_task_var = np.zeros((n_simulations, len(Ns_array), len(n_trials_array)))
        count_explosions_arr = np.zeros((n_simulations, len(Ns_array), len(n_trials_array)))

        # unpack the parameters
        expected_C, distribution, mu_subjects_mu, mu_subjects_std = all_param_combinations_clean

        if distribution == 'lognormal':
            # calculate lognormal mean and variance
            m = np.exp(mu_subjects_mu + mu_subjects_std**2 / 2.0)
            v = np.exp(2 * mu_subjects_mu + mu_subjects_std**2) * (np.exp(mu_subjects_std**2) - 1)
        elif distribution == 'gaussian':
            m = np.nan
            v = np.nan
        elif distribution == 'beta':
            # note that for beta distribution, the coefficients are labelled as alpha=a=mu_subjects_mu and beta=b=mu_subjects_std
            m = mu_subjects_mu / (mu_subjects_mu + mu_subjects_std)
            v = mu_subjects_mu * mu_subjects_std/((mu_subjects_mu+mu_subjects_std+1) * (mu_subjects_mu+mu_subjects_std)**2)
        else:
            m, v = None, None


        # create a sample that would determine the true C, it must be a big one to estimate the true C because because
        # of the clipping, the distribution that we create does not need to be the same as we define
        real_distribution = get_subject_means(mu_subjects_mu, mu_subjects_std, unconstrained_N, distribution, rng_C)

        # compute and save true C based on created distribution (what we sample from but bigger)
        true_C = analytical_C(real_distribution.mean(), real_distribution.var())

        # for the saving
        true_C_arr = np.array([true_C])

        # start timer
        start_C = time.time()

        ###
        # run the simulation
        s = 0
        while s < n_simulations:
            # generate means (proficiencies)
            mu_subjects = get_subject_means(mu_subjects_mu, mu_subjects_std, total_N, distribution, rng_mu)

            # generate matrix of trials, total_n_trials per participants -> generate the full distribution that will be sampled
            tmp = bin_samples_rand4(rng, total_N, total_n_trials, mu_subjects, ravel=False)

            # update=redefine generators
            if update_rng_per_simulation:
                rng_mu = np.random.default_rng(s+444)
                rng = np.random.default_rng(s+444)

            k = 0
            # go over Ns
            while k < len(Ns_array):
                # update=redefine generators
                if update_rng_per_simulation:
                    rng_k = np.random.default_rng(s+k+222)
                # sample given Number of participants
                tmp_N_array = rng_k.choice(tmp, Ns_array[k], axis=0, replace=False)

                l = 0
                while l < len(n_trials_array):
                    # update=redefine generators
                    if update_rng_per_simulation:
                        rng_l = np.random.default_rng(s+l+k+333)
                    # sample n trials
                    all_trials_N_n = rng_l.choice(tmp_N_array, n_trials_array[l], axis=1, replace=False)

                    # get the stats
                    mu = all_trials_N_n.mean()
                    var = all_trials_N_n.mean(axis=1).var()

                    # at the moment checking only if it's close to 0
                    if np.isclose(var - (mu - mu ** 2) / n_trials_array[l], 0, atol=atol):
                        count_explosions_arr[s, k, l] += 1
                        # continue # TODO uncomment if needed
                    else:
                        if np.abs(analytical_C_unbiased(mu, var, n_trials_array[l])) > 5000:
                            print(f'This is bad:\nC = {simulated_C_teor_unbiased[s, k, l]}\nN = {Ns_array[k]}, L={n_trials_array[l]}, s={s}\nmu={mu}\nvar={var}\n')

                    # save the group mean and variance
                    simulated_task_mu[s, k, l] = mu  #all_trials_N_n.mean()
                    simulated_task_var[s, k, l] = var  #all_trials_N_n.mean(axis=1).var()

                    # get analytical fit
                    assert all_trials_N_n.shape[1] == n_trials_array[l]
                    simulated_C_teor_unbiased[s, k, l] = analytical_C_unbiased(mu,
                                                                               var,
                                                                               n_trials_array[l])

                    # increase count
                    l += 1

                # subtract the true C from the estimated one -- could be done across Ns and simulations
                difference_C[s, k, :] = simulated_C_teor_unbiased[s, k, :] - true_C

                # save those results
                df_all_results = df_all_results.append(pd.DataFrame({
                    'distribution': distribution,
                    'true_C': true_C,  # C coefficient
                    'expected_C': expected_C,  # what was defined
                    'mu_or_a': mu_subjects_mu,
                    'sigma_or_b': mu_subjects_std,
                    'mu_lognorm_or_beta': m,
                    'var_lognorm_or_beta': v,
                    'mu_simulated': simulated_task_mu[s, k, :],
                    'var_simulated': simulated_task_var[s, k, :],
                    'N': Ns_array[k],
                    'total_n_trials': n_trials_array,
                    'fit_theoretical': simulated_C_teor_unbiased[s, k, :],
                    'fit_theoretical_dist': difference_C[s, k, :],
                    'exploded': count_explosions_arr[s, k, :],
                }))

                # increase count
                k += 1

            # increase count
            s += 1

        # compute median error per C across the simulations
        median_difference_C = np.median(difference_C, axis=0) / true_C
        mean_difference_C = np.mean(difference_C, axis=0) / true_C

        print(f"C took: {time.time() - start_C:.2f} s which is {(time.time() - start) / 60:.2f} min.")


    else:
        # define arrays
        difference_C = np.zeros((len(all_param_combinations_clean), n_simulations, len(Ns_array), len(n_trials_array)))
        median_difference_C = np.zeros((len(all_param_combinations_clean), len(Ns_array), len(n_trials_array)))
        mean_difference_C = np.zeros((len(all_param_combinations_clean), len(Ns_array), len(n_trials_array)))
        #
        simulated_C_teor_unbiased = np.zeros((len(all_param_combinations_clean), n_simulations, len(Ns_array), len(n_trials_array)))
        simulated_task_mu = np.zeros((len(all_param_combinations_clean), n_simulations, len(Ns_array), len(n_trials_array)))
        simulated_task_var = np.zeros((len(all_param_combinations_clean), n_simulations, len(Ns_array), len(n_trials_array)))
        count_explosions_arr = np.zeros((len(all_param_combinations_clean), n_simulations, len(Ns_array), len(n_trials_array)))

        true_C_arr = np.zeros(len(all_param_combinations_clean))

        for i, (expected_C, distribution, mu_subjects_mu, mu_subjects_std) in enumerate(all_param_combinations_clean):

            if distribution == 'lognormal':
                # calculate lognormal mean and variance
                m = np.exp(mu_subjects_mu + mu_subjects_std ** 2 / 2.0)
                v = np.exp(2 * mu_subjects_mu + mu_subjects_std ** 2) * (np.exp(mu_subjects_std ** 2) - 1)
            elif distribution == 'gaussian':
                m = np.nan
                v = np.nan
            elif distribution == 'beta':
                # note that for beta distribution, the coefficients are labelled as alpha=a=mu_subjects_mu and beta=b=mu_subjects_std
                m = mu_subjects_mu / (mu_subjects_mu + mu_subjects_std)
                v = mu_subjects_mu * mu_subjects_std / (
                            (mu_subjects_mu + mu_subjects_std + 1) * (mu_subjects_mu + mu_subjects_std) ** 2)
            else:
                m, v = None, None

            # create a sample that would determine the true C, it must be a big one to estimate the true C because because
            # of the clipping, the distribution that we create does not need to be the same as we define
            real_distribution = get_subject_means(mu_subjects_mu, mu_subjects_std, unconstrained_N, distribution, rng_C)

            # compute and save true C based on created distribution (what we sample from but bigger)
            true_C = analytical_C(real_distribution.mean(), real_distribution.var())

            # save true C
            true_C_arr[i] = true_C

            # start timer
            start_C = time.time()

            ###
            # run the simulation
            s = 0
            while s < n_simulations:
                # generate means (proficiencies)
                mu_subjects = get_subject_means(mu_subjects_mu, mu_subjects_std, total_N, distribution, rng_mu)

                # generate matrix of trials, total_n_trials per participants -> generate the full distribution that will be sampled
                tmp = bin_samples_rand4(rng, total_N, total_n_trials, mu_subjects, ravel=False)

                # update=redefine generators
                if update_rng_per_simulation:
                    rng_mu = np.random.default_rng(s + 444)
                    rng = np.random.default_rng(s + 444)

                k = 0
                # go over Ns
                while k < len(Ns_array):
                    # update=redefine generators
                    if update_rng_per_simulation:
                        rng_k = np.random.default_rng(s + k + 222)
                    # sample given Number of participants
                    tmp_N_array = rng_k.choice(tmp, Ns_array[k], axis=0, replace=False)

                    l = 0
                    while l < len(n_trials_array):
                        if update_rng_per_simulation:
                            rng_l = np.random.default_rng(s + l + k + 333)
                        # sample n trials
                        all_trials_N_n = rng_l.choice(tmp_N_array, n_trials_array[l], axis=1, replace=False)

                        # get the stats
                        mu = all_trials_N_n.mean()
                        var = all_trials_N_n.mean(axis=1).var()
                        # at the moment checking only if it's close to 0
                        if np.isclose(var - (mu - mu ** 2) / n_trials_array[l], 0, atol=atol):
                            count_explosions_arr[i, s, k, l] += 1
                            # continue # TODO uncomment if needed

                        # save the group mean and variance
                        simulated_task_mu[i, s, k, l] = mu  # all_trials_N_n.mean()
                        simulated_task_var[i, s, k, l] = var  # all_trials_N_n.mean(axis=1).var()

                        # get analytical fit
                        simulated_C_teor_unbiased[i, s, k, l] = analytical_C_unbiased(mu,
                                                                                      var,
                                                                                      n_trials_array[l])

                        # increase count
                        l += 1

                    # subtract the true C from the estimated one -- could be done across Ns and simulations
                    difference_C[i, s, k, :] = simulated_C_teor_unbiased[i, s, k, :] - true_C

                    # save those results
                    df_all_results = df_all_results.append(pd.DataFrame({
                        'distribution': distribution,
                        'true_C': true_C,  # C coefficient
                        'expected_C': expected_C,  # what was defined
                        'mu_or_a': mu_subjects_mu,
                        'sigma_or_b': mu_subjects_std,
                        'mu_lognorm_or_beta': m,
                        'var_lognorm_or_beta': v,
                        'N': Ns_array[k],
                        'total_n_trials': n_trials_array,
                        'fit_theoretical': simulated_C_teor_unbiased[i, s, k, :],
                        'fit_theoretical_dist': difference_C[i, s, k, :],
                        'exploded': count_explosions_arr[i, s, k, :],
                    }))

                    # increase count
                    k += 1

                # increase count
                s += 1

            # compute median error per C across the simulations
            median_difference_C[i] = np.median(difference_C[i], axis=0) / true_C
            mean_difference_C[i] = np.mean(difference_C[i], axis=0) / true_C

            print(f"C took: {time.time() - start_C:.2f} s which is {(time.time() - start) / 60:.2f} min.")

    # save the dataframe
    df_all_results.to_csv(os.path.join(path_output, save_name+f'_{distribution}_{mu_subjects_mu}_{mu_subjects_std}.csv'), index=False)

    # save all the arrays
    np.savez(
        os.path.join(path_output, save_name+f'_{distribution}_{mu_subjects_mu}_{mu_subjects_std}.npz'),
        median_difference_C=median_difference_C,
        mean_difference_C=mean_difference_C,
        difference_C=difference_C,
        Ns_array=Ns_array,
        total_N=np.array(total_N),
        n_trials_array=n_trials_array,
        total_n_trials=np.array(total_n_trials),
        true_C=true_C_arr,
        simulated_task_mu=simulated_task_mu,
        simulated_task_var=simulated_task_var,
        simulated_C_teor=simulated_C_teor_unbiased,
        all_parameters=np.array(all_param_combinations_clean),
        count_explosions_arr=count_explosions_arr,
    )

    end = time.time()
    print(f"Process took: {end - start:.2f} s which is {(end - start) / 60:.2f} min.")


def run_estimate_C_with_unbiased_fit_theory_only(params, N, total_n_trials, n_simulations=10 ** 2,
                                                 distribution='gaussian',
                                                 rng=None, rng_mu=None, update_rng_per_simulation=False,
                                                 rng_C=None, unconstrained_N=10 ** 7, save=True,
                                                 mu_subjects=None, vary='N',
                                                 path_output=path_output):
    """
    Run simulations of experiments and return values of C (previously named BA) fits using only one method
    (theory -- unbiased C)

    Parameters
    ----------
    params: tuple of floats, mean and sigma of the distribution we want to sample
    N: int, number of participants
    total_n_trials: int, length of sequence of 1 and 0 that will be generated for each subject
    n_simulations: int, default 100, how many simulations we want to run
    distribution: str, default gaussian, only lognormal or gaussian or beta are implemented, which distribution to sample
    rng: np random state, default np.random.default_rng(0), for reproducibility in generating the sample distribution
    rng_mu: rng: np random state, default np.random.default_rng(0), for reproducibility in generating distribution of mu
    update_rng_per_simulation: bool, default False, if True, it will generate a new seed (rng) for every iteration of
                    the simulations in creating the data
    rng_C: np random state, default np.random.default_rng(0), for reproducibility, generates the unconstrained true C distribution
    unconstrained_N: int, default 10**7, size (N) of the unconstrained distribution that will be used to determine true C
    save: bool, default True, whether to save the results
    mu_subjects: array of floats <0,1> of len N, default None, array of probabilities of success (p) for each subject,
                    if None, it will be computed from given sigma and variance (params)
    vary: str, default 'N', label, whether we are varying N or n_trials
    path_output: str, where to save the results, default should be the same as stated above

    Returns
    -------
    3 arrays:
    * mean and variance of the generated distribution, each of shape n_simulations

    Can save csv with pandas dataframe.
    True C or true unbiased C: computed using either original or corrected formula, calculates the C coefficient on
        large sample (of size unconstrained_N) and reflects the true value of C coefficient that we should find. Note
        that for large L, these two should be identical. The unbiased C does not make much sense in this particular
        design because we generate the P distribution (actual proficiencies) and not its approximation Z, so it's
        disabled at the moment.
    Expected C or expected unbiased C: computed using the distribution definitions. This can be not precise and should
        be used with caution as if used with distributions that need clipping (aren't between 0 and 1, e.g. some
        gaussian), these will give wrong and erroneous results. On the other hand, this can serve as a QC for the
        sampled distribution.

    """
    # define saving
    if vary == "N":
        path_output = os.path.join(path_output, str(N))
    elif vary == "n_trials":
        path_output = os.path.join(path_output, str(total_n_trials))

    # define seeds, it is done this way to avoid different and inconsistent seeds if run in parallel
    if rng is None:
        rng = np.random.default_rng(0)
    if rng_C is None:
        rng_C = np.random.default_rng(0)
    if rng_mu is None:
        rng_mu = np.random.default_rng(0)

    # start the timer
    start = time.time()

    # save logs
    logs = []

    # unpack params
    mu_subjects_mu, mu_subjects_std = params
    logs.append(f'Processing mean/a {mu_subjects_mu} and std/b {mu_subjects_std} for {distribution} distribution.\n')

    # save log name
    log_name = f'log_{distribution}_N{N}_{mu_subjects_mu}_{mu_subjects_std}.txt'

    generate_means = False
    if mu_subjects is None:
        # generate means
        generate_means = True
        # mu_subjects = get_subject_means(mu_subjects_mu, mu_subjects_std, N, distribution, rng)

        # create a sample that would determine the true C, it must be a big one to estimate the true C because because
        # of the clipping, the distribution that we create does not need to be the same as we define
        real_distribution = get_subject_means(mu_subjects_mu, mu_subjects_std, unconstrained_N, distribution, rng_C)

        # compute and save true C based on created distribution (what we sample from but bigger)
        true_C = analytical_C(real_distribution.mean(), real_distribution.var())
        # true_C_unbiased = analytical_C_unbiased(real_distribution.mean(), real_distribution.var(), total_n_trials)

        # note
        logs.append('Generating distribution because none was provided.\n\n')

    # define arrays
    simulated_C_teor_unbiased = np.zeros(n_simulations)

    simulated_task_mu = np.zeros(n_simulations)
    simulated_task_var = np.zeros(n_simulations)

    ###
    # run the simulation
    for s in range(n_simulations):
        if generate_means:
            # generate means (proficiencies)
            mu_subjects = get_subject_means(mu_subjects_mu, mu_subjects_std, N, distribution, rng_mu)

        # generate matrix of trials, total_n_trials per participants
        tmp = bin_samples_rand4(rng, N, total_n_trials, mu_subjects, ravel=False)

        # update=redefine generators
        if update_rng_per_simulation:
            rng_mu = np.random.default_rng(s + 444)
            rng = np.random.default_rng(s + 444)
        # define random state for the simulation, used for splitting data in half
        # rng_sim = np.random.default_rng(s)

        # save the group mean and variance
        simulated_task_mu[s] = tmp.mean(axis=1).mean()
        simulated_task_var[s] = tmp.mean(axis=1).var()

        # get analytical
        simulated_C_teor_unbiased[s] = analytical_C_unbiased(simulated_task_mu[s], simulated_task_var[s],
                                                             total_n_trials)

    end = time.time()
    # print(f"Process took: {end - start:.2f} s which is {(end - start) / 60:.2f} min.")
    logs.append(f"Process took: {end - start:.2f} s which is {(end - start) / 60:.2f} min.\n")
    logs.append('\n\n-----------------------------------\n')

    # save the logs
    write_log(os.path.join(path_output, log_name), logs)

    if distribution == 'lognormal':
        # calculate lognormal mean and variance
        m = np.exp(mu_subjects_mu + mu_subjects_std ** 2 / 2.0)
        v = np.exp(2 * mu_subjects_mu + mu_subjects_std ** 2) * (np.exp(mu_subjects_std ** 2) - 1)
        expected_C = analytical_C(m, v)
        # expected_C_unbiased = analytical_C_unbiased(m, v, total_n_trials)
    elif distribution == 'gaussian':
        m = np.nan
        v = np.nan
        expected_C = analytical_C(mu_subjects_mu, mu_subjects_std ** 2)
        # expected_C_unbiased = analytical_C_unbiased(mu_subjects_mu, mu_subjects_std ** 2, total_n_trials)
    elif distribution == 'beta':
        # note that for beta distribution, the coefficients are labelled as alpha=mu_subjects_mu and beta=mu_subjects_std
        m = mu_subjects_mu / (mu_subjects_mu + mu_subjects_std)
        v = mu_subjects_mu * mu_subjects_std / (
                    (mu_subjects_mu + mu_subjects_std + 1) * (mu_subjects_mu + mu_subjects_std) ** 2)
        expected_C = mu_subjects_mu + mu_subjects_std  # analytical_C(m, v) --> See the derivation
        # expected_C_unbiased = analytical_C_unbiased(m, v, total_n_trials)
    else:
        m, v, expected_C, expected_C_unbiased = None, None, None, None

    # save
    if save:
        pd.DataFrame({
            'fit_theoretical_unbiased': simulated_C_teor_unbiased,
            'distribution': distribution,
            'true_C': true_C,  # C coefficient
            # 'true_C_unbiased': true_C_unbiased,
            'expected_C': expected_C,
            # 'expected_C_unbiased': expected_C_unbiased,
            'mu_or_alpha': mu_subjects_mu,
            'sigma_or_beta': mu_subjects_std,
            'mu_lognorm_or_beta': m,
            'var_lognorm_or_beta': v,
            'N': N,
        }).to_csv(os.path.join(path_output,
                               f'C_coefficients_{distribution}_{n_simulations}sim_{N}_{mu_subjects_mu}_{mu_subjects_std}.csv'),
                  index=False)

    return simulated_task_mu, simulated_task_var, simulated_C_teor_unbiased


def run_estimate_C_with_unbiased_fit_theory_only_explosions(params, N, total_n_trials, n_simulations=10 ** 2,
                                                 distribution='gaussian',
                                                 rng=None, rng_mu=None, update_rng_per_simulation=False,
                                                 rng_C=None, unconstrained_N=10 ** 7, save=True,
                                                 mu_subjects=None, vary='N',
                                                 path_output=path_output, atol=1*10**-3):
    """
    Run simulations of experiments and return values of C (previously named BA) fits using four different methods
    (added unbiased C)

    Parameters
    ----------
    params: tuple of floats, mean and sigma of the distribution we want to sample
    N: int, number of participants
    total_n_trials: int, length of sequence of 1 and 0 that will be generated for each subject
    n_simulations: int, default 100, how many simulations we want to run
    distribution: str, default gaussian, only lognormal or gaussian are implemented, which distribution to sample
    rng: np random state, default np.random.default_rng(0), for reproducibility in generating the sample distribution
    rng_mu: rng: np random state, default np.random.default_rng(0), for reproducibility in generating distribution of mu
    update_rng_per_simulation: bool, default False, if True, it will generate a new seed (rng) for every iteration of
                    the simulations in creating the data
    rng_C: np random state, default np.random.default_rng(0), for reproducibility, generates the unconstrained true C distribution
    unconstrained_N: int, default 10**7, size (N) of the unconstrained distribution that will be used to determine true C
    save: bool, default True, whether to save the results
    mu_subjects: array of floats <0,1> of len N, default None, array of probabilities of success (p) for each subject,
                    if None, it will be computed from given sigma and variance (params)
    vary: str, default 'N', label, whether we are varying N or n_trials
    path_output: str, where to save the results, default should be the same as stated above
    atol: float, default 1*10 ** -3, how close should the denominator be to 0 to count it as an explosion and discard

    Returns
    -------
    3 arrays:
    * mean and variance of the generated distribution, each of shape n_simulations
    * array of number/count of explosions of shape LxN

    Can save csv with pandas dataframe.
    True C or true unbiased C: computed using either original or corrected formula, calculates the C coefficient on
        large sample (of size unconstrained_N) and reflects the true value of C coefficient that we should find. Note
        that for large L, these two should be identical. The unbiased C does not make much sense in this particular
        design because we generate the P distribution (actual proficiencies) and not its approximation Z, so it's
        disabled at the moment.
    Expected C or expected unbiased C: computed using the distribution definitions. This can be not precise and should
        be used with caution as if used with distributions that need clipping (aren't between 0 and 1, e.g. some
        gaussian), these will give wrong and erroneous results. On the other hand, this can serve as a QC for the
        sampled distribution.

    """
    # define saving
    if vary == "N":
        path_output = os.path.join(path_output, str(N))
    elif vary == "n_trials":
        path_output = os.path.join(path_output, str(total_n_trials))

    # define seeds, it is done this way to avoid different and inconsistent seeds if run in parallel
    if rng is None:
        rng = np.random.default_rng(0)
    if rng_C is None:
        rng_C = np.random.default_rng(0)
    if rng_mu is None:
        rng_mu = np.random.default_rng(0)

    # start the timer
    start = time.time()

    # save logs
    logs = []

    # unpack params
    mu_subjects_mu, mu_subjects_std = params
    logs.append(f'Processing mean/a {mu_subjects_mu} and std/b {mu_subjects_std} for {distribution} distribution.\n')

    # save log name
    log_name = f'log_{distribution}_N{N}_{mu_subjects_mu}_{mu_subjects_std}.txt'

    generate_means = False
    if mu_subjects is None:
        # generate means
        generate_means = True
        # mu_subjects = get_subject_means(mu_subjects_mu, mu_subjects_std, N, distribution, rng)

        # create a sample that would determine the true C, it must be a big one to estimate the true C because because
        # of the clipping, the distribution that we create does not need to be the same as we define
        real_distribution = get_subject_means(mu_subjects_mu, mu_subjects_std, unconstrained_N, distribution, rng_C)

        # compute and save true C based on created distribution (what we sample from but bigger)
        true_C = analytical_C(real_distribution.mean(), real_distribution.var())
        # true_C_unbiased = analytical_C_unbiased(real_distribution.mean(), real_distribution.var(), total_n_trials)

        # note
        logs.append('Generating distribution because none was provided.\n\n')

    # define arrays
    simulated_C_teor_unbiased = np.zeros(n_simulations)

    simulated_task_mu = np.zeros(n_simulations)
    simulated_task_var = np.zeros(n_simulations)
    count_explosions_arr = np.zeros(n_simulations)

    ###
    # run the simulation
    s = 0
    # for s in range(n_simulations):
    while s < n_simulations:
        if generate_means:
            # generate means (proficiencies)
            mu_subjects = get_subject_means(mu_subjects_mu, mu_subjects_std, N, distribution, rng_mu)

        # generate matrix of trials, total_n_trials per participants
        tmp = bin_samples_rand4(rng, N, total_n_trials, mu_subjects, ravel=False)

        # update=redefine generators
        if update_rng_per_simulation:
            rng_mu = np.random.default_rng(s + 444)
            rng = np.random.default_rng(s + 444)
        # define random state for the simulation, used for splitting data in half
        # rng_sim = np.random.default_rng(s)

        # get the stats
        mu = tmp.mean()
        var = tmp.mean(axis=1).var()
        L = tmp.shape[1]

        # at the moment checking only if it's close to 0
        if np.isclose(var - (mu - mu ** 2) / L, 0, atol=atol):
            count_explosions_arr[s] += 1
            # continue # TODO uncomment

        # save the group mean and variance
        simulated_task_mu[s] = mu #tmp.mean(axis=1).mean()
        simulated_task_var[s] = var #tmp.mean(axis=1).var()

        # get analytical
        simulated_C_teor_unbiased[s] = analytical_C_unbiased(simulated_task_mu[s], simulated_task_var[s],
                                                             total_n_trials)

        # increase count
        s += 1

    end = time.time()
    # print(f"Process took: {end - start:.2f} s which is {(end - start) / 60:.2f} min.")
    logs.append(f"Process took: {end - start:.2f} s which is {(end - start) / 60:.2f} min.\n")
    logs.append('\n\n-----------------------------------\n')

    # save the logs
    write_log(os.path.join(path_output, log_name), logs)

    if distribution == 'lognormal':
        # calculate lognormal mean and variance
        m = np.exp(mu_subjects_mu + mu_subjects_std ** 2 / 2.0)
        v = np.exp(2 * mu_subjects_mu + mu_subjects_std ** 2) * (np.exp(mu_subjects_std ** 2) - 1)
        expected_C = analytical_C(m, v)
        # expected_C_unbiased = analytical_C_unbiased(m, v, total_n_trials)
    elif distribution == 'gaussian':
        m = np.nan
        v = np.nan
        expected_C = analytical_C(mu_subjects_mu, mu_subjects_std ** 2)
        # expected_C_unbiased = analytical_C_unbiased(mu_subjects_mu, mu_subjects_std ** 2, total_n_trials)
    elif distribution == 'beta':
        # note that for beta distribution, the coefficients are labelled as alpha=mu_subjects_mu and beta=mu_subjects_std
        m = mu_subjects_mu / (mu_subjects_mu + mu_subjects_std)
        v = mu_subjects_mu * mu_subjects_std / (
                    (mu_subjects_mu + mu_subjects_std + 1) * (mu_subjects_mu + mu_subjects_std) ** 2)
        expected_C = mu_subjects_mu + mu_subjects_std  # analytical_C(m, v) --> See the derivation
        # expected_C_unbiased = analytical_C_unbiased(m, v, total_n_trials)
    else:
        m, v, expected_C, expected_C_unbiased = None, None, None, None

    # save
    if save:
        pd.DataFrame({
            'fit_theoretical_unbiased': simulated_C_teor_unbiased,
            'distribution': distribution,
            'true_C': true_C,  # C coefficient
            # 'true_C_unbiased': true_C_unbiased,
            'expected_C': expected_C,
            # 'expected_C_unbiased': expected_C_unbiased,
            'mu_or_alpha': mu_subjects_mu,
            'sigma_or_beta': mu_subjects_std,
            'mu_lognorm_or_beta': m,
            'var_lognorm_or_beta': v,
            'N': N,
            'exploded': count_explosions_arr, # TODO remove
        }).to_csv(os.path.join(path_output,
                               f'C_coefficients_explosions_{distribution}_{n_simulations}sim_{N}_{mu_subjects_mu}_{mu_subjects_std}.csv'),
                  index=False)

        # # save all the arrays
        # np.savez(os.path.join(path_output, f'explosions_array_{distribution}_{n_simulations}sim_{N}_{mu_subjects_mu}_{mu_subjects_std}.npz'),
        #          count_explosions_arr=count_explosions_arr, N=np.asarray([N]), L=np.asarray([total_n_trials]))

    return simulated_task_mu, simulated_task_var, simulated_C_teor_unbiased, count_explosions_arr



def run_estimate_BA(params, N, total_n_trials, n_trials_list, n_simulations=10 ** 2, n_repeats=10 ** 2,
                    distribution='gaussian', fit_functions=[linear, hyperbolic_fit], rng=np.random.default_rng(0),
                    rng_BA=np.random.default_rng(0), unconstrained_N=10**7, save=True, mu_subjects=None, vary='N',
                    path_output=path_output):
    """
    Run simulations of experiments and return values of BA fits using three different methods

    Parameters
    ----------
    params: tuple of floats, mean and sigma of the distribution we want to sample
    N: int, number of participants
    total_n_trials: int, length of sequence of 1 and 0 that will be generated for each subject
    n_trials_list: list/array of int, which L (chunks/samples of data) to go over when calculating reliability to create
                    a good data for the fit
    n_simulations: int, default 100, how many simulations we want to run
    n_repeats: int, default 100, repeats for reliability sampling -- how many times to calculate the reliability
    distribution: str, default gaussian, only lognormal or gaussian are implemented, which distribution to sample
    fit_functions: list of functions, default [linear, hyperbolic_fit], functions we want to use for fitting using lmfit
    rng: np random state, default np.random.default_rng(0), for reproducibility
    rng_BA: np random state, default np.random.default_rng(0), for reproducibility, generates the unconstrained true BA distribution
    unconstrained_N: int, default 10**7, size (N) of the unconstrained distribution that will be used to determine true BA
    save: bool, default True, whether to save the results
    mu_subjects: array of floats <0,1> of len N, default None, array of probabilities of success (p) for each subject,
                    if None, it will be computed from given sigma and variance (params)
    vary: str, default 'N', label, whether we are varying N or n_trials
    path_output: str, where to save the results, default should be the same as stated above

    Returns
    -------
    6 arrays:
    * three are the three fits of size n_simulations (one of them has a shape (2, n_simulations) since it has intercept
      and slope)
    * mean and variance of the generated distribution, each of shape n_simulations
    * reliability array of shape (n_simulations, len(n_trials_list)) providing reliability for each simulation and each
      value of L

    """
    # define saving
    if vary == "N":
        path_output = os.path.join(path_output, str(N))
    elif vary == "n_trials":
        path_output = os.path.join(path_output, str(total_n_trials))

    # start the timer
    start = time.time()

    # save logs
    logs = []

    # unpack params
    mu_subjects_mu, mu_subjects_std = params
    logs.append(f'Processing mean {mu_subjects_mu} and std {mu_subjects_std} for {distribution} distribution.\n')

    # save log name
    log_name = f'log_{distribution}_N{N}_{mu_subjects_mu}_{mu_subjects_std}.txt'

    if mu_subjects is None:
        # generate means
        mu_subjects = get_subject_means(mu_subjects_mu, mu_subjects_std, N, distribution, rng)

        # create a sample that would determine the true BA, it must be a big one to estimate the true BA because because
        # of the clipping, the distribution that we create does not need to be the same as we define
        real_distribution = get_subject_means(mu_subjects_mu, mu_subjects_std, unconstrained_N, distribution, rng_BA)

        # compute and save true BA based on created distribution (what we sample from but bigger)
        true_BA = analytical_C(real_distribution.mean(), real_distribution.var())

        # note
        logs.append('Generating distribution because none was provided.\n\n')

    # define arrays
    simulated_BA_teor = np.zeros(n_simulations)
    simulated_BA_fit_A = np.zeros(n_simulations)
    simulated_BA_fit_lin = np.zeros((n_simulations, 2))

    simulated_task_mu = np.zeros(n_simulations)
    simulated_task_var = np.zeros(n_simulations)
    simulated_reliability = np.zeros((n_simulations, len(n_trials_list)))

    ###
    # run the simulation
    for s in range(n_simulations):
        # generate matrix of trials, total_n_trials per participants
        tmp = bin_samples_rand4(rng, N, total_n_trials, mu_subjects, ravel=False)
        all_trials = tmp.ravel()

        # save the group mean and variance
        simulated_task_mu[s] = tmp.mean(axis=1).mean()
        simulated_task_var[s] = tmp.mean(axis=1).var()

        #### Calculate reliability
        # define corr array
        array_corr_trials_psychofit = np.zeros((len(n_trials_list), n_repeats))

        for j, n_trials in enumerate(n_trials_list):

            # check that it's possible
            assert n_trials <= total_n_trials // 2

            # go over iterations
            for i in range(n_repeats):
                arr_first = None
                arr_second = None

                # split the data into two halves -- it works the same for arrays as for df
                arr_first, arr_second = split_dataframes_faster_chunks(all_trials, total_n_trials, n_trials, N)

                # calculate correlation
                array_corr_trials_psychofit[j, i] = np.corrcoef(arr_first.reshape(N, -1).mean(axis=1), arr_second.reshape(N, -1).mean(axis=1))[0, 1]

        # save the reliability
        simulated_reliability[s, :] = array_corr_trials_psychofit.mean(axis=1)  #np.nanmean(array_corr_trials_psychofit, axis=1)

        ### Do the fit
        # go over all the functions
        for (k, fx) in enumerate(fit_functions):

            result = None
            gmodel = None

            # initiate the model
            gmodel = Model(fx, nan_policy='omit')

            # define variables
            if gmodel.name == 'Model(hyperbolic_fit)':
                x = n_trials_list
                y = simulated_reliability[s]
            else:
                # deal with undefined values
                simulated_reliability[s, simulated_reliability[s] == 0] = np.nan
                # skip if fit is not possible
                if np.isnan(simulated_reliability).all():
                    simulated_BA_fit_lin[s] = np.array([np.nan, np.nan])
                    continue

                else:
                    x = 1 / n_trials_list
                    y = 1 / simulated_reliability[s]

            # set params, it differs for different functions
            if len(gmodel.param_names) == 1:
                params = gmodel.make_params(a=3)
            #             print(f"For model {gmodel.name} taking 1 param.")

            elif len(gmodel.param_names) == 2:
                if 'cdf_lognormal' in gmodel.name or 'cdf_normal' in gmodel.name:
                    params = gmodel.make_params(mu=0.3, sigma=0.3)
                else:
                    params = gmodel.make_params(a=3, b=1)
            #             print(f"For model {gmodel.name} taking 2 params.")

            # perform the fit
            try:
                result = gmodel.fit(y, params, x=x)
                # check if we have b
                if 'b' in result.best_values.keys():
                    simulated_BA_fit_lin[s] = np.array([result.best_values["a"], result.best_values["b"]])
                else:
                    simulated_BA_fit_A[s] = result.best_values["a"]

            except ValueError:
                print(f'failed to fit {gmodel.name}')
                logs.append(f'failed to fit {gmodel.name}')

            except TypeError as e:
                print(f'There was an error {e}, failed to fit {gmodel.name}\nParameters were:\nx={x}\ny={y}\nn trials list={n_trials_list}\nntrials={n_trials}\ns={s}\nmean {mu_subjects_mu} and std {mu_subjects_std} for {distribution} distribution.')
                logs.append(f'There was an error {e}, failed to fit {gmodel.name}\nParameters were:\nx={x}\ny={y}\nn trials list={n_trials_list}\nntrials={n_trials}\ns={s}\nmean {mu_subjects_mu} and std {mu_subjects_std} for {distribution} distribution.')

        ## get analytical
        simulated_BA_teor[s] = analytical_C(simulated_task_mu[s], simulated_task_var[s])

    end = time.time()
    # print(f"Process took: {end - start:.2f} s which is {(end - start) / 60:.2f} min.")
    logs.append(f"Process took: {end - start:.2f} s which is {(end - start) / 60:.2f} min.\n")
    logs.append('\n\n-----------------------------------\n')

    # save the logs
    write_log(os.path.join(path_output, log_name), logs)

    if distribution == 'lognormal':
        # calculate lognormal mean and variance
        m = np.exp(mu_subjects_mu + mu_subjects_std**2 / 2.0)
        v = np.exp(2 * mu_subjects_mu + mu_subjects_std**2) * (np.exp(mu_subjects_std**2) - 1)
        expected_BA = analytical_C(m, v)
    elif distribution == 'gaussian':
        m = np.nan
        v = np.nan
        expected_BA = analytical_C(mu_subjects_mu, mu_subjects_std ** 2)
    elif distribution == 'beta':
        m = mu_subjects_mu/(mu_subjects_mu+mu_subjects_std)  # note that for beta distribution, the coefficients are labelled as alpha=mu_subjects_mu and beta=mu_subjects_std
        v = mu_subjects_mu*mu_subjects_std / ((mu_subjects_mu+mu_subjects_std+1) * (mu_subjects_mu+mu_subjects_std)**2)
        expected_BA = analytical_C(m, v)
    else:
        m, v, expected_BA = None, None, None

    # save
    if save:
        pd.DataFrame({
                       'hyperbolic_fit': simulated_BA_fit_A,
                       'fit_linear_slope': simulated_BA_fit_lin[:, 0],
                       'fit_linear_intercept': simulated_BA_fit_lin[:, 1],
                       'fit_theoretical': simulated_BA_teor,
                       'distribution': distribution,
                       'true_BA': true_BA,  # C coefficient
                       'expected_BA': expected_BA,
                       'mu': mu_subjects_mu,
                       'sigma': mu_subjects_std,
                       'mu_lognorm': m,
                       'var_lognorm': v,
                       'N': N,
                   }).to_csv(os.path.join(path_output, f'BA_coefficients_{distribution}_{n_simulations}sim_{N}_{mu_subjects_mu}_{mu_subjects_std}.csv'), index=False)

    return simulated_BA_fit_A, simulated_BA_fit_lin, simulated_BA_teor, simulated_task_mu, simulated_task_var, simulated_reliability


def get_subject_means(mu_subjects_mu, mu_subjects_std, N, distribution, rng=np.random.default_rng(0)):
    """
    Generate means for N subjects to fit a given distribution with mean of mu_subjects_mu and standard deviation of
    mu_subjects_std.

    Parameters
    ----------
    mu_subjects_mu: float, mean of distribution that we want to create; parameter alpha if the distribution is beta
    mu_subjects_std: float, standard of distribution that we want to create; parameter beta if the distribution is beta
    N: int, how big, how many elements, for how many subjects the distribution is
    distribution: str, either gaussian or lognormal or beta, determines the shape of this distribution
    rng: default state

    Returns
    -------
    Array of N means

    """
    # get the means
    if distribution == 'gaussian':
        # generate probabilities, the probabilities are Gaussian and clipped
        mu_subjects = rng.normal(mu_subjects_mu, mu_subjects_std, N)

    elif distribution == 'lognormal':
        # generate probabilities, the probabilities are lognormal and clipped, 1- is just to resemble experiments
        mu_subjects = 1 - rng.lognormal(mu_subjects_mu, mu_subjects_std, N)

    elif distribution == 'beta':
        # check that the parameters are ok
        assert mu_subjects_mu > 0
        assert mu_subjects_std > 0
        # generate probabilities, the probabilities are beta and clipped although that should be by definition
        mu_subjects = rng.beta(mu_subjects_mu, mu_subjects_std, N)

    else:
        raise ValueError(f'Distribution {distribution} is not defined, quitting.')
        exit()

    # make sure that the accuracy/probability is not greater than 1 or less than 0
    mu_subjects = np.clip(mu_subjects, 0, 1)

    return mu_subjects


def write_log(name, text):
    """
    Helper function for writing logs to file
    Parameters
    ----------
    name: str, name of the file into which we want to save
    text: str or list of str, what to write

    Returns
    -------

    """
    with open(name, "w") as f:
        # check what type we have
        if isinstance(text, str):
            f.write(text)
        elif isinstance(text, list):
            f.writelines(text)
