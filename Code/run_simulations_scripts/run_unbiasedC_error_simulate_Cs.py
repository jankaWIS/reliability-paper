import os, sys
import time
import numpy as np
import itertools
import multiprocessing
from functools import partial

# Define path
path_results = "../../Data/results"
path_simulation_general = os.path.join(path_results, "curve_fits", "compare_fitting_methods")
path_output = os.path.join(path_simulation_general, "search_Nxntrials_space")
functions_modul = "../functions"


# Importing module for functions
sys.path.insert(0, functions_modul)

# from reliability_functions import analytical_C, analytical_C_unbiased, bin_samples_rand4
from simulation_functions import get_subject_means, run_simulation_unbiasedC_space_explosions
from reliability_functions import analytical_C


# 28 s per 10
def main():
    # start the timer
    start = time.time()

    # number of simulations
    n_simulations = 10*10**2
    
    # define if to run with multiprocess
    run_multiprocess = True

    # number of N
    Ns_array = np.arange(10, 210, 10)

    # number of n trials
    n_trials_array = np.arange(10, 260, 10)

    # define total number of subjects and trials, this will be 1.5x more as the max we sample to
    total_N = 10 * Ns_array.max() // 2
    total_n_trials = 10 * n_trials_array.max() // 2

    # define how big N we want to have to estimate the true C
    unconstrained_N = 10**7
    print(f"Running {n_simulations} simulations, unconstrained dataset size is {unconstrained_N}, going over N from {Ns_array.min()}-{Ns_array.max()} and n trials from {n_trials_array.min()}-{n_trials_array.max()}\nRunning MP:{run_multiprocess}.\n")

    # define C range
    C_max = 100
    C_min = 1

    # how close should the denominator be to 0 to count it as an explosion and discard
    atol = 1*10**-3

    ### create space of all of the possible distributions
    all_param_combinations = []

    # simulate both distributions
    # for distribution in ["gaussian", "lognormal"]:
    for distribution in ["beta"]:
        print(f'Going over {distribution}...\n')
        if distribution == "beta":
             # define the beta
             all_param_combinations += [('beta', alpha, beta) for alpha, beta in
                                       zip(np.arange(1, 40.5, 0.5), np.arange(1, 40.5, 0.5))]

        else:

            if distribution == "gaussian":
                # define the gaussian - this was updated
                mu_subjects_mu_list = [0.3, 0.4, 0.5, 0.7, 0.8]
                mu_subjects_std_list = [0.08, 0.1, 0.13, 0.15, 0.18, 0.2]

            elif distribution == "lognormal":
                # define the lognormal
                mu_subjects_mu_list = [-0.9, -1.1, -1.3, -1.6, -1.8, -2]
                mu_subjects_std_list = [0.2, 0.3, 0.4, 0.5, 0.7]

            # get all their combinations and add distribution as a label
            all_param_combinations += [(distribution,) + x for x in
                                       list(itertools.product(mu_subjects_mu_list, mu_subjects_std_list))]

    print(f'Number of all combinations before cleaning for C limits: {len(all_param_combinations)}')

    # make sure that all are in the C range, if not, drop them
    all_param_combinations_clean = []

    # iterate over all the combinations of mean and std per distribution
    for i, (distribution, mu_subjects_mu, mu_subjects_std) in enumerate(all_param_combinations):

        # get the means
        if distribution == 'gaussian':
            # get the real C
            expected_C = analytical_C(mu_subjects_mu, mu_subjects_std ** 2)

        elif distribution == 'lognormal':
            # calculate lognormal mean and variance
            m = np.exp(mu_subjects_mu + mu_subjects_std ** 2 / 2.0)
            v = np.exp(2 * mu_subjects_mu + mu_subjects_std ** 2) * (np.exp(mu_subjects_std ** 2) - 1)
            expected_C = analytical_C(m, v)

        elif distribution == 'beta':
            # note that for beta distribution, the coefficients are labelled as alpha=mu_subjects_mu and beta=mu_subjects_std
            expected_C = mu_subjects_mu + mu_subjects_std  # analytical_C(m, v) --> See the derivation

        if C_min <= expected_C <= C_max:
            all_param_combinations_clean.append((expected_C, distribution, mu_subjects_mu, mu_subjects_std))
        else:
            print(f'Removing C = {expected_C}')

    print(f'Number of all combinations after cleaning for C limits: {len(all_param_combinations_clean)}')
    print(f'The combinations are:\n{*all_param_combinations_clean,}')


    if run_multiprocess:
        # split processes and map them into the function, define the number of processors
        num_cpu = os.cpu_count()  # max(4, os.cpu_count()) # TODO decide
        print(f'\nThere is {os.cpu_count()} processors, I take {num_cpu}\n')
        pool = multiprocessing.Pool(num_cpu)

        # https://stackoverflow.com/questions/5442910/how-to-use-multiprocessing-pool-map-with-multiple-arguments
        pool.map(partial(run_simulation_unbiasedC_space_explosions, Ns_array=Ns_array, n_trials_array=n_trials_array,
                         n_simulations=n_simulations, unconstrained_N=unconstrained_N, total_N=total_N,
                         total_n_trials=total_n_trials, update_rng_per_simulation=True,
                         path_output=os.path.join(path_simulation_general, 'simulate_unbiasedC_error', 'multi'),
                         save_name=f'simulation_unbiasedC_error_explosions_{n_simulations}sim_{total_n_trials}trials_{total_N}N_resampling',
                         run_multiprocess=run_multiprocess, atol=atol,
                         ),
                 all_param_combinations_clean)
        pool.close()

    else:
        run_simulation_unbiasedC_space_explosions(all_param_combinations_clean=all_param_combinations_clean,
                                                  Ns_array=Ns_array, n_trials_array=n_trials_array, total_N=total_N,
                                                  total_n_trials=total_n_trials, n_simulations=n_simulations,
                                                  unconstrained_N=unconstrained_N, update_rng_per_simulation=True,
                                                  path_output=os.path.join(path_simulation_general,
                                                                           'simulate_unbiasedC_error'),
                                                  save_name=f'simulation_unbiasedC_error_explosions_{n_simulations}sim_{total_n_trials}trials_{total_N}N_resampling',
                                                  run_multiprocess=run_multiprocess, atol=atol,
                                                  )

    end = time.time()
    print(f"Entire run took: {end - start:.2f} s which is {(end - start) / 60:.2f} min.")


if __name__ == "__main__":
    main()