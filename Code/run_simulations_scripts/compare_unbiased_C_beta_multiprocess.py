import os, sys
import time
import multiprocessing
import numpy as np
from functools import partial

# Define path
path_results = "../../Data/results"
path_output = os.path.join(path_results, "curve_fits", "compare_fitting_methods", "beta_simulate_N_ntrials250")
functions_modul = "../functions"

# Importing module for functions
sys.path.insert(0, functions_modul)

from simulation_functions import write_log, get_subject_means, run_estimate_C_with_unbiased_explosions


def main():
    # define main logs
    main_logs = []

    # max number of trials
    total_n_trials = 250

    # define if to run with multiprocess
    run_multiprocess = True

    # log where we run
    main_logs.append(f'Running in {sys.prefix} \nRunning multiprocess: {run_multiprocess}')

    # define repeats for reliability sampling -- how many times to calculate the reliability
    n_repeats = 5*10**2
    step = None

    # define step
    if step is None:
        # have smaller steps for less trials
        if total_n_trials <= 100:
            step = 2
        else:
            step = 5

    n_trials_list = np.arange(step, (total_n_trials + step) // 2, step)
    print(f"Going over {*n_trials_list,} trials per task, ie. {len(n_trials_list)} items")
    main_logs.append(f"\nGoing over {*n_trials_list,} trials per task, ie. {len(n_trials_list)} items.\n")

    # how many simulations do we want to run
    n_simulations = 10**3
    print(f"Running {n_simulations} simulations and sampling {n_repeats} times when calculating reliability.\n")
    main_logs.append(f"Running {n_simulations} simulations and sampling {n_repeats} times when calculating reliability.\n")

    # define how many subjects we want
    N_list = np.arange(10, 210, 10)
    main_logs.append(f"\nGoing over {*N_list,} parameters per task, ie. {len(N_list)} conditions.\n")

    # define how big N we want to have to estimate the true C
    unconstrained_N = 10**7
    main_logs.append(f"\nSize of the unconstrained dataset is {unconstrained_N}.\n")

    # define C range
    C_max = 100
    C_min = 1

    # how close should the denominator be to 0 to count it as an explosion and discard
    atol = 1 * 10**-3

    # start the timer
    start = time.time()

    all_param_combinations = []
    # simulate distributions
    for distribution in ["beta"]:
        print(f'Going over {distribution}...\n')
        if distribution == "beta":
             # define the beta
             all_param_combinations = [('beta', alpha, beta) for alpha, beta in
                                       zip(np.arange(1, 40.5, 0.5), np.arange(1, 40.5, 0.5))]


        print(f'Number of all combinations before cleaning for C limits: {len(all_param_combinations)}')

        # make sure that all are in the C range, if not, drop them
        parameter_combinations = []

        # iterate over all the combinations of mean and std per distribution
        for i, (distribution, mu_subjects_mu, mu_subjects_std) in enumerate(all_param_combinations):

            # get the means
            if distribution == 'beta':
                # note that for beta distribution, the coefficients are labelled as alpha=mu_subjects_mu and beta=mu_subjects_std
                expected_C = mu_subjects_mu + mu_subjects_std  # analytical_C(m, v) --> See the derivation

            if C_min <= expected_C <= C_max:
                parameter_combinations.append((mu_subjects_mu, mu_subjects_std))
            else:
                print(f'Removing C = {expected_C}')

        main_logs.append(f"\n---> Running {distribution} distribution.\nGoing over {*parameter_combinations,} parameters per task, ie. {len(parameter_combinations)} pairs.\n")
        print(f'Number of all combinations after cleaning for C limits: {len(parameter_combinations)}')


        # Go over all combinations
        for N in N_list:
            # make a dir for each N
            directory = os.path.join(path_output, str(N))
            if not os.path.exists(directory):
                os.mkdir(directory)
            main_logs.append(f'\nCreated {directory} folder\n')

            print(f'\n---> Simulating {total_n_trials} trials for {N} participants\n')
            main_logs.append(f'\n---> Simulating {total_n_trials} trials for {N} participants\n')

            if run_multiprocess:
                # split processes and map them into the function, define the number of processors
                num_cpu = os.cpu_count()#max(4, os.cpu_count())
                main_logs.append(f'\nThere is {os.cpu_count()} processors, I take {num_cpu}\n')
                pool = multiprocessing.Pool(num_cpu)

                # https://stackoverflow.com/questions/5442910/how-to-use-multiprocessing-pool-map-with-multiple-arguments
                pool.map(partial(run_estimate_C_with_unbiased_explosions, N=N, total_n_trials=total_n_trials,
                                 n_trials_list=n_trials_list, n_simulations=n_simulations, n_repeats=n_repeats,
                                 distribution=distribution, update_rng_per_simulation=True,
                                 unconstrained_N=unconstrained_N, path_output=path_output, vary="N", atol=atol,
                                 # rng=np.random.default_rng(0), rng_C=np.random.default_rng(0)
                                 ),
                         parameter_combinations)
                pool.close()

            else:
                # no multiprocess
                for p in parameter_combinations:
                    run_estimate_C_with_unbiased_explosions(p, N=N, total_n_trials=total_n_trials,
                                                            n_trials_list=n_trials_list, n_simulations=n_simulations,
                                                            n_repeats=n_repeats, distribution=distribution,
                                                            update_rng_per_simulation=True,
                                                            unconstrained_N=unconstrained_N,
                                                            path_output=path_output, vary="N", atol=atol,
                                                            # rng=np.random.default_rng(1000),
                                                            # rng_C=np.random.default_rng(0),
                                                            )


    end = time.time()
    print(f"Process took: {end - start:.2f} s which is {(end - start) / 60:.2f} min.")
    main_logs.append(f"\nProcess took: {end - start:.2f} s which is {(end - start) / 60:.2f} min.\n")

    write_log(os.path.join(path_output, f'main_log-beta-ntrials{total_n_trials}_Nfrom{N_list.min()}-{N_list.max()}.txt'), main_logs)


if __name__ == "__main__":
    main()