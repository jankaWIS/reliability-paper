import os, sys
import time
import multiprocessing
import numpy as np
import pandas as pd
from functools import partial


# Define path
path_results = "../../Data/results"
path_simulation_general = os.path.join(path_results, "curve_fits", "compare_fitting_methods")
path_output = os.path.join(path_simulation_general, "test_beta_ratio_noexplosions")
functions_modul = "../functions"

# Importing module for functions
sys.path.insert(0, functions_modul)

from reliability_functions import analytical_C, analytical_C_unbiased

from simulation_functions import write_log, get_subject_means, run_estimate_C_with_unbiased_fit_theory_only_explosions

# runs 20.5 min with 10^4 sim and N from 10-250
def main():
    # define main logs
    main_logs = []

    # max number of trials
    total_n_trials = 250

    # define if to run with multiprocess
    run_multiprocess = True

    # log where we run
    main_logs.append(f'Running in {sys.prefix} \nRunning multiprocess: {run_multiprocess}')

    # how many simulations do we want to run
    n_simulations = 10**3
    print(f"Running {n_simulations} simulations.\n")
    main_logs.append(f"Running {n_simulations} simulations.\n")

    # define how many subjects we want
    N_list = np.array([50])#np.arange(10, 260, 10)#np.arange(80, 100, 10)#
    main_logs.append(f"\nGoing over {*N_list,} parameters per task, ie. {len(N_list)} conditions.\n")

    # define how big N we want to have to estimate the true C
    unconstrained_N = 10 ** 7
    main_logs.append(f"\nSize of the unconstrained dataset is {unconstrained_N}.\n")

    # define C range
    C_max = 60
    C_min = 1

    # how close should the denominator be to 0 to count it as an explosion and discard
    atol = 1 * 10 ** -3

    # start the timer
    start = time.time()

    all_param_combinations = []
    # simulate distributions
    for distribution in ["beta"]:
        print(f'Going over {distribution}...\n')
        if distribution == "beta":
            # define the beta
            target_C = [2, 5, 10, 20, 40]
            ratios = [0.25, 0.5, 0.8, 1.0, 1.25, 1.5, 2.0, 2.5, 3.0, 4.0, 5.0, 7.5, 10.0, 15.0, 20.0, 30.0]

            df_Cs = pd.DataFrame()

            for C in target_C:
                for r in ratios:
                    beta = C / (r + 1)
                    alpha = r * beta

                    df_Cs = pd.concat([df_Cs, pd.DataFrame({
                        "C": [C],
                        "a2b_ratio": [r],
                        "a": [alpha],
                        "b": [beta],
                    })])

                    all_param_combinations += [(distribution, alpha, beta)]
        else:
            raise ValueError(f'The following distribution is not defined: {distribution}.')

        print(f'Number of all combinations before cleaning for C limits: {len(all_param_combinations)}')

        # make sure that all are in the C range, if not, drop them
        parameter_combinations = []

        # iterate over all the combinations of mean and std per distribution
        for i, (distribution, mu_subjects_mu, mu_subjects_std) in enumerate(all_param_combinations):

            # get the means
            if distribution == 'gaussian':
                # get the real C
                true_C = analytical_C(mu_subjects_mu, mu_subjects_std ** 2)

            elif distribution == 'lognormal':
                # calculate lognormal mean and variance
                m = np.exp(mu_subjects_mu + mu_subjects_std ** 2 / 2.0)
                v = np.exp(2 * mu_subjects_mu + mu_subjects_std ** 2) * (np.exp(mu_subjects_std ** 2) - 1)
                true_C = analytical_C(m, v)

            elif distribution == 'beta':
                # note that for beta distribution, the coefficients are labelled as alpha=mu_subjects_mu and beta=mu_subjects_std
                # m = mu_subjects_mu / (mu_subjects_mu + mu_subjects_std)
                # v = mu_subjects_mu * mu_subjects_std / ((mu_subjects_mu + mu_subjects_std + 1) * (mu_subjects_mu + mu_subjects_std) ** 2)
                true_C = mu_subjects_mu + mu_subjects_std  # analytical_C(m, v) --> See the derivation

            if C_min <= true_C <= C_max:
                parameter_combinations.append((mu_subjects_mu, mu_subjects_std))
            else:
                print(f'Removing C = {true_C}')

        main_logs.append(
            f"\n---> Running {distribution} distribution.\nGoing over {*parameter_combinations,} parameters per task, ie. {len(parameter_combinations)} pairs.\n")
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
                num_cpu = os.cpu_count()#max(4, os.cpu_count()) # TODO decide
                main_logs.append(f'\nThere is {os.cpu_count()} processors, I take {num_cpu}\n')
                pool = multiprocessing.Pool(num_cpu)

                # https://stackoverflow.com/questions/5442910/how-to-use-multiprocessing-pool-map-with-multiple-arguments
                # pool.map(partial(run_estimate_C_with_unbiased_fit_theory_only, N=N, total_n_trials=total_n_trials,
                #                  n_simulations=n_simulations, distribution=distribution, update_rng_per_simulation=True,
                #                  unconstrained_N=unconstrained_N, path_output=path_output, vary="N"),
                #          parameter_combinations)
                pool.map(partial(run_estimate_C_with_unbiased_fit_theory_only_explosions, N=N, total_n_trials=total_n_trials,
                                 n_simulations=n_simulations, distribution=distribution, update_rng_per_simulation=True,
                                 unconstrained_N=unconstrained_N, path_output=path_output, vary="N", atol=atol,
                                 # rng=np.random.default_rng(0), rng_C=np.random.default_rng(0)
                                 ),
                         parameter_combinations)
                pool.close()

            else:
                # no multiprocess
                for p in parameter_combinations:
                    run_estimate_C_with_unbiased_fit_theory_only_explosions(p, N=N, total_n_trials=total_n_trials,
                                                                            n_simulations=n_simulations,
                                                                            distribution=distribution,
                                                                            unconstrained_N=unconstrained_N,
                                                                            update_rng_per_simulation=True,
                                                                            path_output=path_output, vary="N",
                                                                            # rng=np.random.default_rng(1000),
                                                                            # rng_C=np.random.default_rng(0),
                                                                            atol=atol)


    end = time.time()
    print(f"Process took: {end - start:.2f} s which is {(end - start) / 60:.2f} min.")
    main_logs.append(f"\nProcess took: {end - start:.2f} s which is {(end - start) / 60:.2f} min.\n")

    write_log(os.path.join(path_output, f'main_log-test_beta_ratios-ntrials{total_n_trials}_Nfrom{N_list.min()}-{N_list.max()}.txt'), main_logs)


if __name__ == "__main__":
    main()