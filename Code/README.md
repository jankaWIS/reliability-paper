<!-- #region -->
# Code logic
All helper functions are stored in the `functions` folder. It contains in total 5 files:
   * `composite_measures_functions.py` -- contains functions to compute composite measures (PGNG PCIT/PCTT, d', ...)
   * `data_processing_functions.py` -- contains functions used in preprocessing and cleaning the data
   * `plot_functions.py` -- contains functions used for plotting the data
   * `reliability_functions.py` -- contains function related to reliability including fit functions and some simulation functions
   * `simulation_functions.py` -- contains functions used for simulating the error in $C$.

There is a special set of files that were used to run the simulations using multiprocessing in python on our clusters. The important functions and actuall calculations are in `simulation_functions.py` but the run scripts are the following saved in `run_simulations_scripts`:
* `run_unbiasedC_error_simulate_Cs.py` -- run only MV fit across several $N$s and $L$s to create error matrices. This code is written both to be run in parallel across $C$s as well in a linear fashion. Implemented are also other distributions that are not used in the paper.
* `compare_unbiased_C_beta_multiprocess.py` -- run all four fits (direct, linearised, MV, naive MV) for all the 79 $C$s for $L=250$ and across several $N$s.
* `unbiased_C_beta_ratios_multiprocess.py` -- this runs a simulation of the different $\alpha/\beta$ ratios and their effect on error in $C$.


Besides these function, there are several Jupyter notebooks that detail or analyse specific parts. These contains:
* `Process_raw_data.ipynb` -- this notebook goes over all used tasks and does preprocessing, exclusions and cleaning.
* `Reliability_split_halves.ipynb` -- this ntb introduces split halves calculations, it's lenghty and not useful in itself (lots of code repetitions) but it does produce the `reliability_summary.csv` file that documents reliability per task per form.
* `Reliability-data4curves.ipynb` -- this ntb goes over all tasks in their fullest (all forms concatenated) and creates reliability curves that are saved as `csv` files and used in the paper Figures (eg. Fig. 1 and SI Fig. 1).
* `Reliability-meaning-largeL.ipynb` -- this ntb looks at the effect of reliability on correlation (Fig. 3).
* `process_save_error_estimates_analytic_C.ipynb` -- processes and concatenates all simulation data for estimating error in C. Produces several `csv` files that are necessary for Fig. 4 and Fig. 5, namely:
    * `all_simulations_varyN_ntrials_250_C_between_1-100.csv` that concatenates all the csv files that come from the simulations for all the combinations of C, all fitting methods, all N and simulations.
    * 4 files that are `df_perc_{mean/median}_{noexplosions_}varyN_ntrials_250_unbiasedC_between_1-100.csv` where it is always percent error (either mean or median, specified in the name) and SD of the percent error for all the different methods. The other separation is whether simulations which were marked as "exploded", i.e., the denominator in MV fit was close to zeto, were counted or discarded (`_noexplosions`).
    * The correction matrices implemented in the web app named `unbiasedC_search_N_200xntrials_250_all_arrays_nsim1000_median_3C_selection_test4web.npz`.

* `estimate_shrinking_error.ipynb` -- notebook to reproduce SI Fig 3.
* `statistical_dependence_tests.ipynb` -- verifies statistical dependence or independence of several measures that are claimed in the paper.
---
<!-- #endregion -->
