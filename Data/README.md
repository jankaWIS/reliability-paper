# Data

## General notes
Because of the high volume of all the raw data, the raw data is not part of this repository and is stored at 10.5281/zenodo.8117652. The codes and scripts expect it to be in this directory in `raw_CSV` folder, so please if you want to run them, download the raw data separately and put them to this folder.

Organisation:
```
├── cleaned_CSV           <- Cleaned and processed data for all tasks.
├── demographic_data      <- Demographics for all subjects with at least one task (as provided by Prolific).
├── raw_CSV               <- Raw data for all tasks.
├── results
    ├── curve_fits                      <- Contains reliability curves data for each task.
        ├── compare_fitting_methods     <- Contains simulation data for different fit methods.
    ├── reliability_meaning             <- Data for Fig. 3.
    ├── reliability_split_halves        <- Contains split halves reliability data for each task.
    
```
The folder also contains a summary file of all the reliabilities and other statistics per task in `reliability_summary.csv`, table with how many trials are needed to achieve given reliability thresholds (`needed_trials_thresholds.csv`), and number of trials and subjects per task (`task-num_trials{}.csv`).

## Clean data
Clean data contain only information about each and single trial, ie., it contains `userID`, `RT`, `answer` (correct/incorrect), `stage` or `level` if relevant. It is the minimal data that is required to run all the analysis successfully. 

The clean data can be reconstructed by running the `Process_raw_data.ipynb` notebook. See the instructions there -- each task has its own section and that has (is intended) to be run separately. All the relevant steps in data cleaning are there.

## Demographic data
This folder contains a minimal set of demographics provided by Prolific of all the users that took part in at least some of the experiments. Those are without any personal identifiers and contain:
* userID
* Age	
* Sex	
* Ethnicity	simplified
* Nationality	
* Country of birth	
* Country of residence	
* Employment status	
* Language
