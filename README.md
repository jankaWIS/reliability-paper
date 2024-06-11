<!-- #region -->
# *A measure of reliability convergence to select and optimize cognitive tasks for individual differences research*: Codes and data to the article, FAQ about the web tool



### Citation


---
#### Associated bioarxiv paper
*Putting cognitive tasks on trial: A measure of reliability convergence*

Jan Kadlec, Catherine Walsh, Uri Sadé, Ariel Amir, Jesse Rissman, Michal Ramot, bioRxiv 2023.07.03.547563; doi: [https://doi.org/10.1101/2023.07.03.547563](https://doi.org/10.1101/2023.07.03.547563).

---
#### Associated ZENODO repository


[![DOI](https://zenodo.org/badge/658647441.svg)](https://zenodo.org/doi/10.5281/zenodo.11564064)



## Author


Jan Kadlec (https://jankawis.github.io/)

## Repository organisation

The organization of this project is as follows:

```
├── Code                     <- All code and functions used in the manuscript.
├── Data                     <- Not part of this repository, saved at OSF (10.17605/OSF.IO/CRE2B) but the code assumes this folder.
├── Figures                  <- Code to generate the figures and the figures.
```

Some folders have their own README files that contain all relevant information necessary to reproduce the findings.

## Data
There are no data in this repository for size limitations and are stored at [10.17605/OSF.IO/CRE2B](10.17605/OSF.IO/CRE2B). For more details, see README there.

---

# FAQ about the online tool
## How to use this app
Detailed instructions are provided in the SI of the original article. To use the online web application ([https://jankawis.github.io/reliability-web-app/](https://jankawis.github.io/reliability-web-app/)), follow this protocol:
1. Run your pilot experiment with a small number of participants and trials per participant (we recommend at least $N=30$, though preferably $N\sim 50$ participants, and $L\geq 30$ trials per participant). 
1. Using a smaller number of trials or participants is possible but will result in larger confidence intervals.
1. Calculate the mean score for each participant across trials.
1. Calculate the mean ($E[Z]$) and sample variance (Var($Z$)) of those scores across the group of scores. Sample variance is the default in many statistical tools (pandas, R, Matlab) except for numpy where `ddof=1` has to be explicitly used.
1. Enter the mean and variance computed in step 3., and the number of participants and trials per participant into the online tool and plot the corresponding reliability curve.

Toggle the reliability $R$ to see how many trials $L$ you need to reach your desired reliability level. You can also optionally provide the time it took to collect these L trials and plot reliability versus the time it would take to collect the necessary number of trials.

In some cases, for which the calculated C is very high, corresponding to very low variance (potentially as a result of floor or ceiling effects), the app will provide a warning. In such cases, the use of that measure for studying individual differences should be reconsidered. 

A warning will also be issued if the error of the fit cannot be properly estimated for very small combinations of $L$ and $N$. 

Note that a further **limitation** is that the MV fit implemented in the app requires measures to be sampled from a Bernoulli distribution (i.e., each trial can yield only two possible outcomes). For any other measures which do not obey these assumptions, we suggest applying the directly fitted or linearized fits (not through the app). See [*What are the limitations of this app? Can I use it for any task?*](What-are-the-limitations-of-this-app?-Can-I-use-it-for-any-task?) section of the FAQ.

## What are the limitations of this app? Can I use it for any task?
Unfortunately not. This web app uses a method that is limited to binary tasks, meaning tasks in which the outcome of any given trial is binary and there are only two possible outcomes (0/1, correct/incorrect, blue/red, etc.). Multiple choice tasks with more than two alternatives can be used as long as the outcome of the trial is binary (e.g., tasks with 4 options, 1 correct and 3 incorrect). Examples where the online tool **cannot** be directly used are any difference measures (difference of scores in trials of category “A” and category “B”, e.g. before and after some intervention, or of different difficulties), composite measures such as d’, Cowan’s k, beta, Likert scale scores, etc.

However, if your task is NOT binary, you **can** still compute the $C$ coefficient and create the reliability curve using the other two methods – direct and linearized fits. Please consult our paper to find out more.

## What to do with missing data per subject?
This question is complex and task-dependent. Our current recommendation is to follow the same protocol as we adopted in the current paper – discard participants who have too many missing data points, and utilize functions such as `nanmean` to calculate summary statistics by ignoring missing data. It is also possible to downsample the number of trials per participant to the minimum number of trials available for all participants. If this significantly changes the mean and variance of the distribution, perhaps the inclusion of participants with missing data should be reconsidered.

## Which variance should I use? Sample or population?
You should use sample variance of participants' scores. This means calculating the scores for all participants and then determining the variance of this set/distribution of scores. It is the default in pandas, R, and Matlab. If you use numpy, you need to manually set ddof=1 (see [the official documentation](https://numpy.org/doc/stable/reference/generated/numpy.var.html)). If you are an Excel user, please use VAR.S (or VARA, [see the documentation](https://support.microsoft.com/en-gb/office/var-s-function-913633de-136b-449d-813e-65a00b2b990b)).

## What is the "Time to collect the trials" good for?
The optional "Time to collect the trials"  performs a simple calculation of the time that it will take to collect data to achieve the desired reliability. It is to give you a sense of the practical implications for collecting a given number of trials and help inform decisions about the reliability vs time to collect data trade-off. If you provide the necessary information, you can toggle the "Plot time" button and see the reliability with experiment time rather than the number of trials. 

## What are the upper and lower limits (P + 1SD and P - 1SD) and which one should I use? (aka How to read the reliability plot)
**Short answer:** Move along the x-axis to explore the confidence interval. The lower limit is the minimum number of trials for a desired reliability (purple, P - 1SD), while the upper limit (red, P + 1SD) ensures you shouldn’t need more trials than specified to achieve that reliability level.

**Long answer:** The upper and lower limits in the reliability plot show the confidence margin, or the error, in estimating the C coefficient based on the user-provided parameters. It is a range indicating the mean error plus or minus the mean standard deviation (SD) for the given combination of L and N that were determined through simulations (refer to Figure 5 in the paper).

In cases where the SD is greater than 100%, a warning is presented to the user and SD of 100% is used.

To understand the "lower" and "upper" bounds: focus on the reliability (y-axis) of the plot and follow a line parallel with the x-axis. Intercepts with the curves will tell you the lower and upper limits. 

**Lower Limit (P - 1SD):** This tells you the minimum number of trials needed to achieve a specific reliability. Less than this number of trials will NOT reach the desired reliability level. 

**Upper Limit (P + 1SD):** This indicates that you won't need more trials than this number to achieve the desired level of reliability. It ensures you're on the safe side.

## What do the numbers mean when hovering over the plot?
It is the $(x,y)$ coordinate of the point, so the first says the number of trials ($L$) and the second is reliability ($R$) corresponding to this number of trials.

## The app does not allow me to enter a mean that is less than zero or more than 1. What should I do?
If your mean is less than 0 or more than 1, it very likely means that the outcomes of trials in your experiment or the measure you use are not binary (see section [*What are the limitations of this app? Can I use it for any task?*](What-are-the-limitations-of-this-app?-Can-I-use-it-for-any-task?)) and this tool **cannot** be used for your data. If, however, your task is binary, this may simply mean that you need to scale your data appropriately (for instance – a participant might have a score of 0.8 instead of 80% correct, or 40/50 trials correct). If your task is not binary, you can still create the reliability curve yourself, calculate C and obtain reliability predictions based on the direct and linearized fits. Please see the main text for details on how to compute these curves. 

<!-- #endregion -->
