# Simulate reliability range
 This folder contains one notebook per each task/measure that randomly splits the data into "days" and, based on that, estimates the confidence interval of the expected (true test-retest) reliability that is used in Figure 4 for statistical significance testing. These simulations' results are saved to a dictionary in the `Data/results` folder in the pickle format under the name `simulated_reliability_range_largestL_nsim_1000.pkl`. 
 
 The range is obtained by running 1000 splits (simulations) and it is run on the real dataset (not using synthetic data) for the largest possible $L$. This $L$ is usually equal to half of the total number of trials in one form as we need to perform split-halves reliability calculation on non-overlapping sets of data. This limits the maximum $L$ for this analysis to half of the total number of trials. As we use data from two days (= from two forms), we can split them still randomly into two `pseudo-days` and still obtain one full form for each for the analysis.
 
 ## The calculation
 We use the attenuation correction formula first proposed by Spearman<sup>1,2,3</sup> in the following form:

$$ R_{x'y'} = R_{xy} \sqrt{R_{x'x'} \cdot R_{y'y'}},  $$

where $R_{x'y'}$ is the observed correlation/reliability between variables X and Y calculated as split-halves reliability of day 1 vs day 2;  and $R_{x'x'}$, resp. $R_{y'y'}$,  is the observed correlation/reliability of variable $X$, resp. $Y$ with itself, i.e. regular split-halves reliability that was calculated and used throughout the article.

### Calculation of split-halves reliability of day 1 vs day 2, i.e. $R_{x'y'}$
The general idea and approach is the same as for the split-halves reliability, with the difference that trials from different days never get mixed and always remain separate. We first sample $L$ trials per participant from day 1 and calculate the score/measure per participant, then we sample $L$ trials from day 2 per participant and calculate the score/measure, and then we correlate these scores from the two days across participants. We repeat this sampling 1000 times for each $L$, and the mean of these 1000 samples is $R_{x'y'}$.


## Literature
<sup>1</sup> Spearman, C. Correlation Calculated from Faulty Data. Br. J. Psychol. 1904-1920 3, 271–295 (1910).

<sup>2</sup> Spearman, C. The Proof and Measurement of Association between Two Things. Am. J. Psychol. 15, 72–101 (1904).

<sup>3</sup>  Spearman, C. Demonstration of Formulæ for True Measurement of Correlation. Am. J. Psychol. 18, 161–169 (1907).
