## Plotting & Data collect shown in report

- CIL_plotter draws the two plots of the report. They are already included in this directory (see pdfs).
- data_train.csv holds dataset, which the plotter uses for the data-distribution plot.
- The data for the other plot can be found in the probabilities_data directory. \
File names work as follows: D_W_propX.csv, with D being the depth, W being the width. X stands for either Ø, "_alt", or "_re". Names without suffix (Ø) holds data collecting with bagging (with original loss), "_re" refers to data collected without bagging (with original loss, called "denoising" in the .csv file), and the "_alt" suffix indicates alternate loss (called "standard" in the .csv file). The bagging data shows no loss_type columns (it was collected before the additional loss feature was implemented). \
See the report for details.
