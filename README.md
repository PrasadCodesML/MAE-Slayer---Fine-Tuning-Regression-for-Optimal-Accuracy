# All-Out-on-Regression
In this dataset i selected a complex regression dataset and tried to increase accuracy as much as possible at starting i use ANN and using `KerasTuner` on ANN, then i used Supervised ML using AutoML

There were 17 model tried using ANN and AutoML and the best model was of supervised ML using
H2o for AutoML and the best model was `StackedEnsemble`, ANN was close by

1. After importing the data i observed that there were too many categorical columns so i decided to keep those columns only which have less no. of category, but almost all of the columns were having proper no. of category.

2. After then i did label encoding on all the categorical columns

3. Then i performed some feature selection
    1. I used `VarianceThreshold` and `Pearson Correlation` for removing all the similar columns but i did not found it very promising so i dropped these techiques
    2. I used `mutual_info_selection` and `chi square test` to remove most important features and took a intersection of columns of both the techniques and there were 24 columns in common which were selected

4. I performed normalization on the dataset

5. Created a ANN architecture and custom tuned it's hyperparameters (Model 0)

6. Used Keras tuner for tuning the ANN architecture by 
    1. Selecting no of layers
    2. Selecting no of neurons per layer
    3. Selecting the optimizer
    4. All in one tuner

7. Used more techniques to decrease the overfitting
    1. Early Stopping
    2. Data scaling
    3. Regularization
    4. Weight Initialization
    5. Batch Normalization
    6. Dropout
