# Hyperparameter optimization

This the last step before we can used our newly trained model to classify peaks from other datasets. In this section, we will not only see how to use the automated method NeatMS provides to optimize the target parameter, but we will also explain the impact this parameter has on the classifier and how we can set it manually.

The last layer of the neural network uses a softmax activation function, which means that the output of the network is a probability distribution of 3 probabilities between 0 and 1, all of them adding up to 1 (3 is number of classes, we have one probability per class). The most simple approach would be to label the peak with the label corresponding to the highest probability returned by the model, but we can also investigate these probabilities and their impact on the prediction. We will particularly focus on the probability that corresponds to the `High quality` class and see how it impacts the prediction of the validation set (which remains untouched so far).

> You can follow the same approach using other classes than `High quality`. However, as `High quality` is the default class used in NeatMS, we will only cover this case here.

Calling the method `get_threshold()` will compute and return the optimal threshold using the validation set that you can then pass everytime you use this neural network. However, this threshold is optimal according to very specific criteria, keep reading to understand how it is computed and learn how to set it manually according to your own criteria.

Internaly, `get_threshold` call the method `get_true_vs_false_positive_df(label='High_quality')` which returns the following table:


| Probablity_threshold | True    | False    | False_low | False_noise |
|----------------------|---------|----------|-----------|-------------|
| 0.00                 | 1.0 | 1.0 | 1.0  | 1.0    |
| 0.01                 | 1.0 | 0.440 | 0.803  | 0.206    |
| ...                  | ...     | ...      | ...       | ...      |
| 0.99                 | 0.0 | 0.0 | 0.0  | 0.0    |

*Recall table*

Using this table, you can decide on the threshold value that you would like to use. For example, if we were to select a `0.01` threshold, 100% of `High_quality` peaks would be correctly predicted but we would have 44% of false positive (80% of `Low_quality`, and 20% of `Noise` peaks would be predicted as `High_quality`). This is obviously not a good threshold to choose in this case. 

The `get_threshold()` function returns the threshold that has the highest value when subtracting `False` to `True` positives. You can decide to be more conservative by choosing a lower threshold which will result in a higher true positive rate but also a higher false positive rate (higher sensitivity but lower specificty). A higher threshold will have the opposite effect, returning a lower sensitivity but higher specifictiy.

You can evaluate the general performance of your model using a ROC curve and calculating the area under the curve.

``` python
# Import the required libraries first
import numpy as np
from sklearn.metrics import auc
import pandas as pd

# Get probability dataframe
prob_df = nn_handler.get_true_vs_false_positive_df()

# Sort the dataframe to create the ROC curve
prob_df_roc = prob_df.sort_values(by=['Probablity_threshold'],ascending=False)

# Compute the area under the curve
auc(prob_df_roc['False'],prob_df_roc['True'])
```
If correctly trained, you should obtain an AUC higher than 95.0.

Finally, you can plot the ROC curve like so:

``` python
prob_df_roc.plot(x='False', y='True', figsize=(10,10), grid=True)
```


