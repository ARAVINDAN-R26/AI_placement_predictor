###  Advanced Modeling — Ensembles, Tuning, and Full ML Pipeline

## TASK 1
- The input train and test files from part 2 are loaded successfully
- Then Decision tree classifier classification is selected and implemented

**Reporting training and testing accuracy**
Default Decision Tree
Training accuracy: 0.9990
Test accuracy: 0.6917

**RESULT**
- The Decision Tree model shows clear signs of **overfitting**. 
- The training accuracy is very high (near to 100%), while the test accuracy is significantly lower. 
- This means the model has learned the training data very well, including its patterns and noise, but it does not generalize well to unseen data.

Decision Trees are considered **high-variance models** because they build the tree by choosing the best split at each step using a greedy approach. 
Once a split is made, the algorithm does not go back and reconsider earlier decisions. 
As a result, small changes in the training data can produce a very different tree. This makes the model more likely to fit the training data too closely, leading to overfitting and reduced performance on the test data.

## TASK 2
**Training and test accuracy for the controlled decision tree**
Controlled Decision Tree (max_depth=5, min_samples_split=20)
Training accuracy: 0.7828
Test accuracy: 0.7683

- The "max_depth" parameter limits how deep the Decision Tree can grow. By restricting the depth of the tree, the model becomes less complex and is less likely to memorize the training data. This helps reduce overfitting by lowering the model's variance, but it can still introduce a small amount of bias.

- The "min_samples_split" parameter sets the minimum number of samples needed to split a node. If a node has fewer samples than this value, it is not split further. This avoids creating splits based on very small groups of data, which are often caused by noise, and helps reduce overfitting.

- The Decision Tree that is controlled has a difference between the training accuracy and the test accuracy of the normal Decision Tree. The training accuracy of the controlled Decision Tree is a little lower than the training accuracy of the Decision Tree. However the test accuracy of the controlled Decision Tree is closer to the training accuracy of the controlled Decision Tree. This means that the controlled Decision Tree is less overfitted than the Decision Tree and the controlled Decision Tree performs better on data, than the normal Decision Tree.


## TASK 3
Gini tree test accuracy: 0.7683
Entropy tree test accuracy: 0.7683

The formula for **Gini Impurity** is

**Gini Impurity = 1 − Σ pi²**
where *pi*'s the proportion of samples that belong to each class in the **Gini Impurity** formula.

The formula for **Entropy** is

**Entropy = −Σ pi log₂(pi)**
where *pi* is the proportion of samples that belong to each class in the **Entropy** formula.

- A node with **Gini Impurity = 0** means all the samples in that node belong to one class.
- This is called a node because there is no mixing of different classes in the **Gini Impurity**.
- Decision Trees try to create nodes with **Gini Impurity** or **Entropy** values so that the classes are separated as much, as possible in the **Gini Impurity** and **Entropy** calculations.

## TASK 4
**Random Forest**
Training accuracy: 0.8830
Test accuracy: 0.7827
Random Forest test ROC-AUC: 0.8551

Top 5 Random Forest feature importances:
HSC_Marks            0.228630
AptitudeTestScore    0.184074
SSC_Marks            0.124213
Projects             0.107742
StudentID            0.089961

- The Random Forest method figures out how important each feature is by seeing how much it helps make the data more orderly when it is used to divide the data. 
- It does this for every division in every tree. Then takes the average value as the importance of the feature.
- A feature that has an importance value is really helpful for the model to make good predictions.

- The importance of a feature is not the same as a coefficient in a regression. The importance of a feature only tells us how useful it is for making predictions. It does not tell us if the feature has an bad effect.
- A coefficient in a regression tells us two things: it tells us the direction, which means if the effect is good or bad and it tells us how much the target value changes when the feature changes. The Random Forest feature importance and the linear regression coefficient are two things.
- Feature importance is about how useful a feature is, for making predictions and that is all.

- Random Forest uses a technique called bagging. Each tree is trained on a random sample of the training data selected with replacement, so some samples may appear more than once while others may not be selected. At each split, the tree also considers only a random subset of √(number of features) instead of all the features. Since every tree is different, their predictions are combined by averaging (or majority voting for classification).
- This reduces the variance of the model and makes it less likely to overfit compared to a single deep Decision Tree.

## TASK 4 (a)
Gradient Boosting
Training accuracy: 0.8023
Test accuracy: 0.7827
Gradient Boosting test ROC-AUC: 0.8611

## TASK 4 (b)
Five removed features: 'SoftSkillsRating', 'Workshops/Certifications', 'ExtracurricularActivities_yes', 'Internships', 'PlacementTraining_yes'
Full Random Forest test ROC-AUC: 0.8551
Reduced Random Forest test ROC-AUC: 0.8375

- The features that were removed were actually helping the model because the test ROC-AUC went down from **0.8551** to **0.8375** after they were taken out. This means that the removed features had some information that helped the model make better predictions.

- A simpler model with features is a good thing because it takes less time to make predictions and it is cheaper to maintain. We should only use a simpler model if the decrease in ROC-AUC is not too big and it is okay for what we are using the model for. In this case the model did not work well after the features were removed. So we have to balance how simple the model is and how well it makes predictions. The model and the features are important. The features that were removed were important, to the model.


## TASK 5
5-fold cross-validated ROC-AUC:

                   Model    CV_Mean_AUC   CV_Std_AUC
     Logistic Regression       0.8340      0.0101
Controlled Decision Tree       0.8291      0.0087
           Random Forest       0.8498      0.0055
       Gradient Boosting       0.8525      0.0062

Cross-validation is a way to see how well a model will work because it tries the model many times on different parts of the data. This helps because sometimes you get an bad split of the data just by chance. The model is. Tested many times on different splits of the data.
The final result is the average of all the tries. It is a better way to know how the model will do with new data. Cross-validation gives an idea of how the model will perform on unseen data, which is important.

## TASK 6
**Best Random Forest parameters:**
- randomforestclassifier__max_depth: 5
- randomforestclassifier__min_samples_leaf: 1
- randomforestclassifier__n_estimators: 200

Best cross-validated ROC-AUC: 0.8544

**Observations**
Total parameter combinations => 3×3×2=18
After 5-fold cross-validation: 18×5=90

- A total of **18** parameter combinations were tested. We used **5fold cross-validation** so our model was trained and evaluated **90 times** (18 times 5).
- Here is how Grid Search and Randomized Search work:
* Grid Search checks every combination of the given parameter values to find the best model. This gives us the result from the selected values. It takes more time because every combination is tested.
* Randomized Search only checks a set of parameter combinations. So it is faster. Requires less computation.. Sometimes it may not find the best combination because it does not test every possible value.


## TASK 7
**Manual learning curve:**
 Training fraction  Training AUC  Test AUC
            0.2000        0.8985    0.8526
            0.4000        0.8789    0.8575
            0.6000        0.8657    0.8581
            0.8000        0.8687    0.8592
            1.0000        0.8668    0.8587

* The training AUC usually goes down when the training set gets bigger. This makes sense because the model does not do well when it is trained on a lot of data it is less likely to make mistakes.
* The test AUC usually goes up when we use training data but it goes down a little bit at the end. This tells us that adding more training data helps the model do better with data that it has not seen before.
* Overall the model seems to have problems because it does not have training data not because the model is not good enough. The test AUC is still getting better as the training set gets bigger so getting data will probably make the model work better. The model will perform better with training data the training data is what the model needs to get better.

## Summary and comparison

## Summary and Comparison

| Model                    | 5-Fold CV Mean AUC | 5-Fold CV Std AUC | Test AUC |
| ------------------------ | -----------------: | ----------------: | -------: |
| Logistic Regression      |             0.8340 |            0.0101 |   0.8329 |
| Controlled Decision Tree |             0.8291 |            0.0087 |   0.7683 |
| Random Forest            |             0.8498 |            0.0055 |   0.8551 |
| Gradient Boosting        |             0.8525 |            0.0062 |   0.8611 |

**Recommended Model**

I would recommend the **Gradient Boosting** model. It achieved the highest test ROC-AUC (**0.8611**) and also the highest 5-fold cross-validation mean ROC-AUC (**0.8525**). The cross-validation standard deviation is also low, which shows that the model performs consistently across different data splits. Overall, it gives the best balance between accuracy and reliability, making it the best choice for this dataset.
