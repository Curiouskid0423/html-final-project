# Final Project for ML

### Notes

#### Preprocessing :rocket:

- **Issue 1**: Many non-numeric features to be converted.
- **Issue 2**: Bad data entries to be removed / editted.
- **Issue 3**: How to engineer **meaningful** features, i.e. kind of like "transform" but not numeric. E.g. 國定假日、國家. 

#### Training :rocket:

##### Baseline model (Try the following):

- **(Naive) linear regression**. One model.
Use this as initialization.（懷兟）
- **Linear Regression with Transform**: Fit the features to a k-degree polynomial of your choice.
3+ models.（聿騰）
- SVR: SVM with regression (I dont know if this work)

##### Novel attempts

Hyperparameter tuning open to discussion.

- Neural Net (3-layers)
- **Recurrent Neural Net**
Learn the basics then apply Tensorflow.
- RNN with LSTM.
- Boosting (worth attempting.)

#### Testing :rocket:

Validation proportion: 
- Training (80%) vs Validation (20%)


#### Notes :rocket: