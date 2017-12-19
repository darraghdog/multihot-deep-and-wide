# multihot-deep-and-wide

- Execute by running,  
`python tf_wide_v1.py`
- Line 1 thru 184 is the original FTRL code, which works for us, from https://www.kaggle.com/c/avazu-ctr-prediction/discussion/10927
- Line 184 thru 229 is our shot at Tensorflow LinearClassifier which did not work so far. We fit the data into a sparsetensor; and then form there into categorical_column, but it does not converge. 
