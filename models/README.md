# Trained AI detector model

`ai_detector.joblib` is the trained DocVerify TF-IDF + Logistic Regression bundle. It contains `pipeline` and `metadata`; class 0 is human-written and class 1 is AI-generated. The app expects scikit-learn 1.6.1. Only load model artifacts from sources you trust.
