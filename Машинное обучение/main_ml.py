from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.model_selection import cross_val_predict
from cleanlab.classification import CleanLearning
from cleanlab.filter import find_label_issues
from catboost import CatBoostClassifier, Pool
import numpy as np
import pandas as pd
from pathlib import Path
import random
import sys

SEED = 123456

np.random.seed(SEED)
random.seed(SEED)


def main(csv_in: Path):
    df = pd.read_csv(csv_in, index_col=0)

    X_raw = df.drop(columns='target')  # features (pandas DataFrame)
    y_raw = df['target']  # labels (pandas Series)

    cat_features = X_raw.select_dtypes("category").columns
    X_encoded = pd.get_dummies(X_raw, columns=cat_features, drop_first=True)
    num_features = X_raw.select_dtypes("float64").columns
    scaler = StandardScaler()
    X_scaled = X_encoded.copy()
    X_scaled[num_features] = scaler.fit_transform(X_encoded[num_features])
    y = y_raw.map({"no": 0, "maybe": 1})

    # initialize data
    train_data = np.random.randint(0,
                                100, 
                                size=(100, 10))

    train_labels = np.random.randint(0,
                                    2,
                                    size=(100))

    test_data = catboost_pool = Pool(train_data, 
                                    train_labels)

    model = CatBoostClassifier(iterations=2,
                            depth=2,
                            learning_rate=1,
                            loss_function='Logloss',
                            verbose=True)
    # train the model
    model.fit(train_data, train_labels)
    # make the prediction using the resulting model
    preds_class = model.predict(test_data)
    preds_proba = model.predict_proba(test_data)
    print("class = ", preds_class)
    print("proba = ", preds_proba)
    \
    #chose the best solver
    
    # # https://scikit-learn.org/stable/modules/linear_model.html#logistic-regressio
     clf = CatBoostClassifier(solver='saga', max_iter=10000)

     num_crossval_folds = 3  # for efficiency; values like 5 or 10 will generally work better
     pred_probs = cross_val_predict(
         clf,
         X_scaled,
         y,
         cv=num_crossval_folds,
         method="predict_proba",
     )
     ranked_label_issues = find_label_issues(
         labels=y, pred_probs=pred_probs, return_indices_ranked_by="self_confidence"
     )

     print(f"Cleanlab found {len(ranked_label_issues)} potential label errors.")

     X_train, X_test, y_train, y_test = train_test_split(
         X_encoded,
         y,
         test_size=0.25,
         random_state=SEED,
     )
     scaler = StandardScaler()
     X_train[num_features] = scaler.fit_transform(X_train[num_features])
     X_test[num_features] = scaler.transform(X_test[num_features])

     X_train = X_train.to_numpy()
     y_train = y_train.to_numpy()
     X_test = X_test.to_numpy()
     y_test = y_test.to_numpy()

     clf.fit(X_train, y_train)
     acc_og = clf.score(X_test, y_test)
     print(f"Test accuracy of original logistic regression: {acc_og}")

     Note we first re-initialize clf

     # https://scikit-learn.org/stable/modules/linear_model.html#logistic-regression
     clf = LogisticRegression(solver='saga', max_iter=10000)

     cl = CleanLearning(clf)  # cl has same methods/attributes as clf
     _ = cl.fit(X_train, y_train)
     preds = cl.predict(X_test)
     acc_cl = accuracy_score(y_test, preds)
     print(f"Test accuracy of cleanlab's CatBoost: {acc_cl}")

    # PU
    pulearning = (None, 0, 1)
    # Only works for 2 class datasets. Set to the integer of the class
    # that is perfectly labeled (you are certain that there are no errors
    # in that class).
    l = CleanLearning(clf=clf, pulearning=pulearning[1])
    print(pulearning[1])
     print(X_train, y_train)
     sys.exit()
    cl.fit(X=X_train, labels=y_train)
    # Estimate the predictions you would have gotten by training with *no* label errors.
    predicted_test_labels = cl.predict(X_test)

    accuracy = accuracy_score(y_true=y_test, y_pred=predicted_test_labels)
    print(f"Test accuracy of cleanlab's CatBoost with pulearning: {accuracy} %")


if __name__ == '__main__':
    csv_folder = Path('confident_learning', 'csv')
    csv = Path('promejutok_2.csv')
    main(csv_in=csv_folder / csv)
