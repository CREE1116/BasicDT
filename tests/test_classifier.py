import os
import tempfile
import numpy as np
import pandas as pd
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split
from basicdt import BasicDTClassifier, load_model


def test_basicdt_classifier_fit_predict():
    X, y = make_classification(
        n_samples=200, n_features=10, n_classes=2, random_state=42
    )
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    clf = BasicDTClassifier(
        n_estimators=10, max_depth=4, learning_rate=0.1, random_state=42, verbose=True
    )
    clf.fit(X_train, y_train, eval_set=[(X_test, y_test)])

    assert clf.get_n_trees() > 0

    preds = clf.predict(X_test)
    assert preds.shape == (len(y_test),)
    assert np.all((preds == 0) | (preds == 1))

    probas = clf.predict_proba(X_test)
    assert probas.shape == (len(y_test), 2)
    assert np.allclose(probas.sum(axis=1), 1.0)


def test_basicdt_serialization():
    X, y = make_classification(n_samples=100, n_features=5, random_state=42)
    clf = BasicDTClassifier(n_estimators=5, max_depth=3, random_state=42)
    clf.fit(X, y)

    # Save to a temporary file
    with tempfile.TemporaryDirectory() as tmpdir:
        model_path = os.path.join(tmpdir, "basicdt_model.pkl")
        clf.save(model_path)

        # Load back
        clf_loaded = load_model(model_path)
        assert clf_loaded.get_n_trees() == clf.get_n_trees()

        # Compare predictions
        preds_orig = clf.predict(X)
        preds_loaded = clf_loaded.predict(X)
        assert np.array_equal(preds_orig, preds_loaded)

        probas_orig = clf.predict_proba(X)
        probas_loaded = clf_loaded.predict_proba(X)
        assert np.allclose(probas_orig, probas_loaded)


def test_basicdt_categorical_handling():
    # Construct a dataset with categorical column
    rng = np.random.default_rng(42)
    N = 100
    df = pd.DataFrame({
        "num1": rng.normal(size=N),
        "cat1": rng.choice(["apple", "banana", "cherry"], size=N),
        "num2": rng.normal(size=N),
    })
    # Target
    y = (df["num1"] + (df["cat1"] == "banana").astype(float) > 0.5).astype(int)

    clf = BasicDTClassifier(n_estimators=5, max_depth=3, random_state=42)
    clf.fit(df, y)

    preds = clf.predict(df)
    assert preds.shape == (N,)
    probas = clf.predict_proba(df)
    assert np.allclose(probas.sum(axis=1), 1.0)
