#!/usr/bin/env python3
"""
train_models.py
- Loads features from features/features_country.joblib and trains:
   * LogisticRegression
   * GaussianNB (approx GDA)
   * MLPClassifier (neural network)
- Saves model artifacts and prints classification reports.
"""
import joblib, argparse, os
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import GaussianNB
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import Normalizer, StandardScaler
from sklearn.metrics import classification_report, accuracy_score
import numpy as np

def main(args):
    vect = joblib.load(os.path.join(args.features_dir, 'vectorizer.joblib'))
    le_country = joblib.load(os.path.join(args.features_dir, 'le_country.joblib'))
    X_all, y_all, countries = joblib.load(os.path.join(args.features_dir, 'features_country.joblib'))
    # convert to dense for GaussianNB / StandardScaler (OK for small experiment)
    X = X_all.toarray()
    # normalize by L2 as in report then scale to mean 0 var1
    X = Normalizer(norm='l2').fit_transform(X)
    scaler = StandardScaler(with_mean=False)
    X = scaler.fit_transform(X)
    X_train, X_test, y_train, y_test = train_test_split(X, y_all, test_size=0.2, stratify=y_all, random_state=42)
    print("Train shape", X_train.shape)
    # Logistic
    log = LogisticRegression(max_iter=200, multi_class='multinomial', solver='saga', n_jobs=-1)
    log.fit(X_train, y_train)
    p_log = log.predict(X_test)
    print("Logistic accuracy:", accuracy_score(y_test, p_log))
    print(classification_report(y_test, p_log, target_names=le_country.classes_[:len(set(y_all))]))
    # GaussianNB
    gnb = GaussianNB()
    gnb.fit(X_train, y_train)
    p_gnb = gnb.predict(X_test)
    print("GNB accuracy:", accuracy_score(y_test, p_gnb))
    # MLP
    mlp = MLPClassifier(hidden_layer_sizes=(200,), max_iter=30)
    mlp.fit(X_train, y_train)
    p_mlp = mlp.predict(X_test)
    print("MLP accuracy:", accuracy_score(y_test, p_mlp))
    # Save models
    os.makedirs(args.outdir, exist_ok=True)
    joblib.dump(log, os.path.join(args.outdir, 'logreg.joblib'))
    joblib.dump(gnb, os.path.join(args.outdir, 'gnb.joblib'))
    joblib.dump(mlp, os.path.join(args.outdir, 'mlp.joblib'))
    joblib.dump(scaler, os.path.join(args.outdir, 'scaler.joblib'))
    print("Saved models to", args.outdir)

if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument('--features_dir', default='features')
    p.add_argument('--outdir', default='models')
    args = p.parse_args()
    main(args)
