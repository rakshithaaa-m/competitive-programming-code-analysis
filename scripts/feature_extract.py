#!/usr/bin/env python3
"""
feature_extract.py
- Reads processed/<contest>/*.json, builds a corpus (each example is joined tokens),
  uses sklearn CountVectorizer(ngram_range=(1,2), min_df=...) to create features,
  saves vectorizer + X + labels to features/ (joblib).
- Also creates two labels: rank (derived from rating if available) and country
"""
import os, json, argparse
from glob import glob
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.preprocessing import LabelEncoder
import joblib
import numpy as np

def rating_to_rank_label(rating):
    # map numeric rating to rank index 0..9 (Newbie..Legendary)
    if rating is None: return None
    r = int(rating)
    if r >= 3000: return 'LegendaryGrandmaster'
    if r >= 2600: return 'InternationalGrandmaster'
    if r >= 2400: return 'Grandmaster'
    if r >= 2300: return 'InternationalMaster'
    if r >= 2100: return 'Master'
    if r >= 1900: return 'CandidateMaster'
    if r >= 1600: return 'Expert'
    if r >= 1400: return 'Specialist'
    if r >= 1200: return 'Pupil'
    return 'Newbie'

def main(args):
    corpus = []
    meta = []
    for contest_dir in os.listdir(args.processed_dir):
        pdir = os.path.join(args.processed_dir, contest_dir)
        if not os.path.isdir(pdir): continue
        for jf in os.listdir(pdir):
            obj = json.load(open(os.path.join(pdir, jf), encoding='utf-8'))
            tokens = obj.get('tokens', [])
            text = ' '.join(tokens)
            if len(text) < 5: continue
            corpus.append(text)
            meta.append(obj)
    print("examples:", len(corpus))
    vect = CountVectorizer(ngram_range=(1,2), min_df=args.min_df)
    X = vect.fit_transform(corpus)
    # labels
    countries = [m.get('country') or 'Unknown' for m in meta]
    le_country = LabelEncoder()
    y_country = le_country.fit_transform(countries)
    # Save
    os.makedirs(args.outdir, exist_ok=True)
    joblib.dump(vect, os.path.join(args.outdir, 'vectorizer.joblib'))
    joblib.dump(le_country, os.path.join(args.outdir, 'le_country.joblib'))
    joblib.dump((X, y_country, countries), os.path.join(args.outdir, 'features_country.joblib'))
    print("Saved features to", args.outdir)

if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument('--processed_dir', default='processed')
    p.add_argument('--outdir', default='features')
    p.add_argument('--min_df', type=float, default=0.01)  # relative freq threshold; tune if dataset small
    args = p.parse_args()
    main(args)
