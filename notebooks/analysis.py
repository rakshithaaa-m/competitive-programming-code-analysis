# notebooks/analysis.py  (or paste into notebook cell)
import os, re, glob
import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import seaborn as sns
import joblib

DATA_SNIPPETS = "../data/snippets"  # if running from notebooks/ make path ../data
LABELS_CSV = "../data/labels.csv"
RESULTS_DIR = "../results"
os.makedirs(RESULTS_DIR, exist_ok=True)

# --- helper functions ---
def read_labels(path=LABELS_CSV):
    df = pd.read_csv(path)
    return df

def read_snippet(filename):
    with open(os.path.join("../data/snippets", filename), "r", encoding="utf-8") as f:
        return f.read()

# Simple preprocessing: remove comments, replace string and char literals, replace identifiers roughly
def preprocess_cpp(code):
    # remove single-line comments
    code = re.sub(r"//.*", " ", code)
    # remove /* ... */ comments
    code = re.sub(r"/\*.*?\*/", " ", code, flags=re.DOTALL)
    # replace string literals and char literals
    code = re.sub(r'".*?"', ' !!STR ', code)
    code = re.sub(r"'.*?'", ' !!CHR ', code)
    # replace numbers
    code = re.sub(r'\b\d+\b', ' !!NUM ', code)
    # very crude identifier normalization: replace words that look like identifiers but not keywords
    keywords = set("""
    auto bool break case catch char class const continue default delete do double else enum extern false float for goto
    if inline int long namespace new noexcept nullptr operator private protected public register return short signed sizeof
    static struct switch template this throw true try typedef typeid typename union unsigned using virtual void volatile while
    """.split())
    # tokenization by non-word:
    tokens = re.split(r'(\W)', code)
    # replace tokens that look like identifiers and are not keywords or numbers
    tokens2 = []
    for t in tokens:
        if re.fullmatch(r'[A-Za-z_]\w*', t) and t not in keywords:
            tokens2.append('!!VAR')
        else:
            tokens2.append(t)
    return " ".join(tokens2)

# Build dataset
labels_df = read_labels()
texts, ys, files = [], [], []
for _, row in labels_df.iterrows():
    fname = row['filename']
    lab = int(row['label'])
    code = read_snippet(fname)
    proc = preprocess_cpp(code)
    texts.append(proc)
    ys.append(lab)
    files.append(fname)

print("Loaded", len(texts), "samples")

# --- Features ---
# Use TF-IDF on token sequences with unigrams+bigrams
vectorizer = TfidfVectorizer(ngram_range=(1,2), token_pattern=r'(?u)\b\w+\b', max_features=2000)
X = vectorizer.fit_transform(texts)
y = np.array(ys)

# Add simple handcrafted numeric features: code length (tokens), count of 'scanf'/'printf', count of 'cin'/'cout', count of 'template' and '#include'
def extra_features(texts):
    features = []
    for t in texts:
        token_count = len(t.split())
        scanf_cnt = t.count("scanf") + t.count("printf")
        cin_cnt = t.count("cin") + t.count("cout")
        template_cnt = t.count("template")
        include_cnt = t.count("#include")
        features.append([token_count, scanf_cnt, cin_cnt, template_cnt, include_cnt])
    return np.array(features, dtype=float)

X_extra = extra_features(texts)

# combine sparse TF-IDF with dense features: stack horizontally
from scipy.sparse import hstack
X_combined = hstack([X, X_extra])

# --- Train / test split ---
X_train, X_test, y_train, y_test, files_train, files_test = train_test_split(
    X_combined, y, files, test_size=0.3, random_state=42, stratify=y if len(set(y))>1 else None
)

# --- Logistic Regression model (baseline) ---
clf = LogisticRegression(max_iter=1000, class_weight='balanced', multi_class='auto', solver='liblinear')
clf.fit(X_train, y_train)
y_pred = clf.predict(X_test)

acc = accuracy_score(y_test, y_pred)
print("Logistic Regression accuracy:", acc)
print(classification_report(y_test, y_pred))

# Save metrics
with open(os.path.join(RESULTS_DIR, "metrics.txt"), "w") as f:
    f.write("Logistic Regression accuracy: %.4f\n\n" % acc)
    f.write(classification_report(y_test, y_pred))

# confusion matrix plot
cm = confusion_matrix(y_test, y_pred)
plt.figure(figsize=(5,4))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
plt.title("Confusion Matrix (LogReg)")
plt.xlabel("pred")
plt.ylabel("true")
plt.savefig(os.path.join(RESULTS_DIR, "confusion_matrix_logreg.png"), bbox_inches='tight')
plt.show()

# Save model and vectorizer
joblib.dump(clf, os.path.join(RESULTS_DIR, "logreg_model.joblib"))
joblib.dump(vectorizer, os.path.join(RESULTS_DIR, "vectorizer.joblib"))

# --- Simple Neural Network (Keras) ---
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout
from tensorflow.keras.callbacks import EarlyStopping

# Convert sparse to dense for simplicity (only ok for small demo)
X_train_dense = X_train.toarray()
X_test_dense = X_test.toarray()

num_classes = len(set(y))
model = Sequential([
    Dense(128, activation='relu', input_shape=(X_train_dense.shape[1],)),
    Dropout(0.5),
    Dense(64, activation='relu'),
    Dropout(0.3),
    Dense(num_classes, activation='softmax')
])
model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
es = EarlyStopping(patience=5, restore_best_weights=True)
model.fit(X_train_dense, y_train, validation_split=0.2, epochs=50, batch_size=8, callbacks=[es], verbose=2)

# evaluate
loss, acc_nn = model.evaluate(X_test_dense, y_test, verbose=0)
print("Neural network accuracy:", acc_nn)
with open(os.path.join(RESULTS_DIR, "metrics.txt"), "a") as f:
    f.write("\nNeural network accuracy: %.4f\n" % acc_nn)

# Save Keras model
model.save(os.path.join(RESULTS_DIR, "nn_model.h5"))

# Example: predict on one file and print top probabilities
i = 0
sample_file = files_test[i]
sample_text = preprocess_cpp(read_snippet(sample_file))
sample_vec = vectorizer.transform([sample_text])
sample_feat = extra_features([sample_text])
from scipy.sparse import hstack
sample_combined = hstack([sample_vec, sample_feat])
pred = clf.predict(sample_combined)
probs = clf.predict_proba(sample_combined)
print("Sample file:", sample_file, "predicted label (logreg):", pred[0], "probs:", probs[0])
