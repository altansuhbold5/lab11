import re
import math
from collections import Counter

import pandas as pd
from sklearn.model_selection import train_test_split


def tokenize(s):
    return re.findall(r"[a-z0-9$%']+", s.lower())


class NaiveBayes:
    def __init__(self, alpha=1.0):
        self.alpha = alpha

    def fit(self, texts, labels):
        self.classes_ = sorted(set(labels))
        self.n_docs = len(labels)
        self.prior_count = Counter()
        self.word_count = {c: Counter() for c in self.classes_}
        self.vocab = set()

        for text, c in zip(texts, labels):
            tokens = tokenize(text)
            self.prior_count[c] += 1
            self.word_count[c].update(tokens)
            self.vocab.update(tokens)

        self.V = len(self.vocab)
        self.total_tokens = {
            c: sum(self.word_count[c].values())
            for c in self.classes_
        }
        return self

    def log_prior(self, c):
        return math.log(self.prior_count[c] / self.n_docs)

    def log_likelihood(self, w, c):
        num = self.word_count[c][w] + self.alpha
        den = self.total_tokens[c] + self.alpha * self.V
        return math.log(num / den)

    def log_score(self, text, c):
        score = self.log_prior(c)

        for w in tokenize(text):
            if w in self.vocab:
                score += self.log_likelihood(w, c)

        return score

    def predict(self, text):
        return max(
            self.classes_,
            key=lambda c: self.log_score(text, c)
        )

    def predict_proba(self, text):
        logs = {
            c: self.log_score(text, c)
            for c in self.classes_
        }

        m = max(logs.values())
        exps = {
            c: math.exp(v - m)
            for c, v in logs.items()
        }
        z = sum(exps.values())

        return {
            c: e / z
            for c, e in exps.items()
        }


def confusion(y_true, y_pred, pos='spam', neg='ham'):
    tp = sum(t == pos and p == pos for t, p in zip(y_true, y_pred))
    fp = sum(t == neg and p == pos for t, p in zip(y_true, y_pred))
    fn = sum(t == pos and p == neg for t, p in zip(y_true, y_pred))
    tn = sum(t == neg and p == neg for t, p in zip(y_true, y_pred))

    return tp, fp, fn, tn


def metrics(tp, fp, fn, tn):
    n = tp + fp + fn + tn
    acc = (tp + tn) / n
    prec = tp / (tp + fp) if tp + fp else 0.0
    rec = tp / (tp + fn) if tp + fn else 0.0
    f1 = 2 * prec * rec / (prec + rec) if prec + rec else 0.0

    return acc, prec, rec, f1


if __name__ == '__main__':
    df = pd.read_csv('spam_emails_en.csv')

    train, test = train_test_split(
        df,
        test_size=0.2,
        random_state=42,
        stratify=df['label']
    )

    print('Train/Test:', len(train), len(test), test['label'].value_counts().to_dict())

    train_vocab = {
        w for t in train['text']
        for w in tokenize(t)
    }

    print('Train үгсийн сангийн хэмжээ:', len(train_vocab))
    print('tokenize:', tokenize('YOU WON $1,000,000!!! Click NOW'))

    nb = NaiveBayes(alpha=1.0).fit(
        train['text'],
        train['label']
    )

    y_true = test['label'].tolist()
    y_pred = [nb.predict(t) for t in test['text']]

    tp, fp, fn, tn = confusion(y_true, y_pred)
    acc, prec, rec, f1 = metrics(tp, fp, fn, tn)

    print(f'Accuracy={acc:.4f}')

    tests = [
        "Congratulations winner, click the link to claim your free prize",
        "What time does the database class start tomorrow",
        "There is a discount book sale on campus, want to go together",
    ]

    for t in tests:
        print(
            nb.predict(t),
            {k: round(v, 4) for k, v in nb.predict_proba(t).items()},
            '|',
            t
        )

    print('0.001 ** 200 =', 0.001 ** 200)
    print('Log result:', 200 * math.log(0.001))

    print('CM (мөр=бодит, багана=таамаглал): ham/spam')
    print('ham :', tn, fp)
    print('spam:', fn, tp)

    print(f'precision={prec:.4f} recall={rec:.4f} F1={f1:.4f}')

    for a in [0.01, 1, 5]:
        m = NaiveBayes(alpha=a).fit(
            train['text'],
            train['label']
        )

        p = [m.predict(t) for t in test['text']]
        cm = confusion(y_true, p)

        print(
            'alpha', a,
            [round(x, 4) for x in metrics(*cm)],
            'CM', cm
        )

    wrong = test[
        [p != t for p, t in zip(y_pred, y_true)]
    ].copy()

    wrong['model'] = [
        p for p, t in zip(y_pred, y_true)
        if p != t
    ]

    wrong['proba_spam'] = [
        round(nb.predict_proba(t)['spam'], 4)
        for t in wrong['text']
    ]

    pd.set_option('display.max_colwidth', 200)
    pd.set_option('display.width', 250)

    print(
        wrong[
            ['label', 'model', 'proba_spam', 'text']
        ].to_string()
    )

    probas = pd.Series([
        nb.predict_proba(t)['spam']
        for t in test['text']
    ], index=test.index)

    print(
        'max proba > 0.99 хувь:',
        round((probas.clip(lower=0.5) > 0.99).mean() * 100, 1),
        '| max proba > 0.99 (нийт):',
        round(((probas > 0.99) | (probas < 0.01)).mean() * 100, 1)
    )
