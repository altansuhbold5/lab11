import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB, BernoulliNB
from sklearn.metrics import confusion_matrix, precision_score, recall_score, f1_score
from sklearn.model_selection import train_test_split

from nb_scratch import NaiveBayes, tokenize

PAT = r"[a-z0-9$%']+"

df = pd.read_csv('spam_emails_en.csv')
train, test = train_test_split(df, test_size=0.2, random_state=42, stratify=df['label'])

cv = CountVectorizer(token_pattern=PAT)
Xtr = cv.fit_transform(train['text'])
Xte = cv.transform(test['text'])

clf = MultinomialNB(alpha=1.0).fit(Xtr, train['label'])
pred_sk = clf.predict(Xte)

print('sklearn accuracy:', round((pred_sk == test['label']).mean(), 4))
print('Xtr shape:', Xtr.shape)

nb = NaiveBayes(alpha=1.0).fit(train['text'], train['label'])
pred_own = np.array([nb.predict(t) for t in test['text']])

print('Өөрийн ба sklearn-ийн таарсан хувь:', round((pred_own == pred_sk).mean() * 100, 2), '%')

labels = ['ham', 'spam']
print(confusion_matrix(test['label'], pred_sk, labels=labels))

y = test['label']

print(
    'precision', round(precision_score(y, pred_sk, pos_label='spam'), 4),
    'recall', round(recall_score(y, pred_sk, pos_label='spam'), 4),
    'f1', round(f1_score(y, pred_sk, pos_label='spam'), 4)
)

si = list(clf.classes_).index('spam')
hi = 1 - si

ratio = clf.feature_log_prob_[si] - clf.feature_log_prob_[hi]
vocab = np.array(cv.get_feature_names_out())

print('Спамыг заадаг:', vocab[np.argsort(-ratio)[:8]].tolist())
print('Энгийнийг заадаг:', vocab[np.argsort(ratio)[:8]].tolist())

proba = clf.predict_proba(Xte)
maxp = proba.max(axis=1)

print('max proba > 0.99:', int((maxp > 0.99).sum()), 'of', len(maxp))

p_spam = proba[:, si]

for t in [0.5, 0.9, 0.99]:
    pr = np.where(p_spam >= t, 'spam', 'ham')
    print('threshold', t, 'accuracy', round((pr == y.values).mean(), 4))

train_set = set(train['text'])
dup_in_test = test['text'].isin(train_set).sum()

print('Test-ийн имэйлийн тоо, сургалтад яг адил текст байгаа:', dup_in_test, 'of', len(test))

my = pd.DataFrame({
    'label': ['ham', 'ham', 'ham', 'ham', 'ham', 'spam', 'spam', 'spam', 'spam', 'spam'],
    'text': [
        "Hi, can we move our group meeting to Thursday? I am free after 3 pm. Thanks",
        "Your free trial of the library database ends next week, so please save your notes before then.",
        "Reminder: the project report is due Monday at noon. Send it to me by email.",
        "Quantum computer lecture slides are uploaded to the course page. See you Wednesday.",
        "Can you send me the lunch menu for the cafeteria tomorrow? Thanks in advance.",
        "Congratulations! You have won a free iPhone. Click here to claim now http://bit.ly/x9y",
        "Dear customer, your bank account is locked. Verify your details at http://secure-bank.top to unlock.",
        "Hello friend, I have a business proposal worth $50,000. Reply to me privately for details.",
        "Нэг удаагийн санал! Танд 500000 төгрөгийн бэлэг хүлээгдэж байна. Линк дээр дарна уу http://gift-mn.xyz",
        "Поздравляем! Вы выиграли 1000$. Нажмите ссылку, чтобы получить приз http://prize.ru",
    ],
})

print()

for _, r in my.iterrows():
    p = nb.predict_proba(r['text'])
    print(f"{r['label']:5s} -> {nb.predict(r['text']):5s} spam_p={p['spam']:.4f} | {r['text'][:60]}")

print(
    'Үгсийн санд байхгүй үгс: "quantum", "computer" -> train vocab:',
    'quantum' in nb.vocab,
    'computer' in nb.vocab
)

print('Кирилл үсгийн токен:', tokenize(my['text'].iloc[8]), tokenize(my['text'].iloc[9]))

bn_c = CountVectorizer(token_pattern=PAT, binary=True)
Xb_tr = bn_c.fit_transform(train['text'])
Xb_te = bn_c.transform(test['text'])

bnb = BernoulliNB(alpha=1.0).fit(Xb_tr, train['label'])

print('BernoulliNB accuracy:', round(bnb.score(Xb_te, y), 4))

tf = TfidfVectorizer(token_pattern=PAT)
Xt_tr = tf.fit_transform(train['text'])
Xt_te = tf.transform(test['text'])

tfnb = MultinomialNB(alpha=1.0).fit(Xt_tr, train['label'])

print('TF-IDF + MultinomialNB accuracy:', round(tfnb.score(Xt_te, y), 4))

my_X = cv.transform(my['text'])

print('MNB(counts) on my 10:', (clf.predict(my_X) == my['label']).mean())
print('BNB on my 10:', (bnb.predict(bn_c.transform(my['text'])) == my['label']).mean())
print('TF-IDF MNB on my 10:', (tfnb.predict(tf.transform(my['text'])) == my['label']).mean())