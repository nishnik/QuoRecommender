################################################################################
# Information Retrieval 2016
#   Group: 13
#   Project: 26
#
# Workbench to train and benchmark classifiers#
#
# Current stats (on 8K question with 20% as test set):
#   kNN:
#     precision: 78.32%
#     recall: 3.60%
#     f1 score: 0.0688
#   OneVsRest:
#     precisio: 84.39%
#     recall: 2.17%
#     f1 score: 0.0423
#   LDA:
#
# Implementation:
#   Preprocessing with TfidfVectorizer
#     - n_grams, stop words
#   kNN Classifier k = 10
#   OneVsRest Classifier with LinearSVC
#   Performance evaluation (performance, recall and f1_score)
#
# NOTE:
#   1. Currently runs on small datasets of ~8k questions
#   2. Recall of kNN and OVR is very low as classifiers aren't predicting any tags
#   3. Suppressed LinearSVC warnings due to sparse topic matrix
#   4. Minor TODOs to improve workbench spread throughout code
#   5. Currently using just precision to plot graphs
#
# TODO Major:
#   1. Improve recall of kNN and OneVsRest
#         - Force each classifier to make atleast 1 prediction
#            => http://stackoverflow.com/questions/34561554/scikit-learn-label-not-x-is-present-in-all-training-examples
#            =>  https://github.com/davidthaler/Greek_media
#         - Select all labels above threshold of max score in labels for query
#         - Priority 1 to either improve recall or succesfully implement LDA
#   2. Test on entire corpus
#         - Will probably need to limit tags to top 1000 labels in traininig/testing
#         - OR Incorporate clustering OR Both
#         - Needs to be done only once if done correctly on 8K questions first
#         - Train once and store classifier to disk as .pkl
#   3. Add LDA Classifier
#   4. Make interface
#         - If testing on query from outside corpus,
#             might need to remove non vocab words from query
#
# Future Scope:
#   1. Feature reduction by chi-squared or mutual-info to reduce time
#   2. GridSearch on a validation set to find optimal hyperparameters of
#       different classifiers
#   3. Other classifiers like OneVSOne
#   4. Train/Predict on question bodies as well, instead of just question headings.
#       More work on feature selection in general
#   5.
#
###############################################################################


import json
import sys
import random
from optparse import OptionParser
from time import time
import numpy as np
import matplotlib.pyplot as plt
import warnings


from sklearn.cross_validation import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.feature_selection import SelectKBest, chi2
from sklearn.svm import LinearSVC, SVC
from sklearn.multiclass import OneVsRestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.utils.extmath import density
from sklearn import metrics
from sklearn.preprocessing import MultiLabelBinarizer

# """Trim string to fit on terminal (assuming 80-column display)"""
def trim(s):
    return s if len(s) <= 80 else s[:77] + "..."

# Parse commandline arguments
# NOTE: report works
# TODO: test/debug top10, implement tags, fix/replace chi2
op = OptionParser()
op.add_option("--report",
              action="store_true", dest="print_report",
              help="Print a detailed classification report.")
op.add_option("--show_samples",
              action="store_true", dest="show_samples",
              help="Print 10 sample predictions from each classifier.")
op.add_option("--hide_warnings",
              action="store_true", dest="hide_warnings",
              help="Hide warnings in OVR generated due to sparse label matrix.")
op.add_option("--top10",
              action="store_true", dest="print_top10",
              help="Print ten most discriminative terms per class"
                   " for every classifier.")
op.add_option("--tags",
              action="store", type = "int", dest="tag_count",
              help="limit to top k tags to train on, default all")
# op.add_option("--chi2_select",
#               action="store", type="int", dest="select_chi2",
#               help="Select some number of features using a chi-squared test")
# op.add_option("--confusion_matrix",
#               action="store_true", dest="print_cm",
#               help="Print the confusion matrix.")

(opts, args) = op.parse_args()
if len(args) > 0:
    op.error("this script takes no arguments.")
    sys.exit(1)
print(__doc__)
op.print_help()
print("\n")
print('=' * 80)

# TODO: train and test on full corpus
data_file = open('test8k.json')
data = json.load(data_file)

# TODO: experiment with different arguments of tfidfvectorizer, like tokenizer
# Each N-gram adds O(N_vocab) features,
# If using full corpus, might need to disable ngram or reduce features with chi^2/MI
vectorizer = TfidfVectorizer(sublinear_tf=True,
                        max_df=0.5,
                        stop_words='english',
                        ngram_range = (1, 1))
print("Fitting features from the corpus to a TFIDF vectorizer")
questions = data.keys()
t0 = time()
vectorizer.fit(questions)
duration = time() - t0
print("done in %fs" % (duration))
print("")

tags = data.values()
mlb = MultiLabelBinarizer()
mlb.fit(tags)

questions_train, questions_test, tags_train, tags_test = train_test_split(questions, tags, test_size=0.2, random_state = random.randint(1, 100))

print("Extracting features from the training data using the vectorizer")
t0 = time()
X_train = vectorizer.transform(questions_train)
duration = time() - t0
print("done in %fs" % (duration))
print("n_samples: %d, n_features: %d" % X_train.shape)
print("")

print("Extracting features from the test data using the vectorizer")
t0 = time()
X_test = vectorizer.transform(questions_test)
duration = time() - t0
print("done in %fs" % (duration))
print("n_samples: %d, n_features: %d" % X_test.shape)
print("")
feature_names = vectorizer.get_feature_names()
if feature_names:
    feature_names = np.asarray(feature_names)

y_train = mlb.transform(tags_train)
y_test = mlb.transform(tags_test)
tags = list(mlb.classes_)
print("n_unique_tags = %d" % len(tags))

print("")



# chi2 can be used to reduce the number of features to the top k most relevant

# if opts.select_chi2:
#     print("Extracting %d best features by a chi-squared test" %
#           opts.select_chi2)
#     t0 = time()
#     ch2 = SelectKBest(chi2, k=opts.select_chi2)
#     X_train = ch2.fit_transform(X_train, y_train)
#     X_test = ch2.transform(X_test)
#     if feature_names:
#         # keep selected feature names
#         feature_names = [feature_names[i] for i
#                          in ch2.get_support(indices=True)]
#     print("done in %fs" % (time() - t0))
#     print("\n")


###############################################################################
# Benchmark classifiers
#
#  Classifier must implement at least fit, predict to use this function
def benchmark(clf):
    print('_' * 80)
    print("Training: ")
    print(clf)
    t0 = time()
    clf.fit(X_train, y_train)
    train_time = time() - t0
    print("training time: %0.3fs" % train_time)
    print("")

    print('_' * 80)
    print("Predicting: ")
    t0 = time()
    pred = clf.predict(X_test)
    test_time = time() - t0
    print("prediction time:  %0.3fs" % test_time)
    print("")

    if opts.show_samples :
        print("_" * 80)
        print("Sample predictions 10 out of %s:" % len(questions_test))
        sample_questions = questions_test[:10]
        sample_true = mlb.inverse_transform(y_test[:10])
        sample_pred = mlb.inverse_transform(pred[:10])
        for i in range(len(sample_questions)) :
            print(". " * 40)
            print(sample_questions[i])
            print("True labels= ", sample_true[i])
            print("Predicted labels = ", sample_pred[i])
            print("")

    print("_" * 80)
    print("Calculating performance metrics")
    t0 = time()
    # 'micro':
    #       Calculate metrics globally by counting the total true positives,
    #       false negatives and false positives.
    # 'macro':
    #       Calculate metrics for each label, and find their unweighted mean.
    #       This does not take label imbalance into account.
    # 'weighted':
    #       Calculate metrics for each label, and find their average,
    #       weighted by support (the number of true instances for each label).
    #       This alters 'macro' to account for label imbalance;
    #       it can result in an F-score that is not between precision and recall.
    precision, recall, f1_score, support = metrics.precision_recall_fscore_support(y_test, pred, average='micro')
    metrics_time = time() - t0
    print("metrics calculation time:  %0.3fs" % metrics_time)
    print("precision: %0.2f %%" % (precision * 100))
    print("recall: %0.2f %%" % (recall * 100))
    print("f score: %0.4f" % f1_score)
    print("")

    # TODO: test/debug
    if hasattr(clf, 'coef_'):
        print("_" * 80)
        print("dimensionality: %d" % clf.coef_.shape[1])
        print("density: %f" % density(clf.coef_))
        print("")

        if opts.print_top10 and feature_names is not None:
            print("_" * 80)
            print("top 10 keywords per class:")
            for i, category in enumerate(tags):
                top10 = np.argsort(clf.coef_[i])[-10:]
                print(trim("%s: %s"
                      % (category, " ".join(feature_names[top10]))))
            print("")

    if opts.print_report:
        print("_" * 80)
        print("classification report:")
        print(metrics.classification_report(y_test, pred,
                                            target_names=tags))
        print("")

    # if opts.print_cm:
    #     print("confusion matrix:")
    #     print(metrics.confusion_matrix(y_test, pred))

    # TODO: return f1_score instead of precision after improving recall
    print("\n")
    clf_descr = str(clf).split('(')[0]
    return clf_descr, precision, train_time, test_time


results = []



print('=' * 80)
print("kNN Classifier")
results.append(benchmark(KNeighborsClassifier(n_neighbors=10)))

print('=' * 80)
print("OVR LinearSVC Classifier")
if opts.hide_warnings:
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        results.append(benchmark(OneVsRestClassifier(LinearSVC())))
else:
    results.append(benchmark(OneVsRestClassifier(LinearSVC())))

# TODO: Add LDA Classifier
# def benchmarkLDAClassifier()
# should return name, precision (OR f1score), train_time, test_time
#
# print('=' * 80)
# print("LDA Classifier")
# results.append(benchmarkLDAClassifier())


# make some plots

indices = np.arange(len(results))

results = [[x[i] for x in results] for i in range(4)]

clf_names, score, training_time, test_time = results
training_time = np.array(training_time) / (np.max(training_time))
test_time = np.array(test_time) / (np.max(test_time))

plt.figure(figsize=(12, 8))
plt.title("Score")
plt.barh(indices, score, .2, label="precision", color='r')
plt.barh(indices + .3, training_time, .2, label="training time", color='g')
plt.barh(indices + .6, test_time, .2, label="test time", color='b')
plt.yticks(())
plt.legend(loc='best')
plt.subplots_adjust(left=.25)
plt.subplots_adjust(top=.95)
plt.subplots_adjust(bottom=.05)

for i, c in zip(indices, clf_names):
    plt.text(-.3, i, c)

plt.show()
