import numpy as np
import pandas as pd

from subprocess import check_output
#print(check_output(["ls", "survey.csv"]).decode("utf8"))

from sklearn.preprocessing import LabelEncoder
le = LabelEncoder()

from sklearn.cluster import DBSCAN
from sklearn import metrics
from sklearn.datasets.samples_generator import make_blobs
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.model_selection import cross_val_score
from sklearn import naive_bayes

from sklearn import svm

import matplotlib.pyplot as plt


df = pd.read_csv("survey.csv")
list(df)
df.head()


###
#CLEAN THE DATA

df.family_history = le.fit_transform(df.family_history)
df.mental_health_consequence = le.fit_transform(df.mental_health_consequence)
df.phys_health_consequence = le.fit_transform(df.phys_health_consequence)
df.coworkers = le.fit_transform(df.coworkers)
df.supervisor = le.fit_transform(df.supervisor)
df.mental_health_interview = le.fit_transform(df.mental_health_interview)
df.phys_health_interview = le.fit_transform(df.phys_health_interview)
df.mental_vs_physical = le.fit_transform(df.mental_vs_physical)
df.obs_consequence = le.fit_transform(df.obs_consequence)
df.remote_work = le.fit_transform(df.remote_work)
df.tech_company = le.fit_transform(df.tech_company)
df.benefits = le.fit_transform(df.benefits)
df.care_options = le.fit_transform(df.care_options)
df.wellness_program = le.fit_transform(df.wellness_program)
df.seek_help = le.fit_transform(df.seek_help)
df.anonymity = le.fit_transform(df.anonymity)

df.loc[df['work_interfere'].isnull(),['work_interfere']]=0 #replace all NaNs with zero


#Deal with nulls
df['self_employed'].fillna('Don\'t know',inplace=True)
df.self_employed = le.fit_transform(df.self_employed)
df.loc[df['comments'].isnull(),['comments']]=0 # replace all no comments with zero
df.loc[df['comments']!=0,['comments']]=1 # replace all comments with a flag 1

df['leave'].replace(['Very easy', 'Somewhat easy', "Don\'t know", 'Somewhat difficult', 'Very difficult'],
                     [1, 2, 3, 4, 5],inplace=True)
df['work_interfere'].replace(['Never','Rarely','Sometimes','Often'],[1,2,3,4],inplace=True)


#Keep Ordering (Maybe change this to Male, Neither or Female only)
df.loc[df['Gender'].str.contains('F|w', case=False,na=False),'Gender']=2
df.loc[df['Gender'].str.contains('queer/she',case=False,na=False),'Gender']=1
df.loc[df['Gender'].str.contains('male leaning',case=False,na=False),'Gender']=-1
df.loc[df['Gender'].str.contains('something kinda male',case=False,na=False),'Gender']=-1
df.loc[df['Gender'].str.contains('ish',case=False,na=False),'Gender']=-1
df.loc[df['Gender'].str.contains('m',case=False,na=False),'Gender']=-2
df.loc[df['Gender'].str.contains('',na=False),'Gender']=0


#Keep Ordering
df.loc[df['no_employees']=='1-5',['no_employees']]=1
df.loc[df['no_employees']=='6-25',['no_employees']]=2
df.loc[df['no_employees']=='26-100',['no_employees']]=3
df.loc[df['no_employees']=='100-500',['no_employees']]=4
df.loc[df['no_employees']=='500-1000',['no_employees']]=5
df.loc[df['no_employees']=='More than 1000',['no_employees']]=6


#Drop Elements
drop_elements = ['Timestamp','Country','state','work_interfere']
df = df.drop(drop_elements, axis = 1)

###

#X is features, y is dependent variable
X = df.drop(['treatment','comments'],axis=1)
pca = PCA(n_components=16)

pca.fit(X)
X_transformed = pca.fit_transform(X)
print(X_transformed)

y = df['treatment']
labels_true = y
y = le.fit_transform(y)


#DBSCAN STUFF#
#NEED TO CHOOSE EPS AND MIN_SAMPLES#
db = DBSCAN(eps=3, min_samples=5).fit(X_transformed)
core_samples = db.core_sample_indices_
print(len(X))
core_samples_mask = np.zeros_like(db.labels_, dtype=bool)
core_samples_mask[db.core_sample_indices_] = True
labels = db.labels_

n_clusters_ = len(set(labels)) - (1 if -1 in labels else 0)

print('Estimated number of clusters: %d' % n_clusters_)
print("Homogeneity: %0.3f" % metrics.homogeneity_score(labels_true, labels))
print("Completeness: %0.3f" % metrics.completeness_score(labels_true, labels))
print("V-measure: %0.3f" % metrics.v_measure_score(labels_true, labels))
print("Adjusted Rand Index: %0.3f"
      % metrics.adjusted_rand_score(labels_true, labels))
print("Adjusted Mutual Information: %0.3f"
      % metrics.adjusted_mutual_info_score(labels_true, labels))
print("Silhouette Coefficient: %0.3f"
      % metrics.silhouette_score(X, labels))

# Black removed and is used for noise instead.
unique_labels = set(labels)
colors = plt.cm.Spectral(np.linspace(0, 1, len(unique_labels)))
for k, col in zip(unique_labels, colors):
    if k == -1:
        # Black used for noise.
        col = 'k'

    class_member_mask = (labels == k)

    xy = X[class_member_mask & core_samples_mask]
    plt.plot(xy.iloc[:, 0], xy.iloc[:, 1], 'o', markerfacecolor=col,
             markeredgecolor='k', markersize=14)

    xy = X[class_member_mask & ~core_samples_mask]
    plt.plot(xy.iloc[:, 0], xy.iloc[:, 1], 'o', markerfacecolor=col,
             markeredgecolor='k', markersize=6)

plt.title('Estimated number of clusters: %d' % n_clusters_)

#SVM
clf = svm.SVC(decision_function_shape = 'ovo')
clf.fit(X_transformed, y)
scores_ = cross_val_score(clf, X_transformed, y, cv=10, scoring='accuracy')
training_score_SVM = clf.score(X_transformed,y)
testing_score_SVM = scores_.mean()

print("SVM training accuracy: " + str(training_score_SVM))
print("SVM testing accuracy: " + str(testing_score_SVM))

#Naive Bayes
gnb = naive_bayes.GaussianNB()
gnb.fit(X_transformed,y)
gnb_scores = cross_val_score(gnb, X_transformed, y, cv=10, scoring='accuracy')
training_score_bayes = gnb.score(X_transformed,y)
testing_score_bayes = gnb_scores.mean()

print("Naive Bayes training accuracy" + str(training_score_bayes))
print("Naive Bayes testing accuracy" + str(testing_score_bayes))
