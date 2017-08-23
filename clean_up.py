# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

from subprocess import check_output
print(check_output(["ls", "input/survey.csv"]).decode("utf8"))

# Any results you write to the current directory are saved as output.

from sklearn.preprocessing import LabelEncoder
le = LabelEncoder()

#Have a look at the data you have
df = pd.read_csv("input/survey.csv")
list(df)

print("DF", df)

#organize data by header
df.head()


# CLEAN UP

#Use labelencoder to replace all categorical information with ordinal information where
#you don't care about the order
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

df.loc[df['work_interfere'].isnull(),['work_interfere']]=0 # replace all NaNs with zero

#dealing with nulls
df['self_employed'].fillna('Don\'t know',inplace=True)
df.self_employed = le.fit_transform(df.self_employed)

#another way to deal with nulls
# Now change comments column to flag whether or not respondent made additional comments
df.loc[df['comments'].isnull(),['comments']]=0 # replace all no comments with zero
df.loc[df['comments']!=0,['comments']]=1 # replace all comments with a flag 1

#Preserve Order in some of the features
df['leave'].replace(['Very easy', 'Somewhat easy', "Don\'t know", 'Somewhat difficult', 'Very difficult'],
                     [1, 2, 3, 4, 5],inplace=True)
df['work_interfere'].replace(['Never','Rarely','Sometimes','Often'],[1,2,3,4],inplace=True)
#df.loc[df['work_interfere'].isnull(),['work_interfere']]=0 # replace all no comments with zero

#From assessing the unique ways in which gender was described above, the following script replaces gender on
#a -2 to 2 scale:
#-2:male
#-1:identifies male
#0:gender not available
#1:identifies female
#2: female.

#note that order of operations matters here, particularly for the -1 assignments that must be done before the
#male -2 assignment is done

df.loc[df['Gender'].str.contains('F|w', case=False,na=False),'Gender']=2
df.loc[df['Gender'].str.contains('queer/she',case=False,na=False),'Gender']=1
df.loc[df['Gender'].str.contains('male leaning',case=False,na=False),'Gender']=-1
df.loc[df['Gender'].str.contains('something kinda male',case=False,na=False),'Gender']=-1
df.loc[df['Gender'].str.contains('ish',case=False,na=False),'Gender']=-1
df.loc[df['Gender'].str.contains('m',case=False,na=False),'Gender']=-2
df.loc[df['Gender'].str.contains('',na=False),'Gender']=0

#preserve order in company size
df.loc[df['no_employees']=='1-5',['no_employees']]=1
df.loc[df['no_employees']=='6-25',['no_employees']]=2
df.loc[df['no_employees']=='26-100',['no_employees']]=3
df.loc[df['no_employees']=='100-500',['no_employees']]=4
df.loc[df['no_employees']=='500-1000',['no_employees']]=5
df.loc[df['no_employees']=='More than 1000',['no_employees']]=6

# Feature selection
drop_elements = ['Timestamp','Country','state','work_interfere']#work interfere goes because by defnition, if it inteferes with your work, then you definitely have a mental health issue
df = df.drop(drop_elements, axis = 1)

#Set up features and dependent variable
X = df.drop(['treatment'],axis=1)
y = df['treatment']
y = le.fit_transform(y) # yes:1 no:0

#Split TRAIN/TEST data
from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = \
        train_test_split(X, y, test_size=0.20, random_state=1)
