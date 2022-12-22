import pandas as pd
from sklearn.model_selection import KFold,cross_val_score,train_test_split
from sklearn.preprocessing import LabelEncoder,StandardScaler
from lazypredict.Supervised import LazyClassifier
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.utils import shuffle


num_features=3302
le=LabelEncoder()

#Veri setini kaç kümeye ayıracağımızı belirtiyoruz
kfold=KFold(n_splits=5)

#Sayıları 0 ile 1 arasına sığdırmak için
sc = StandardScaler()
rfc = RandomForestClassifier(n_estimators=50)

dataset = pd.read_csv("otu.csv",encoding="utf8",dtype="unicode")

dataset = dataset.T

#Veri setimiz sıralı olduğu için burada karıştırıyoruz.
dataset=shuffle(dataset)

#Pandas dataframe'ini numpy array'e çevirdim
X= dataset.iloc[:, 1:].astype("float64").to_numpy()

#Etiketlerimi encodeladım
y = le.fit_transform(dataset.iloc[:, 0])

#Sayıları 0 ile 1 arasına sığdırdım
X=sc.fit_transform(X)

#Cross Validation Feature Selection
accs =list()
for f,(train_idx,test_idx) in enumerate(kfold.split(X,y)):
    print("Fold ",f+1)
    X_train,X_test = X[train_idx],X[test_idx]
    y_train,y_test = y[train_idx],y[test_idx]
    
    rfc.fit(X_train,y_train)
    acc = rfc.score(X_test,y_test)
    print("Fold Acc: ",acc)
    accs.append(acc)
    

#Her kümeden gelen accuracy değerlerinin ortalamasını aldım
print("Mean Acc: ", sum(accs)/len(accs))
