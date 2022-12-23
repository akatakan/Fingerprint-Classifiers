import pandas as pd
from sklearn.model_selection import KFold
from sklearn.preprocessing import LabelEncoder
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from sklearn.preprocessing import StandardScaler



num_features=3302

#Etikleleri encodelar
le=LabelEncoder()

#Veri setini kümlere ayırır
kfold=KFold(n_splits=5)

#Sayısal değerleri ölçeklendirir
sc = StandardScaler()

dataset = pd.read_csv("otu.csv",encoding="utf8",dtype="unicode")

dataset = dataset.T

#Sıralı veriyi karıştırır
dataset = dataset.sample(frac = 1).reset_index(drop=True)

X= dataset.iloc[:, 1:].astype("float64").to_numpy()
y = le.fit_transform(dataset.iloc[:, 0])

X=sc.fit_transform(X)

model = Sequential()

model.add(Dense(64,input_dim=num_features))
model.add(Dense(128,activation='relu'))
model.add(Dense(64,activation='relu'))
model.add(Dense(1,activation='sigmoid')) # Sigmoid ikili sınıflandırma için kullanılan bir aktivasyon fonksiyonudur.

model.compile(
    optimizer='adam',
    loss='binary_crossentropy', #Hata hesaplamasında yine ikili sınflandırma için kullanılan binary crossentropy hesaplamasını kullanıyoruz.
    metrics=['accuracy']    
)


accs =list()
losses = list()
for f,(train_idx,test_idx) in enumerate(kfold.split(X,y)):
    print("Fold ",f+1)
    X_train,X_test = X[train_idx],X[test_idx]
    y_train,y_test = y[train_idx],y[test_idx]

    print(f"Fold {f+1} training")
    model.fit(
        X_train,
        y_train,
        epochs=1,
        batch_size=32,
        validation_split=0.2
    )
    
    loss,accuracy = model.evaluate(X_test,y_test)

    print("Fold Acc: ",accuracy)
    accs.append(accuracy)
    losses.append(loss)


print("Mean Acc: ", sum(accs)/len(accs))
print("Mean Loss: ", sum(losses)/len(losses))


from sklearn.metrics import classification_report
y_pred = model.predict(X_test).round()
class_report = classification_report(y_test, y_pred)

print(class_report)
