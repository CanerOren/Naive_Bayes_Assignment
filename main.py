import numpy as np
import pandas as pd
import category_encoders as ce
from sklearn.preprocessing import RobustScaler
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import confusion_matrix




df_train=pd.read_csv('trainSet.csv',',',usecols=['credit_history','credit_amount','employment','property_magnitude','age','class'])
df_train.replace('?',np.NaN,inplace=True)#missing valuelar ile ilgili yapılacak işlemlerin kolaylığı için
#Gerekli Değerler Inte Çevrildi
df_train['age']=df_train['age'].astype('Int64')
df_train['credit_amount']=df_train['credit_amount'].astype('Int64')
#Missing Valuelar Dolduruldu
df_train['credit_amount'].replace([np.NaN],round(df_train['credit_amount'].mean()),inplace=True)
df_train['age'].replace([np.NaN],round(df_train['age'].mean()),inplace=True)
df_train['credit_history'].replace([np.NaN],df_train['credit_history'].mode(),inplace=True)
df_train['employment'].replace([np.NaN],df_train['employment'].mode(),inplace=True)
df_train['property_magnitude'].replace([np.NaN],df_train['property_magnitude'].mode(),inplace=True)
train_values=df_train.drop(['class'],axis=1)
train_class=df_train['class']


df_test=pd.read_csv('testSet.csv',',',usecols=['credit_history','credit_amount','employment','property_magnitude','age','class'])
df_test.replace('?',np.NaN,inplace=True)#missing valuelar ile ilgili yapılacak işlemlerin kolaylığı için
#Bütün sütunlar string olarak gelmişti gerekliler inte çevrildi
df_test['age']=df_test['age'].astype('Int64')
df_test['credit_amount']=df_test['credit_amount'].astype('Int64')
#Missing valueler kontrol edildi
df_test['credit_amount'].replace([np.NaN],round(df_test['credit_amount'].mean()),inplace=True)
df_test['age'].replace([np.NaN],round(df_test['age'].mean()),inplace=True)
df_test['credit_history'].replace([np.NaN],df_test['credit_history'].mode(),inplace=True)
df_test['employment'].replace([np.NaN],df_test['employment'].mode(),inplace=True)
df_test['property_magnitude'].replace([np.NaN],df_test['property_magnitude'].mode(),inplace=True)
test_values=df_test.drop(['class'],axis=1)
test_class=df_test['class']

#Veri Gaussian Naive Bayes için hazırlandı ve belirli bir aralığa yerleştirildi
encoder=ce.OneHotEncoder(cols=['credit_history','employment','property_magnitude'])
train_values=encoder.fit_transform(train_values)
test_values=encoder.fit_transform(test_values)

cols=train_values.columns
scaler=RobustScaler()
train_values=scaler.fit_transform(train_values)
test_values=scaler.fit_transform(test_values)

train_values=pd.DataFrame(train_values,columns=[cols])
test_values=pd.DataFrame(test_values,columns=[cols])

#Gaussian Naive Bayes Uygulandı
gnb= GaussianNB()

gnb.fit(train_values,train_class)

pred=gnb.predict(test_values)

#True Positive,True Negative,False Positive ,False Negative bulundu ve hesaplandı
test_class_df=pd.DataFrame(test_class)
pred_df=pd.DataFrame(pred)
test_class_list=test_class_df.values.tolist()
pred_list=pred_df.values.tolist()
true_positive=0
true_negative=0
false_negative=0
false_positive=0


for i in range(0,250):
    if((pred_list[i]==['good'])and(test_class_list[i]==['good'])):
        true_positive=true_positive+1
    elif((pred_list[i]==['bad'])and(test_class_list[i]==['bad'])):
        true_negative=true_negative+1
    elif((pred_list[i]==['bad'])and(test_class_list[i]==['good'])):
        false_positive=false_positive+1
    elif((pred_list[i]==['good'])and(test_class_list[i]==['bad'])):
        false_negative=false_negative+1


acc=float((true_positive+true_negative)/(true_positive+true_negative+false_positive+false_negative))
tp_rate=float(true_positive/(true_positive+false_negative))
tn_rate=float(true_negative/(true_negative+false_positive))
print("Accuracy (Doğruluk):",acc)
print("TP rate (Gerçek Doğruluk Oranı):",tp_rate)
print("TN rate(Gerçek Negatif Oranı):",tn_rate)
print("TP (Gerçek Pozitif) adedi:",true_positive)
print("TN (Gerçek Negatif):adedi",true_negative)




