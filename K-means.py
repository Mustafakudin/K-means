# -*- coding: utf-8 -*-
"""
Created on Mon Jun 19 20:37:21 2023

@author: Mustafa
"""

from warnings import filterwarnings
filterwarnings("ignore"),
import numpy as np
import pandas as pd 
import seaborn as sns
import matplotlib.pyplot as plt
import scipy as sp
from sklearn.cluster import KMeans

df=pd.read_csv("C:/Users/Mustafa/makine_ogrenmesi/K-means/USArrests.csv")

df.head(10)

df.index=df.iloc[:,0]
df.index

df = df.iloc[:,1:5]

df.index.name=""
##print(df)

print(df.isnull().sum())
print(df.info)
print(df.describe().T)

#df.hist(figsize=(10,10));

from sklearn.cluster import KMeans
kmeans = KMeans(n_clusters=4)
k_fit = kmeans.fit(df)
print(k_fit.cluster_centers_) ## merkezlerine ulaşmak için 
print(k_fit.labels_) ## herbir gözlemin sınıf etiketini class etiketinin sahip oldugunu veriyor clasların hangi eyaletin hangi clastur aldık onu ögrendiiyoruz 


##Görselleştirme  2 boyutta bakalım öncelikle 

kmeans = KMeans(n_clusters=2)
k_fit=kmeans.fit(df)
kumeler = k_fit.labels_
plt.scatter(df.iloc[:,0],df.iloc[:,1],c=kumeler ,s=50,cmap="viridis")
plt.show()
merkezler = k_fit.cluster_centers_
plt.scatter(merkezler[:,0],merkezler[:,1], c="Black" , s=200 ,alpha =0.5)
plt.show()
from mpl_toolkits.mplot3d import Axes3D
kmeans=KMeans(n_clusters=3)
k_fit=kmeans.fit(df)
kumeler = k_fit.labels_
merkezler=k_fit.cluster_centers_

plt.rcParams["figure.figsize" ] = (16,9)
fig=plt.figure()
ax=Axes3D(fig)
ax.scatter(df.iloc[:,0],df.iloc[:,1],df.iloc[:,2])


## Merkezleri görmek için 

fig=plt.figure()
ax=Axes3D(fig)
ax.scatter(df.iloc[:,0],df.iloc[:,1],df.iloc[:,2],c=kumeler)
ax.scatter(merkezler[:,0],merkezler[:,1],merkezler[:,2],marker="*",c="#050505",s=1000)
plt.show()

##Kumeler ve gözlem birimleri 
kmeans=KMeans(n_clusters=3)
k_fit=kmeans.fit(df)
kumeler=k_fit.labels_
print(pd.DataFrame({"Eyaletler" : df.index,"Kumeler" : kumeler }))

#verilerimize kümeleri eklemek istersek 
df["Kume no "] =kumeler
print(df.head(10))

## kume noda misal 0 olmasını istemezsek onun ıcınde 
##df["Kume no "] = df["Kume no "] +1
##print(df)


##Bu da clusters belirlemek için wcss değerlerini görmek için 
sonuclar=[]
for i in range(1,11):
    kmeans=KMeans(n_clusters=i,init='k-means++' , random_state=123)
    kmeans.fit(df)
    sonuclar.append(kmeans.inertia_)  ## wcss değerlerii
plt.plot(range(1,11), sonuclar)  ## 1,10 kadar değerler alsın , sonucları cizdirsin
plt.show()

'''
## Optimum Kume sayısını belirlenmesi   burada  k yi n_cluster kaç secmemiz gerekiyor gibi sorunun cevabı bu 
from  yellowbrick.cluster import KElbowVisualizer
kmeans=KMeans()
visualizer = KElbowVisualizer(kmeans, k=(2,20))
visualizer.fit(df)
visualizer.poof() ## BURADA GRAFİKTE BİZE 4 CLUSTERS TAN OLUSMASI DAHA MAKUL OLMASI GEREKTİGİNİ SOYLUYOR 
'''
##Tekrardan bu bilgilere göre yazarsal 
kmeans=KMeans(n_clusters=4)
k_fit=kmeans.fit(df)
kumeler=k_fit.labels_
dff =pd.DataFrame({"Eyaletler" : df.index , "Kumeler" : kumeler})

print(dff)




