from vectorize import count_vectorize, tfidf_vectorize

from sklearn.cluster import KMeans
import numpy as np
import sklearn
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D



#create df
#read in the excel file from where you saved it
xls = pd.ExcelFile(r"C:\Users\Charlie Vanleuvan\OneDrive - Syracuse University\IST 736 Text Mining\final_project\lyrics_dataframe.xlsx")

#create a data frame from the excel file
df = pd.read_excel(xls, sheet_name="in")


#[0] is the dataframe, [1] is the vectors, [2] is the vectorizer instance
vectorized_data = count_vectorize(df)




####VISUALIZE KMeans via elbow method #####

kmeans_object_Count = KMeans(n_clusters=4)
#print(kmeans_object)
kmeans_object_Count.fit(vectorized_data[0])
# Get cluster assignment labels
labels = kmeans_object_Count.labels_
prediction_kmeans = kmeans_object_Count.predict(vectorized_data[0])
#print(labels)
#print(prediction_kmeans)
# Format results as a DataFrame
Myresults = pd.DataFrame([vectorized_data[0].index,labels]).T


x=vectorized_data[0]["money"]  ## col 1  starting from 0
y=vectorized_data[0]["cars"]    ## col 14  starting from 0
z=vectorized_data[0]["street"]  ## col 2  starting from 0

colnames=vectorized_data[0].columns

#print(x,y,z)
fig1 = plt.figure(figsize=(12, 12))
ax1 = Axes3D(fig1, rect=[0, 0, .90, 1], elev=48, azim=134)

ax1.scatter(x,y,z, cmap="RdYlGn", edgecolor='k', s=200,c=prediction_kmeans)
ax1.w_xaxis.set_ticklabels([])
ax1.w_yaxis.set_ticklabels([])
ax1.w_zaxis.set_ticklabels([])

ax1.set_xlabel('money', fontsize=25)
ax1.set_ylabel('cars', fontsize=25)
ax1.set_zlabel('street', fontsize=25)
#plt.show()
       
centers = kmeans_object_Count.cluster_centers_
#print(centers)
#print(centers)
C1=centers[0,(1,2,14)]
#print(C1)
C2=centers[1,(1,2,14)]
#print(C2)
xs=C1[0],C2[0]
#print(xs)
ys=C1[1],C2[1]
zs=C1[2],C2[2]


ax1.scatter(xs,ys,zs, c='black', s=2000, alpha=0.2)
plt.show()


#define custom elbow plot function
def drawSSEPlot(df, column_indices, n_clusters = 10, max_iter = 300,tol = 1e-04, init = 'k-means++', n_init = 10, algorithm  ='auto'):
    inertia_values = []
    for i in range(1, n_clusters+1):
        km=KMeans(n_clusters = i, max_iter = max_iter, tol=tol, init=init, n_init = n_init, random_state=1, algorithm=algorithm)
        km.fit_predict(df.iloc[:,column_indices])
        inertia_values.append(km.inertia_)
    fig,ax = plt.subplots(figsize=(8,6))
    plt.plot(range(1,n_clusters+1), inertia_values,color='green')
    plt.xlabel('No. of Clusters',fontsize=15)
    plt.ylabel('Sum of Squared Errors',fontsize = 15)
    plt.title('SSE vs No. of Clusters', fontsize = 15)
    plt.grid()
    plt.show()



#plot the count vectorized data
drawSSEPlot(vectorized_data[0], list(range(0,len(vectorized_data[0]) +1)))

#try with tfidf
tf_df = tfidf_vectorize(df)[0]
drawSSEPlot(tf_df, list(range(0,len(tf_df) +1)))