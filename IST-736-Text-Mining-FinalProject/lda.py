from sklearn.decomposition import NMF, LatentDirichletAllocation, TruncatedSVD
from sklearn.feature_extraction.text import CountVectorizer 



import re 
import pandas as pd

import pyLDAvis.sklearn as LDAvis
import pyLDAvis



#import vectorizing functions
from vectorize import count_vectorize, tfidf_vectorize

#enable notebook
pyLDAvis.enable_notebook()

#read in the excel file from where you saved it
xls = pd.ExcelFile(r"C:\Users\Charlie Vanleuvan\OneDrive - Syracuse University\IST 736 Text Mining\final_project\lyrics_dataframe.xlsx")

#create a data frame from the excel file
df = pd.read_excel(xls, sheet_name="in")

#choose number of topics
num_topics = 20

#[0] is the dataframe, [1] is the vectors, [2] is the vectorizer instance
vectorized_data = count_vectorize(df)
tf_idf_data = tfidf_vectorize(df)

lda_model = LatentDirichletAllocation(n_components=num_topics, max_iter = 10, learning_method='online')

#use either count vectorized or tfidf vectorized
lda_Z = lda_model.fit_transform(vectorized_data[1])
lda_Z = lda_model.fit_transform(tf_idf_data[1])

#visualize
panel = LDAvis.prepare(lda_model, vectorized_data[1], vectorized_data[2], mds='tsne')
pyLDAvis.display(panel)

#visualize tfidf
panel = LDAvis.prepare(lda_model, tf_idf_data[1], tf_idf_data[2], mds='tsne')
pyLDAvis.display(panel)

#create a function to print the top topics and relevant terms
def print_topics(model, vectorizer, top_n = 5):
    for idx, topic in enumerate(model.components_):
        print("Topic %d:" % (idx))
        print([(vectorizer.get_feature_names()[i], topic[i])
                        for i in topic.argsort()[:-top_n - 1:-1]])

print("LDA Model:")
print_topics(lda_model, vectorized_data[2])




