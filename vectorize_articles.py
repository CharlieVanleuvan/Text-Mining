import requests 
import os 
import re
import pandas as pd 
from time import sleep 
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer



#create a function to retrieve article headlines and then save to corpus folder
def get_articles(search):
    """A function that retrieves articles about the search term, and saves them locally to a corpus folder"""

    #api key from NY Times developers portal. Need to create an account on their website
    api_key = "ENTER-YOUR-API-KEY-HERE"

    #lazily assign argument to string in case int is supplied. Further development would include try/except block here
    search = str(search)

    #make API call to retrieve articles. NY times API defaults to max 10 articles which works for this project.
    articles = requests.get("https://api.nytimes.com/svc/search/v2/articlesearch.json?q={}&api-key={}".format(search,api_key))

    #create json object from API return. The 'docs' key has a list of dictionaries
    articles_json = articles.json()['response']['docs']

    #loop through the retrieved articles and return the 'lead_paragraph' text
    counter = 1 #counter for number of articles. will be used to name the txt files
    for text in articles_json:
        words = text['lead_paragraph']
        file_path = r"C:\your\file\path\here\{}{}.txt".format(search,str(counter))
        with open(file_path,'w', encoding='utf-8') as f:
            f.write(words)
        counter += 1
    
    return(print("Number of files created: {}".format(str(counter))))



#create a function that will vectorize .txt files in a corpus stored locally
#inputs: file path, choose TFIDF or countvectorizer
def vectorizer(vectorizer):
    #requires sklearn CountVectorizer, TfidfVectorizer, pandas installed, os installed
    #take file path
    path = input("Enter the file path to the corpus: ")
    
    #create list of file names
    file_name_list = os.listdir(path)
    
    #use list comprehension to get list of file paths and list of labels
    full_paths = [path + '\\' + name for name in file_name_list]
    labels = ["".join(re.findall("[a-zA-z]+",file.split('.')[0])) for file in file_name_list]
    
    if vectorizer == 'CountVectorizer':
        #instantiate count vectorizer
        countVect = CountVectorizer(input = 'filename',stop_words = 'english')
        
        #vectorize the files in the full file path list
        vectors = countVect.fit_transform(full_paths)
        
        #get column names as features for the words used in the docs
        columnnames = countVect.get_feature_names()
        
        #create pandas DF
        df = pd.DataFrame(vectors.toarray(), columns = columnnames)
        
        #add labels to dataframe
        df['Target_Label'] = None
        for i in zip(range(0,len(df)+1),labels):
            df.loc[i[0],'Target_Label'] = i[1]
            
    elif vectorizer == 'TfidfVectorizer':
        #instantiate count vectorizer
        tfidfVect = TfidfVectorizer(input = 'filename',stop_words = 'english')
        
        #vectorize the files in the full file path list
        vectors = tfidfVect.fit_transform(full_paths)
        
        #get column names as features for the words used in the docs
        columnnames = tfidfVect.get_feature_names()
        
        #create pandas DF
        df = pd.DataFrame(vectors.toarray(), columns = columnnames)
        
        #add labels to dataframe
        df['Target_Label'] = None
        for i in zip(range(0,len(df)+1),labels):
            df.loc[i[0],'Target_Label'] = i[1]
        
    else:
        print("Error with vectorizer name. Try again.")
        
    return(df)

if __name__ == "__main__":
    #collect the articles first. sleep 6 seconds to avoid getting limited
    get_articles('Russia')
    sleep(6)
    get_articles('China')

    #vectorize the corpus
    cv_df = vectorizer('CountVectorizer')
    tfidf_df = vectorizer('TfidfVectorizer')

    #print the dataframes
    print(cv_df)
    print('\n')
    print('\n')
    print(tfidf_df)

    #save to xlsx in folder
    cv_df.to_excel(r"file-path-to-save-to\file_name_to_save_as.xlsx", index = False)
    tfidf_df.to_excel(r"file-path-to-save-to\file_name_to_save_as.xlsx", index = False)
