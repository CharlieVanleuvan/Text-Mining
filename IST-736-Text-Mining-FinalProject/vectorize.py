import pandas as pd 
import re 
from sklearn.feature_extraction.text import CountVectorizer,TfidfVectorizer

def binary_vectorize(data):
    """A function that vectorizes a dataframe of tweets using the Binary = True paramter in CountVectorizer """
    

    #read in the supplied dataframe
    df = data

    #remove any non ASCII characters from the lyrics text
    df['Lyrics'] = df['Lyrics'].apply(lambda x: re.compile(r"[^\x00-\x7F]+").sub('',x))

    #remove any leading or trailing white space in the lyrics text
    df['Lyrics'] = df['Lyrics'].apply(lambda x: x.strip())

    #Instantiate CountVectorizer, Binary = True
    vectorizer = CountVectorizer(input = 'content', stop_words = 'english', binary = True)

    #create vectors
    vectors = vectorizer.fit_transform(df['Lyrics'].tolist())

    #get column names
    columnnames = vectorizer.get_feature_names()

    #create new dataframe for the reviews and labels
    clean_df = pd.DataFrame(vectors.toarray(), columns = columnnames)

    #add labels to the dataframe. These labels are the State or Region for each song from the lyrics excel file
    clean_df['LABEL'] = None 
    for i in zip(range(0,len(clean_df)+1), df['State']):
        clean_df.loc[i[0],'LABEL'] = i[1]

    #reorder the columns so LABEL column is first
    cols = list(clean_df)
    cols.insert(0, cols.pop(cols.index('LABEL')))
    clean_df  = clean_df.loc[:, cols]

    #save the binary vectorized data set
    clean_df.to_csv(r"C:\Users\Charlie Vanleuvan\OneDrive - Syracuse University\IST 736 Text Mining\final_project\binarized.csv",index = False)
    
    #return the vectorized dataframe
    return(clean_df)

def count_vectorize(data):
    """A function that vectorizes a dataframe of tweets using the term frequency with CountVectorizer """    

    #read in the supplied dataframe
    df = data

    #remove any non ASCII characters from the lyrics text
    df['Lyrics'] = df['Lyrics'].apply(lambda x: re.compile(r"[^\x00-\x7F]+").sub('',x))

    #remove any leading or trailing white space
    df['Lyrics'] = df['Lyrics'].apply(lambda x: x.strip())

    #Instantiate CountVectorizer
    vectorizer = CountVectorizer(input = 'content', stop_words = 'english')

    #create vectors
    vectors = vectorizer.fit_transform(df['Lyrics'].tolist())

    #get column names
    columnnames = vectorizer.get_feature_names()

    #create new dataframe for the reviews and labels
    clean_df = pd.DataFrame(vectors.toarray(), columns = columnnames)
    
    """
    #add labels to the dataframe. These labels are the State or Region for each song from the lyrics excel file
    clean_df['LABEL'] = None 
    for i in zip(range(0,len(clean_df)+1), df['State']):
        clean_df.loc[i[0],'LABEL'] = i[1]

    #reorder the columns so LABEL column is first
    cols = list(clean_df)
    cols.insert(0, cols.pop(cols.index('LABEL')))
    clean_df  = clean_df.loc[:, cols]
    
    """
    #save the count vectorized data set
    clean_df.to_csv(r"C:\Users\Charlie Vanleuvan\OneDrive - Syracuse University\IST 736 Text Mining\final_project\term_frequency_vectorized.csv",index = False)
    
    return(clean_df, vectors, vectorizer)

def tfidf_vectorize(data):
    """A function that vectorizes a dataframe of tweets using the term frequency-Inverse document frequency with TfIDFVectorizer """    

    #read in the supplied dataframe
    df = data

    #remove any non ASCII characters from the lyrics text
    df['Lyrics'] = df['Lyrics'].apply(lambda x: re.compile(r"[^\x00-\x7F]+").sub('',x))

    #remove any leading or trailing white space
    df['Lyrics'] = df['Lyrics'].apply(lambda x: x.strip())

    #Instantiate CountVectorizer
    vectorizer = TfidfVectorizer(input = 'content', stop_words = 'english')

    #create vectors
    vectors = vectorizer.fit_transform(df['Lyrics'].tolist())

    #get column names
    columnnames = vectorizer.get_feature_names()

    #create new dataframe for the reviews and labels
    clean_df = pd.DataFrame(vectors.toarray(), columns = columnnames)

    """ 
    #add labels to the dataframe. These labels are the State or Region for each song from the lyrics excel file
    clean_df['LABEL'] = None 
    for i in zip(range(0,len(clean_df)+1), df['State']):
        clean_df.loc[i[0],'LABEL'] = i[1]

    #reorder the columns so LABEL column is first
    cols = list(clean_df)
    cols.insert(0, cols.pop(cols.index('LABEL')))
    clean_df  = clean_df.loc[:, cols]

    """
    #save the tf-idf vectorized data set
    clean_df.to_csv(r"C:\Users\Charlie Vanleuvan\OneDrive - Syracuse University\IST 736 Text Mining\final_project\tf_idf_vectorized.csv",index = False)

    return(clean_df, vectors, vectorizer)

if __name__ == "__main__":
    #read in the excel file from where you saved it
    xls = pd.ExcelFile(r"C:\Users\Charlie Vanleuvan\OneDrive - Syracuse University\IST 736 Text Mining\final_project\lyrics_dataframe.xlsx")

    #create a data frame from the excel file
    df = pd.read_excel(xls, sheet_name="in")

    #run the vectorizing functions 
    binary_vectorize(df)
    count_vectorize(df)
    tfidf_vectorize(df)