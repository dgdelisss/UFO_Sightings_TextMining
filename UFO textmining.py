# -*- coding: utf-8 -*-
"""TextMining.ipynb

Automatically generated by Colaboratory.

Original file is located at
    https://colab.research.google.com/drive/1SE4x7FHtfmFnvx6MaFnCHmffCe9SqcLE

**Project Description:**
We will analyze data on reported incidents of UFO sightings. Utilizing data collected by an organization dedicated to this topic, we will apply topic clustering techniques to identify commonalities among these sightings and interpret the results to provide a summary of the major themes of these reports. After clustering among the full dataset, we will then focus on comparing UFO sightings in California, Arizona, and Nevada again using clustering to investigate their similarities and differences.  

**Analysis: **
We will perform topic clustering on the text column from our dataset to identify major topics of discussion. We will then use this clustering to analyze any commonalities or anomalies based on descriptors of UFO shape, size, etc. We’ll start with a cluster analysis of the full dataset, and then narrow the focus to comparing sightings exclusively in California, Nevada, and Arizona.

**Deliverables: **
We will provide the following deliverables at the end of the project:
A dataset containing reports of UFO sightings
A set of insights derived from the dataset
A short in-class presentation of our findings, discussions of their meaning, and general “lessons learned” from our project.

# Packages and Installations:
"""

#installs any packages not available by default
#!pip install gensim
#!pip install wordcloud
# %time

#importing packages neeeded for Text Analysis
import pandas as pd
import numpy as np
import nltk
import sklearn
import gensim
import re
import string
#import wordcloud
import os
# %time

##Specific Text Mining Features from SKLEARN
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import TruncatedSVD
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import Normalizer
from sklearn.neighbors import KNeighborsClassifier
from sklearn.cluster import KMeans
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.manifold import MDS
from sklearn.metrics.pairwise import cosine_similarity
#Other specific useful packages
#from wordcloud import WordCloud
from collections import Counter
from nltk.stem import WordNetLemmatizer
from nltk.corpus import wordnet as wn
import matplotlib.pyplot as plt
import matplotlib as mpl
# %time

#Downloading features from nltk
nltk.download('stopwords')
nltk.download('punkt')
nltk.download('wordnet')
# %time

"""# User Defined Functions:"""

#Flatten Function (This will collapse a list of lists into just one list)
flatten = lambda l: [item for sublist in l for item in sublist]

#Stringer

def Stringer(list):
  new_list = []
  for i in list:
    new = str(i)
    new_list.append(new)
  return(new_list)

#Term Vector Function
def Term_Vectors(doc):
  punc = re.compile( '[%s]' % re.escape( string.punctuation ) )
  term_vec = [ ]

  for d in doc:
      d = str(d)
      d = d.lower()
      d = punc.sub( '', d )
      term_vec.append( nltk.word_tokenize( d ) )

  return(term_vec)

#Stop Word Function
def Stop_Word(term_vec, stop_words = nltk.corpus.stopwords.words( 'english' )):

  for i in range( 0, len( term_vec ) ):
      
      term_list = [ ]

      for term in term_vec[i]:
          if term not in stop_words:
              term_list.append( term )

      term_vec[i] = term_list

  return(term_vec)

#Porter Stem Function

def Porter_Stem(term_vec):
  porter = nltk.stem.porter.PorterStemmer()

  for i in range( 0, len( term_vec ) ):
    for j in range( 0, len( term_vec[ i ] ) ):
      term_vec[ i ][ j ] = porter.stem( term_vec[ i ][ j ] )

  return(term_vec)

#Lemmatizer Function
def lemmatizer(term_vec):
  for i in range( 0, len( term_vec ) ):
    for j in range( 0, len( term_vec[ i ] ) ):
      try: pos = str(wn.synsets(j)[0].pos())
      except: pos = "n"
      term_vec[i][j] = str(WordNetLemmatizer().lemmatize(term_vec[i][j],pos))
  return(term_vec)

##Basic Word Cloud Function

#def show_wordcloud(data, title = None):
#    wordcloud = WordCloud(
#        background_color='white',
#        max_words=50,
#        max_font_size=40, 
#        scale=3,
#        random_state=1 # chosen at random by flipping a coin; it was heads
#    ).generate(str(data))
#
#    fig = plt.figure(1, figsize=(12, 12))
#    plt.axis('off')
#    if title: 
#        fig.suptitle(title, fontsize=20)
#        fig.subplots_adjust(top=2.3)
#
#    plt.imshow(wordcloud)
#    plt.show()

"""# Initial Data Importation and Cleaning:"""

#imports ufo dataset from our data.world repo
ufoset = pd.read_csv('https://query.data.world/s/t5l7slkbhurybmuxkfgncobbaknf7i')

#subsets data by selected states, removes every column but State and Text
states = ["CA","NV","AR","NM", "NC"]
subset_ufoset = ufoset.loc[ufoset['state'].isin(states)]

encounters = subset_ufoset[['text','state']]

#New datasets for each state
#CA_encounters = encounters.loc[ufoset['state'] == "CA"]
#NV_encounters = encounters.loc[ufoset['state'] == "NV"]
#AR_encounters = encounters.loc[ufoset['state'] == "AR"]
#NM_encounters = encounters.loc[ufoset['state'] == "NM"]
#NC_encounters = encounters.loc[ufoset['state'] == "NC"]

#Word Vectors
#All_States = ufoset['text'].values.tolist()
SelectStates_vect = encounters['text'].values.tolist()
#CA_vect = CA_encounters['text'].values.tolist()
#NV_vect = NV_encounters['text'].values.tolist()
#AR_vect = AR_encounters['text'].values.tolist()
#NM_vect = NM_encounters['text'].values.tolist()
#NC_vect = NC_encounters['text'].values.tolist()

print("Lists created.")
# %time

"""# Begin Text Processing with Term Vectors, Stopwords, and Stemming:"""

#Creates Term Vectors for all word vectors

#All_term = Term_Vectors(All_States)
SelectStates_term = Term_Vectors(SelectStates_vect)
#CA_term = Term_Vectors(CA_vect)
#NV_term = Term_Vectors(NV_vect)
#AR_term =Term_Vectors(AR_vect)
#NM_term =Term_Vectors(NM_vect)
#NC_term =Term_Vectors(NC_vect)

print("Term Vectors  Complete.")
# %time

stopword = nltk.corpus.stopwords.words('english')
custom_words = ['summary','SUMMARY']
stopword += custom_words

print("Stop Words Created.")
# %time

#Stop Word filter for all Vectors
#All_stop = Stop_Word(All_term,stopword)
SelectStates_stop = Stop_Word(SelectStates_term,stopword)
#CA_stop = Stop_Word(CA_term,stopword)
#NV_stop = Stop_Word(NV_term,stopword)
#AR_stop = Stop_Word(AR_term,stopword)
#NM_stop = Stop_Word(NM_term,stopword)
#NC_stop = Stop_Word(NC_term,stopword)

print("Stop Words filter Applied to Term Vectors.")
# %time

#Lemmatizing for All Vectors
#Results look way cleaner than porter stemming

#All_lem = lemmatizer(All_stop)
SelectStates_lem = lemmatizer(SelectStates_stop)
#CA_lem = lemmatizer(CA_stop)
#NV_lem = lemmatizer(NV_stop)
#AR_lem = lemmatizer(AR_stop)
#NM_lem = lemmatizer(NM_stop)
#NC_lem = lemmatizer(NC_stop)

print("Lemmatization Complete.")
# %time

#Will probably need to refilter the vectors after stemming - not sure how much filter terms are needed yet
nextfilter = ["'","-","look","saw","like","seen","see","could","would","also","got","said","seem","go","well","even"]

#All_filt = Stop_Word(All_lem,nextfilter)
SelectStates_filt = Stop_Word(SelectStates_lem,nextfilter)
#CA_filt = Stop_Word(CA_lem,nextfilter)
#NV_filt = Stop_Word(NV_lem,nextfilter)
#AR_filt = Stop_Word(AR_lem,nextfilter)
#NM_filt = Stop_Word(NM_lem,nextfilter)
#NC_filt = Stop_Word(NC_lem,nextfilter)

print("Text Filtering Complete")
# %time

#Setting Up Vocab Lists

Select_vdict = {'index': SelectStates_filt,'word': SelectStates_term}
Select_vocab = pd.DataFrame(Select_vdict)
Select_vocab = Select_vocab.set_index('index')

print('there are ' + str(Select_vocab.shape[0]) + ' items in Select_vocab')

print("Vocab Vectors Complete")
# %time



"""#tfidf Vectorization & K-Means Clustering"""

#All_tfidf = TfidfVectorizer(All_filt, max_features = 10000, decode_error = "replace", max_df=.9,min_df=.10)
SelectStates_tfidf = TfidfVectorizer(SelectStates_filt, max_features = 10000, decode_error = "replace", max_df=.9,min_df=.10)
#CA_tfidf = TfidfVectorizer(CA_filt, max_features = 10000, decode_error = "replace", max_df=.9,min_df=.10)
#NV_tfidf = TfidfVectorizer(NV_filt, max_features = 10000, decode_error = "replace", max_df=.9,min_df=.10)
#AR_tfidf = TfidfVectorizer(AR_filt, max_features = 10000, decode_error = "replace", max_df=.9,min_df=.10)
#NM_tfidf = TfidfVectorizer(NM_filt, max_features = 10000, decode_error = "replace", max_df=.9,min_df=.10)
#NC_tfidf = TfidfVectorizer(NC_filt, max_features = 10000, decode_error = "replace", max_df=.9,min_df=.10)

print("Tfidf Vectors Complete.")
# %time


##Document Similarity Matrices

#All_matrix = All_tfidf.fit_transform(ufoset['text'].values.astype('U'))
SelectStates_matrix = SelectStates_tfidf.fit_transform(encounters['text'].values.astype('U'))
#CA_matrix = CA_tfidf.fit_transform(CA_encounters['text'].values.astype('U'))
#NV_matrix = NV_tfidf.fit_transform(NV_encounters['text'].values.astype('U'))
#AR_matrix = AR_tfidf.fit_transform(AR_encounters['text'].values.astype('U'))
#NM_matrix = NM_tfidf.fit_transform(NM_encounters['text'].values.astype('U'))
#NC_matrix = NC_tfidf.fit_transform(NC_encounters['text'].values.astype('U'))


print("Similarity Matrices Complete.")
# %time

#Get term names
#All_terms = All_tfidf.get_feature_names()
select_terms = SelectStates_tfidf.get_feature_names()
#CA_terms = CA_tfidf.get_feature_names()
#NV_terms = NV_tfidf.get_feature_names()
#AR_terms = AR_tfidf.get_feature_names()
#NM_terms = NM_tfidf.get_feature_names()
#NC_terms = NC_tfidf.get_feature_names()

print("Term Names Complete.")
# %time

#Pairwise Similaritiy Distances Calculation

#All_dist = 1 - cosine_similarity(All_matrix)
SelectStates_dist = 1 - cosine_similarity(SelectStates_matrix)
#CA_dist = 1 - cosine_similarity(CA_matrix)
#NV_dist = 1 - cosine_similarity(NV_matrix)
#AR_dist = 1 - cosine_similarity(AR_matrix)
#NM_dist = 1 - cosine_similarity(NM_matrix)
#NC_dist = 1 - cosine_similarity(NC_matrix)

print("Pairwise Complete Distances Calculated")
# %time

## KMeans Clustering with n = 5

#All_kmeans = KMeans(n_clusters=5,random_state =0).fit(All_matrix)
SelectStates_kmeans = KMeans(n_clusters=5,random_state =0).fit(SelectStates_matrix)
#CA_kmeans = KMeans(n_clusters=5,random_state =0).fit(CA_matrix)
#NV_kmeans = KMeans(n_clusters=5,random_state =0).fit(NV_matrix)
#AR_kmeans = KMeans(n_clusters=5,random_state =0).fit(AR_matrix)
#NM_kmeans = KMeans(n_clusters=5,random_state =0).fit(NM_matrix)
#NC_kmeans = KMeans(n_clusters=5,random_state =0).fit(NC_matrix)

print("K-Means Clustering Complete")
# %time

#Get Cluster Labels

#All_States_clusters = All_kmeans.labels_.tolist()
SelectStates_clusters = SelectStates_kmeans.labels_.tolist()
#CA_clusters = CA_kmeans.labels_.tolist()
#NV_clusters = NV_kmeans.labels_.tolist()
#AR_clusters = AR_kmeans.labels_.tolist()
#NM_clusters = NM_kmeans.labels_.tolist()
#NC_clusters = NC_kmeans.labels_.tolist()

print("Cluster Labels Complete.")
# %time

Select_FlatTerms = flatten(select_terms)

Select_State = {'index': SelectStates_clusters,'clusters': SelectStates_clusters, 'State': encounters['state'], "Text":encounters['text'] }
Select_Frame = pd.DataFrame(Select_State)
Select_Frame = Select_Frame.set_index('index')

Select_Frame['clusters'].value_counts() #number of 'encounters' per cluster (clusters from 0 to 4)



print("Top terms per cluster:")
print("")
#sort cluster centers by proximity to centroid
order_centroids = SelectStates_kmeans.cluster_centers_.argsort()[:, ::-1] 

for i in range(5):
    print("Cluster words:", i, ":", end='')
    
    for ind in order_centroids[i, :5]: #replace 6 with n words per cluster
        
        #print(Select_vocab.iloc[select_terms[ind]], end =",")
        #print(list(Select_vocab.keys())[list(Select_vocab.values()).index(select_terms[ind])],end =",")
        print(select_terms[ind], end = ",")
        
print("")
print("")

#Multidimension Scaling

import os  # for os.path.basename

import matplotlib.pyplot as plt
import matplotlib as mpl

from sklearn.manifold import MDS

MDS()

# convert two components as we're plotting points in a two-dimensional plane
# "precomputed" because we provide a distance matrix
# we will also specify `random_state` so the plot is reproducible.
mds = MDS(n_components=2, dissimilarity="precomputed", random_state=1)

pos = mds.fit_transform(SelectStates_dist)  # shape (n_components, n_samples)

xs, ys = pos[:, 0], pos[:, 1]
print("")
print("")



#set up colors per clusters using a dict
cluster_colors = {0: '#1b9e77', 1: '#d95f02', 2: '#7570b3', 3: '#e7298a', 4: '#66a61e'}

#set up cluster names using a dict
cluster_names = {0: 'Family, home, war', 
                 1: 'Police, killed, murders', 
                 2: 'Father, New York, brothers', 
                 3: 'Dance, singing, love', 
                 4: 'Killed, soldiers, captain'}

#create data frame that has the result of the MDS plus the cluster numbers and titles
df = pd.DataFrame(dict(x=xs, y=ys, label=SelectStates_clusters, state=encounters['state'])) 

#group by cluster
groups = df.groupby('label')


# set up plot
fig, ax = plt.subplots(figsize=(17, 9)) # set size
ax.margins(0.05) # Optional, just adds 5% padding to the autoscaling

#iterate through groups to layer the plot
#note that I use the cluster_name and cluster_color dicts with the 'name' lookup to return the appropriate color/label
for name, group in groups:
    ax.plot(group.x, group.y, marker='o', linestyle='', ms=12, 
            label=cluster_names[name], color=cluster_colors[name], 
            mec='none')
    ax.set_aspect('auto')
    ax.tick_params(\
        axis= 'x',          # changes apply to the x-axis
        which='both',      # both major and minor ticks are affected
        bottom='off',      # ticks along the bottom edge are off
        top='off',         # ticks along the top edge are off
        labelbottom='off')
    ax.tick_params(\
        axis= 'y',         # changes apply to the y-axis
        which='both',      # both major and minor ticks are affected
        left='off',      # ticks along the bottom edge are off
        top='off',         # ticks along the top edge are off
        labelleft='off')
    
ax.legend(numpoints=1)  #show legend with only 1 point

#add label in x,y position with the label as the film title
for i in range(len(df)):
    ax.text(df.ix[i]['x'], df.ix[i]['y'], df.ix[i]['title'], size=8)  

    
    
plt.show() #show the plot

#uncomment the below to save the plot if need be
#plt.savefig('clusters_small_noaxes.png', dpi=200)




