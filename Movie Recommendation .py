#!/usr/bin/env python
# coding: utf-8

# In[156]:


import pandas as pd
import numpy as np
import difflib
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity


# In[157]:


df = pd.read_csv(r"C:\Users\hplap\Downloads\movies.csv")


# In[158]:


df.head()


# In[159]:


df.info()


# In[160]:


df.isnull().sum()


# In[161]:


df.describe()


# In[162]:


df.shape


# In[163]:


df['popularity'].value_counts()


# In[164]:


# Selecting the relevent features
selected_fea = ['genres','keywords','tagline','cast','director']
print(selected_fea)


# In[165]:


#Replacing the missing the non-values
for feature in selected_fea:
    df[feature] = df[feature].fillna('')


# In[166]:


#combine the all the file selected feature 
combine = df['genres']+' '+df['keywords']+' '+df['tagline']+' '+df['cast']+' '+df['director']


# In[167]:


print(combine)


# In[168]:


#convert the text data to feature vector
vector = TfidfVectorizer()


# In[169]:


vector_feature = vector.fit_transform(combine)


# In[124]:


print(vector_feature)


# # Cosine Similarity

# In[125]:


similarity = cosine_similarity(vector_feature)


# In[126]:


print(similarity)


# In[127]:


print(similarity.shape)


# In[142]:


# Getting the movie name from the user
movie_name = input('Enter your favourite movie name : ')


# In[143]:


#Creating a list of all the movies name
list_of_all_title = df['title'].tolist()
print(list_of_all_title)


# In[144]:


# finding the close match for the movie name given by user 
find_the_close_match = difflib.get_close_matches(movie_name,list_of_all_title)
print(find_the_close_match)


# In[145]:


close_match = find_the_close_match[0]


# In[146]:


close_match


# In[147]:


#Finding the index movie list in the title
index_of_movie = df[df.title==close_match]['index'].values[0]


# In[148]:


index_of_movie


# In[149]:


# geeting a list of a similar movies
similarity_score = list(enumerate(similarity[index_of_movie]))


# In[150]:


print(similarity_score)


# In[151]:


len(similarity_score)


# In[152]:


#sorting the movies the based on the similarity score
sorted_of_movies = sorted(similarity_score, key = lambda x:x[1], reverse=True)


# In[153]:


print(sorted_of_movies)


# In[170]:


#print the name of similar movies based on the index
print("Movies suggested for you : \n")

i = 1
for movie in sorted_of_movies:
    index = movie[0]
    title_from_index = df[df.index==index]['title'].values[0]
    if(i<31):
        print(i, ':',title_from_index)
        i+=1


# # Movie recommendation system

# In[171]:


movie_name = input('Enter your favourite movie name : ')

list_of_all_title = df['title'].tolist()

find_the_close_match = difflib.get_close_matches(movie_name,list_of_all_title)

close_match = find_the_close_match[0]


index_of_movie = df[df.title==close_match]['index'].values[0]



similarity_score = list(enumerate(similarity[index_of_movie]))


sorted_of_movies = sorted(similarity_score, key = lambda x:x[1], reverse=True)


print("Movies suggested for you : \n")

i = 1
for movie in sorted_of_movies:
    index = movie[0]
    title_from_index = df[df.index==index]['title'].values[0]
    if(i<31):
        print(i, ':',title_from_index)
        i+=1


# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:




