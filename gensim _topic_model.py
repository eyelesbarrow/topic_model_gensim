
# coding: utf-8

# In[3]:


import nltk; nltk.download('stopwords')


# In[7]:


import re
import numpy as np
import pandas as pd
from pprint import pprint
import spacy
from nltk.corpus import stopwords
import scipy
# Gensim
import gensim
import gensim.corpora as corpora
from gensim.utils import simple_preprocess
from gensim.models import CoherenceModel
from gensim import corpora, models
import gensim


# In[8]:


stop_words = stopwords.words('english')
stop_words.extend(["say", "official"])
text_df = pd.read_csv('filepath', sep='delimiter', engine='python', header=None)

#print(stop_words)


# In[9]:


text_list = text_df.values.tolist() #transform dataframe to list


# In[10]:


def sent_to_words(sentences):
    for sentence in text_list:
        yield(gensim.utils.simple_preprocess(str(sentence), deacc=True))
        

data_words = list(sent_to_words(text_list))
    


# In[11]:


bigram = gensim.models.Phrases(data_words, min_count=5, threshold=100) # higher threshold fewer phrases.
trigram = gensim.models.Phrases(bigram[data_words], threshold=100)  


# In[12]:


bigram_mod = gensim.models.phrases.Phraser(bigram)
trigram_mod = gensim.models.phrases.Phraser(trigram)


# In[13]:


def delete_stopwords(text): 
    return [[word for word in simple_preprocess(str(doc)) if word not in stop_words] for doc in text]

def make_bigrams(text):
    return [bigram_mod[doc] for doc in text]

def make_trigrams(text):
    return [trigram_mod[bigram_mod[doc]] for doc in text]

def lemmatization(text, allowed_postags=['NOUN', 'ADJ', 'VERB', 'ADV']):
    """https://spacy.io/api/annotation"""
    texts_out = []
    for sent in text:
        doc = nlp(" ".join(sent)) 
        texts_out.append([token.lemma_ for token in doc if token.pos_ in allowed_postags])
    return texts_out


# In[14]:


data_words_nostops = delete_stopwords(data_words)

# Form Bigrams
data_words_bigrams = make_bigrams(data_words_nostops)

# Initialize spacy 'en' model, keeping only tagger component (for efficiency)
nlp = spacy.load('en', disable=['parser', 'ner'])

# Do lemmatization keeping only noun, adj, vb, adv
data_lemmatized = lemmatization(data_words_bigrams, allowed_postags=['NOUN', 'ADJ', 'VERB', 'ADV'])

print(data_lemmatized[:1])


# In[15]:


id2word = corpora.Dictionary(data_lemmatized)

# Create Corpus
texts = data_lemmatized

# Term Document Frequency
corpus = [id2word.doc2bow(text) for text in texts]

# View
#print(corpus[:1])


# In[16]:


#Initializing LDA model
lda_model = gensim.models.ldamodel.LdaModel(corpus=corpus,
                                           id2word=id2word,
                                           num_topics=50, 
                                           random_state=42,
                                           update_every=1,
                                           chunksize=100,
                                           passes=10,
                                           alpha='auto',
                                           per_word_topics=False)


# In[17]:


pprint(lda_model.print_topics())
doc_lda = lda_model[corpus]


# In[18]:





print('\nPerplexity: ', lda_model.log_perplexity(corpus)) 


# In[19]:


coherence_model_lda = CoherenceModel(model=lda_model, texts=data_lemmatized, dictionary=id2word, coherence='c_v')
coherence_lda = coherence_model_lda.get_coherence()
print('\nCoherence Score: ', coherence_lda)

