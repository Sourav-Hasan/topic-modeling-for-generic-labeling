from nltk.tokenize import RegexpTokenizer
from stop_words import get_stop_words
from nltk.stem.porter import PorterStemmer
from nltk.stem import WordNetLemmatizer
from gensim import corpora, models
import re
import nltk
from nltk.corpus import wordnet
import gensim
import pyLDAvis.gensim as gensimvis
import pyLDAvis
import warnings
warnings.filterwarnings('ignore')

tokenizer = RegexpTokenizer(r'\w+')

# create English stop words list
en_stop = get_stop_words('en')

# Create p_stemmer of class PorterStemmer
p_stemmer = PorterStemmer()
l_lemma = WordNetLemmatizer()
    


doc_a = "Traffic jam means a long line of vehicles that can not move or that can move very slowly. It is a common affair in the big cities of our country. There are many causes of traffic jam. Rapid growth of population and the increasing amount of vehicles are the main causes of it."
doc_b = "Vehicles are much more than the roads can accommodate. The indiscriminate playing of rickshaw is another causes of it. Haphazard parking of vehicles alongside the pavement also causes of it. "
doc_c = " Violation of traffic rules is also responsible for it. The drivers do not follow traffic rules. Traffic jam causes untold sufferings to people. Sometimes it raises our mental tension. It causes loss of our valuable time. We have to wait to reach our destination. "
doc_d = " The students, the office-going people, the businessmen and the patients in the ambulance are the worst sufferers of it. Traffic jam can be removed by enforcing traffic jam strictly. "
doc_e = "The narrow roads should be broadened. By pass roads should be constructed in the big towns. One way movement of vehicles and building of fly over can solve this problem. We can reduce it by raising public awareness."
doc_set = [doc_a, doc_b, doc_c, doc_d, doc_e]

# list for tokenized documents in loop
texts = []

# loop through document list
for i in doc_set:
    
    # clean and tokenize document string
    raw = i.lower()
    tokens = tokenizer.tokenize(raw)

    # remove stop words from tokens
    stopped_tokens = [i for i in tokens if not i in en_stop]
    
    # stem tokens
    lemmatize_tokens = [l_lemma.lemmatize(i) for i in stopped_tokens]
    
    
    # add tokens to list
    texts.append(lemmatize_tokens)
    


l=[]
m=[]


a=nltk.pos_tag(texts[0])
#taking only nn and nnp
nn_tagged = [(word,tag) for word, tag in a if tag.startswith('NN') or tag.startswith('NNP')]
        

for i in nn_tagged:
    m.append(i[0])
l.append(m)
            


n=[]
a=nltk.pos_tag(texts[1])
nn_tagged = [(word,tag) for word, tag in a if tag.startswith('NN') or tag.startswith('NNP')]
        

for i in nn_tagged:
    n.append(i[0])
l.append(n)        


o=[]
a=nltk.pos_tag(texts[2])
nn_tagged = [(word,tag) for word, tag in a if tag.startswith('NN') or tag.startswith('NNP')]
        

for i in nn_tagged:
    o.append(i[0])
l.append(o)        


p=[]
a=nltk.pos_tag(texts[3])
nn_tagged = [(word,tag) for word, tag in a if tag.startswith('NN') or tag.startswith('NNP')]
        

for i in nn_tagged:
    p.append(i[0])
l.append(p)        


q=[]
a=nltk.pos_tag(texts[4])
nn_tagged = [(word,tag) for word, tag in a if tag.startswith('NN') or tag.startswith('NNP')]
        

for i in nn_tagged:
    q.append(i[0])
l.append(q)        
print(l)


# turn our tokenized documents into a id <-> term dictionary
dictionary = corpora.Dictionary(l)
    
# convert tokenized documents into a document-term matrix
corpus = [dictionary.doc2bow(text) for text in l]

# generate LDA model
ldamodel = gensim.models.ldamodel.LdaModel(corpus, num_topics=4, id2word = dictionary, passes=2000,random_state=1)

v=ldamodel.print_topics(num_topics=4, num_words=2)
print(v)
       
t1=v[0][1]
t2=v[1][1]
t3=v[2][1]
t4=v[3][1]

print(t1,"\n",t2,"\n",t3,"\n",t4,"\n")

tv2=t1.split()[0]

tv3 = " ".join(re.findall("[a-zA-Z]+", tv2))

tv4=re.split("[^a-zA-Z]*", tv3)

best_topic=""
for item in tv4:
    best_topic=str(item)
    print(best_topic)
    
syns = wordnet.synsets("road")

des=[]

n_doc_a = ""
n_doc_b = ""
n_doc_c = ""
n_doc_d = ""
n_doc_e = "" 

for i in range(1):
    if i==0:
        n_doc_a = syns[i].definition()
        print(n_doc_a)
    elif i== 1:
        n_doc_b = syns[i].definition() 
    elif i== 2: 
        n_doc_c = syns[i].definition()
    elif i== 3:
        n_doc_d = syns[i].definition()
    elif i==4:
        n_doc_e = syns[i].definition()
        

n_doc_set = [n_doc_a, n_doc_b, n_doc_c, n_doc_d, n_doc_e]
n_texts = []

for i in n_doc_set:
    
    # clean and tokenize document string
    raw = i.lower()
    tokens = tokenizer.tokenize(raw)

    # remove stop words from tokens
    stopped_tokens = [i for i in tokens if not i in en_stop]
    
    # stem tokens
    lemmatize_tokens = [l_lemma.lemmatize(i) for i in stopped_tokens]
    
    n_texts.append(lemmatize_tokens)
    
print(n_texts)

n_l=[]
n_m=[]

n_a=nltk.pos_tag(n_texts[0])
n_nn_tagged = [(word,tag) for word, tag in n_a if tag.startswith('NN') or tag.startswith('NNP')]
        

for i in n_nn_tagged:
    n_m.append(i[0])
n_l.append(n_m)
            
n_n=[]
n_a=nltk.pos_tag(n_texts[1])
n_nn_tagged = [(word,tag) for word, tag in n_a if tag.startswith('NN') or tag.startswith('NNP')]
        
for i in n_nn_tagged:
    n_n.append(i[0])
n_l.append(n_n)        

n_o=[]
n_a=nltk.pos_tag(n_texts[2])
n_nn_tagged = [(word,tag) for word, tag in n_a if tag.startswith('NN') or tag.startswith('NNP')]
        
for i in n_nn_tagged:
    n_o.append(i[0])
n_l.append(n_o)        

n_p=[]
n_a=nltk.pos_tag(n_texts[3])
n_nn_tagged = [(word,tag) for word, tag in n_a if tag.startswith('NN') or tag.startswith('NNP')]
        
for i in n_nn_tagged:
    n_p.append(i[0])
n_l.append(n_p)        

n_q=[]
n_a=nltk.pos_tag(n_texts[4])
n_nn_tagged = [(word,tag) for word, tag in n_a if tag.startswith('NN') or tag.startswith('NNP')]
        
for i in n_nn_tagged:
    n_q.append(i[0])
n_l.append(n_q)        
print(n_l)

n_dictionary = corpora.Dictionary(n_l)
    
# convert tokenized documents into a document-term matrix
n_corpus = [n_dictionary.doc2bow(n_text) for n_text in n_l]

# generate LDA model
n_ldamodel = gensim.models.ldamodel.LdaModel(n_corpus, num_topics=1, id2word = n_dictionary, passes=2000,random_state=1)
n_v=n_ldamodel.print_topics(num_topics=1, num_words=1)
print(n_v)

print(ldamodel2.print_topics(num_topics=4, num_words=2))
vis_data = gensimvis.prepare(n_ldamodel, n_corpus, n_dictionary)
pyLDAvis.display(vis_data)
