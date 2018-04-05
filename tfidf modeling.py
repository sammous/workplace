
# coding: utf-8

# In[1]:


import logging
logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO)

from gensim import corpora
from pymongo import MongoClient
from gensim.models import TfidfModel


# Keeping only Noun or ADJ for the topic modeling (removing punct, verbs..)

# In[20]:


ACCEPTED = ['NOUN', 'PROPN', 'ADJ']
def preprocessed(token):
    tokens = token.split('|')
    if len(tokens) == 3:
        tag = token.split('|')[2]
        lemma = token.split('|')[1]
        if tag in ACCEPTED:
            return lemma
    else:
        return None


# In[21]:


class DictCorpus(object):
    def __init__(self):
        client = MongoClient('localhost', 27017)
        db = client.wikipedia
        self.wiki_fr = db.wiki_fr
	print("Total documents in collection: {}".format(self.wiki_fr.find({'text_processed':{'$exists':1}}).count()))
    def __iter__(self):
        for line in self.wiki_fr.find({'text_processed':{'$exists':1}}):
            tokens = line['text_processed'].split()
            lemmas = [x for x in map(lambda x: preprocessed(x), tokens) if x is not None]
            yield lemmas


# In[22]:


dic = corpora.Dictionary(DictCorpus())


# In[23]:


dic.save_as_text('corpus_wikifr.txt')


# Corpus class to avoid loading in memory the whole corpus at the same time

# In[24]:


class TfIdfCorpus(object):
    def __init__(self):
        client = MongoClient('localhost', 27017)
        db = client.wikipedia
        self.wiki_fr = db.wiki_fr
    def __iter__(self):
        for line in self.wiki_fr.find({'text_processed':{'$exists':1}}):
            tokens = line['text_processed'].split()
            lemmas = [x for x in map(lambda x: preprocessed(x), tokens) if x is not None]
            yield dic.doc2bow(lemmas)


# In[ ]:


corpora.MmCorpus.serialize('corpus.mm', TfIdfCorpus())


# In[ ]:


corpus = corpora.MmCorpus('corpus.mm')


# In[ ]:


tfidf_model = TfidfModel(TfIdfCorpus())


# In[ ]:


tfidf_model.save('tfidfwiki_fr.model')


# In[ ]:


from gensim import similarities

index = similarities.SparseMatrixSimilarity(tfidf_model[corpus], num_features=50000)


# In[ ]:


import spacy
text = """
L'expérience concerne des souris, mais dans ce cas précis elles sont un bon modèle animal de l'humain : des chercheurs ont démontré que l'ADN des cellules du cerveau des nouveaux-nés se modifie (jusqu'au moment du sevrage) en fonction du type de maternage qu'ils reçoivent : moins ils sont choyés, plus il change.

L'on savait déjà que les interactions avec l'environnement agissent comme des "interrupteurs" génétiques, activant ou désactivant des gènes - ces "unités fonctionnelles" de l'ADN qui codent un trait de l'organisme (couleurs des yeux ou autre) (voir S&V n°1051, 2005).

Mais on ignorait jusqu'ici que durant la période précédant le sevrage l'environnement peut modifier la structure même de l'ADN des cellules cérébrales, c'est-à-dire la répartition et le nombre de gènes sur le brin d'ADN. Une première !
Maternage et « gènes sauteurs »

Les implications cognitives voire neuropsychiatriques de ce phénomène ne sont pas encore clairement établies, les chercheurs renvoyant à des études postérieures. Ce que l'on sait globalement c'est que le stress est impliqué dans un processus génétique nommé « rétrotransposition » qui pousse certains gènes à multiplier leur copie sur un brin d'ADN. Des « gènes sauteurs ».

Dans l'expérience, les chercheurs ont suivi le mouvement de « copier-coller » de ces gènes sauteurs dans les cellules cérébrales des nouveaux-nés (plus précisément, l'hippocampe), nommés longs éléments nucléaires intercalés (LINE en anglais), sur deux groupes de souris durant deux semaines.
L'ADN des souriceaux moins maternés est plus variable

Le premier groupe était constitué par les souris femelles (et leurs rejetons) dont le comportement observé au cours de l'expérience était protecteur et attentionné (soins, toilettage, etc.) . Le second groupe rassemblait les souris clairement moins attentionnées avec leurs souriceaux.

Les chercheurs ont alors démontré qu'il existait une corrélation inverse entre le nombre de gènes copiés-collés dans l'ADN des cellules cérébrales et le degré de maternage de chaque femelle, mesuré selon une échelle numérique prenant en compte le temps consacré au toilettage des bébés, l'attention de la mère, etc.
Un effet direct de l'environnement

Pour être certains que cette corrélation était réelle, les chercheurs ont pris des précautions, notamment en échangeant des nouveaux-nés : certains nés de mères peu attentives furent confiés à des mères attentives et vice-versa. Ils ont alors constaté le même phénomène, éloignant la possibilité que le taux de gènes copiés-collés soit un effet héréditaire des parents et non une réaction directe à l'environnement.

Si les chercheurs ont émis certaines hypothèses sur la chaîne de réactions biochimiques conduisant le comportement de la mère à provoquer un effet sur les gènes sauteurs de leurs bébés (dont l'effet de méthylation), ce mécanisme demande encore à être dévoilé.

Néanmoins, un suivi postérieur des nouveau-nés a montré que ceux ayant subi de nombreux copier-coller génétiques étaient des adultes plus stressés et inadaptés.
De nouvelles perspectives thérapeutiques

Aussi, le résultat offre une nouvelle perspective pour l'interprétation génétique de certaines pathologies psychiatriques et neurologiques comme la dépression ou la schizophrénie chez les humains : cela pourrait se jouer à l'échelle génétique dans la période d'avant-sevrage (de six mois à un an).

Sans oublier que la découverte ouvre la possibilité d'attaquer ces pathologies par des traitements biochimiques visant directement le mécanisme de rétro-transposition des gènes sauteurs.
"""

nlp = spacy.load('fr')

txt_processed = nlp(text.decode('utf-8'))
tokens = []
for token in txt_processed:
    tokens.append(token.lemma_)


# In[ ]:


id2word = corpora.Dictionary.load_from_text('corpus_wikifr.txt')


# In[ ]:


query_bow = id2word.doc2bow(tokens)
query_tfidf = tfidf_model[query_bow]
index.num_best = 3
print(index[query_tfidf])

