import pickle
import numpy as np
import gzip
import sys
import os
from copy import deepcopy
## Bypass Warnings
import warnings
def warn(*args, **kwargs):
	pass
warnings.warn = warn
## Import Benchmarks
from evaluator import *
## Load English PPDB Databse
lexicon=pickle.load(open("./lexicon_database/lexicon","rb")) 


def download_embeddings():
	'''Download embeddings trained by Ammar et al 2016 (http://128.2.220.95/multilingual/data/)'''
	### First check if File already present
	if (os.path.isfile(file)==True):
		print("Found embeddings")
		return
	
	else:
		print("Downloading embeddings")
		link="http://wordvectors.org/trained-vecs/multilingual/"+file
		status=os.system("wget "+link)
		if status!=0:
			print("Unable to download embeddings")
			print("You can manually download and run again")
		else :
			print("Downloaded!!")

def load_embeddigs(language):
	'''Function to load bilingual embeddings from text file'''
	
	### Read content of file
	
	with gzip.open(file,'rb') as f:
		content = f.read()
		content=content.decode("utf-8") 
	content=content.split("\n") ## reads one null line

	### Load Content of File
	vocab=[]
	embeddings=[]
	for i in range(0,len(content)-1): ## avoid last element as it is empty
		lang_tag=content[i][:2]
		if language==lang_tag:
			tmp=content[i][3:] ### since 'en:' tag is to be avoided
			tmp=tmp.split(" ")
			## Extract Word
			word=tmp[0]
			vocab.append(word)
			
			## Extract Embeddings
			tmp1=[]
			for j in range(1,len(tmp)):
				tmp1.append(float(tmp[j]))
			embeddings.append(np.array(tmp1))

	language={}
	for i in range(0,len(vocab)):
		language[vocab[i]]=embeddings[i]
	return language,embeddings,vocab
	

def load_translations(name):
	''' Load Proper Pre computed KNN Translations ''' 
	tmp=pickle.load(open("./translations/"+name,'rb'))
	return tmp


def create_translations():
    '''Create Fast Translations using Annoy Library (Be sure to install annoy)'''
    from annoy import AnnoyIndex
    f=40 		# Set Dimension of Embeddings 
    tree=54 	# Set number of Trees
    print("Creating Approximate translations now.....")

    ###### TARGET TO SOURCE TRANSLATIONS
    t = AnnoyIndex(f)  # Length of item vector that will be indexed
    for i,j in enumerate(sou_vectors):
        t.add_item(i,j)

    t.build(tree)
    lang_eng={}
    for index,item in enumerate(tar_vectors):
        TheResult = t.get_nns_by_vector(item,2,search_k=-1,include_distances=False)[0]
        lang_eng[tar_vocab[index]]=sou_vocab[TheResult]

    ###### SOURCE TO TARGET TRANSLATIONS
    t = AnnoyIndex(f)  # Length of item vector that will be indexed
    for i,j in enumerate(tar_vectors):
        t.add_item(i,j)

    t.build(tree) 
    eng_lang={}
    for index,item in enumerate(sou_vectors):
        TheResult = t.get_nns_by_vector(item,2,search_k=-1,include_distances=False)[0]
        eng_lang[sou_vocab[index]]=tar_vocab[TheResult]

    return lang_eng,eng_lang

def generate_resource(eta):
	''' Generate the Translated Resource List using Dual Approach'''
	lexicon_generated={}
	lexicon_weight={}
	for k in tar_vocab:  
		try :
			word = target_source[k]
		except KeyError:
			continue
		if (target[k].dot(source[word].T))>eta :
			### Find Lexicons for english meaning
			try :
				lex=lexicon[word] ## Knowledge Source
			except KeyError:
				continue

			### Lexicon Generation
			tmp=[]
			tmp_weight=[]
			for i in lex: 
				if i in source:
					tmp1=source_target[i]
					weight=target[tmp1].dot(source[i].T)
					tmp.append(tmp1)
					tmp_weight.append(weight)
			if(len(tmp)!=0):
				lexicon_generated[k]=tmp
				lexicon_weight[k]=tmp_weight
	return lexicon_generated,lexicon_weight


def retrofit(lexicon, lexicon_weight,filter,numIters):
	'''Retrofitting with Weights and Translated Resource List'''
	wordVecs=deepcopy(target)
	newWordVecs = deepcopy(wordVecs)
	wvVocab = set(newWordVecs.keys())
	loopVocab = wvVocab.intersection(set(lexicon.keys()))
	for it in range(numIters):
		for word in loopVocab:
			
			### Weight Processing
			tmp=lexicon_weight[word]
			numNeighbours=0
			for i in tmp:
				numNeighbours+=i**filter
			#wordNeighbours = set(lexicon[word]).intersection(wvVocab)
			wordNeighbours=lexicon[word]
			#no neighbours, pass - use data estimate
			if len(wordNeighbours) == 0 or numNeighbours==0: ### Redundant
				continue
			# the weight of the data estimate if the number of neighbours
			newVec = numNeighbours * wordVecs[word]
			# loop over neighbours and add to new vector (currently with weight 1)
			for i,ppWord in enumerate(wordNeighbours):
				tmp1=tmp[i]**filter
				#if (word=='professore' and it<=1):
				#    print(newVec,tmp1)
				if tmp1==0:
					continue
				newVec += tmp1*newWordVecs[ppWord]
			newWordVecs[word] = newVec/(2*numNeighbours)
	return newWordVecs



#########	#########
######### Main #########
#########	#########

### Download embeddings
file='three.table3.translation_invariance.window_3+size_40.normalized.gz'
download_embeddings()

### Load Embeddings
print("Loading Embeddings")
source,sou_vectors,sou_vocab=load_embeddigs('en')
target,tar_vectors,tar_vocab=load_embeddigs('it')
print("Completed Loading")

### Load Translations
choice=input("Input 1 to run with pre-computed translations & 2 to transalte now (1 recommended)")
if choice=='1':
	target_source=load_translations('target_to_source')
	source_target=load_translations('source_to_target')
elif choice=='2':
	target_source,source_target=create_translations()
else:
	print("Wrong choice entered...Aborting")
	sys.exit(0)


### Generate Translated Resources
eta=0.70
print("Value of eta :",eta)
lexicon_generated,lexicon_weight=generate_resource(eta)
print("Length of Translated Resource:",len(lexicon_generated))

### Retrofit with Weights and Noise Control
print("Retrofitting for 10 iterations with filter=2 using English PPDB")
fitted_embeddings_target=retrofit(lexicon_generated,lexicon_weight,2,10)

### Display Results
print("Italian WordSim Task for Similairty(Before & After):")
evaluate_embeddings(target,"word_sim_ita")
evaluate_embeddings(fitted_embeddings_target,"word_sim_ita")
print("Italian SimLex 999 (Before & After) :")
evaluate_embeddings(target,"simlex_ita")
evaluate_embeddings(fitted_embeddings_target,"simlex_ita")