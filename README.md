This is the tool to Inject Word embedings with another languages resources.

Requirements : 
1. Python 3.5
2. Numpy
3. Annoy (Optional)

Data :
1. English PPDB database (already present)
2. Word Embeddings (automatically downloaded from http://128.2.220.95/multilingual/data/)

To Execute : 
python biretro.py

Output :
Evaluates the improved Italian Embeddings (by using English PPDB) on Italian WS353 (similarity) and Italian SimLex999 (http://www.leviants.com/ira.leviant/MultilingualVSMdata.html#abstract)



There are Two Modes to run the script : 
1. Run using Precomputed translations: The package comes with pre computed English-Italian Translations using Nearest Neighbour Search. (results are based on this)

2. Run by computing translations right now : For this, we use Annoy (https://github.com/spotify/annoy) that finds approximate nearest neighbour translations very quickly with some loss in quality.


Parameters :
Eta : 0.70 (Adjust to control the size of translated resource )
Filter : 2 (Adjust to control noise, recommended value is 2)
Iterations : 10


SOURCES :
The benchmarks are from sources as cited in the paper.
Lexicon Database is as cited in the paper.
The core part of Retrofitting Code is due to Faruqui et al 2015.



