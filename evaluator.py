import math
import numpy
from operator import itemgetter
from numpy.linalg import norm

EPSILON = 1e-6

def evaluate_embeddings(word_vectors,file):

    ### First Normalize Embeddings
    for k,v in word_vectors.items():
      word_vectors[k] /= math.sqrt((word_vectors[k]**2).sum() + 1e-6)

    ### Now Find Similairty 
    manual_dict, auto_dict = ({}, {})
    not_found, total_size = (0, 0)
    for line in open('./benchmarks/'+file,"r"):
      spaces=line.split(" ") 
      if(len(spaces)>3): ### Avoid MultiWord  words from SemEval 2017 
        continue
      line = line.strip().lower()
      word1, word2, val = line.split()
      if word1 in word_vectors and word2 in word_vectors:
        manual_dict[(word1, word2)] = float(val)
        auto_dict[(word1, word2)] = cosine_sim(word_vectors[word1], word_vectors[word2])
      else:
        not_found += 1
      total_size += 1    
    print ("Found: ",total_size - not_found,"Out of: ",total_size)
    print ("%15.4f" % spearmans_rho(assign_ranks(manual_dict), assign_ranks(auto_dict)))


def cosine_sim(vec1, vec2):
  vec1 += EPSILON * numpy.ones(len(vec1))
  vec2 += EPSILON * numpy.ones(len(vec1))
  return vec1.dot(vec2)/(norm(vec1)*norm(vec2))

def assign_ranks(item_dict):
  ranked_dict = {}
  sorted_list = [(key, val) for (key, val) in sorted(item_dict.items(),
                                                     key=itemgetter(1),
                                                     reverse=True)]
  for i, (key, val) in enumerate(sorted_list):
    same_val_indices = []
    for j, (key2, val2) in enumerate(sorted_list):
      if val2 == val:
        same_val_indices.append(j+1)
    if len(same_val_indices) == 1:
      ranked_dict[key] = i+1
    else:
      ranked_dict[key] = 1.*sum(same_val_indices)/len(same_val_indices)
  return ranked_dict

def correlation(dict1, dict2):
  avg1 = 1.*sum([val for key, val in dict1.iteritems()])/len(dict1)
  avg2 = 1.*sum([val for key, val in dict2.iteritems()])/len(dict2)
  numr, den1, den2 = (0., 0., 0.)
  for val1, val2 in zip(dict1.itervalues(), dict2.itervalues()):
    numr += (val1 - avg1) * (val2 - avg2)
    den1 += (val1 - avg1) ** 2
    den2 += (val2 - avg2) ** 2
  return numr / math.sqrt(den1 * den2)

def spearmans_rho(ranked_dict1, ranked_dict2):
  assert len(ranked_dict1) == len(ranked_dict2)
  if len(ranked_dict1) == 0 or len(ranked_dict2) == 0:
    return 0.
  x_avg = 1.*sum([val for val in ranked_dict1.values()])/len(ranked_dict1)
  y_avg = 1.*sum([val for val in ranked_dict2.values()])/len(ranked_dict2)
  num, d_x, d_y = (0., 0., 0.)
  for key in ranked_dict1.keys():
    xi = ranked_dict1[key]
    yi = ranked_dict2[key]
    num += (xi-x_avg)*(yi-y_avg)
    d_x += (xi-x_avg)**2
    d_y += (yi-y_avg)**2
  return num/(math.sqrt(d_x*d_y))

