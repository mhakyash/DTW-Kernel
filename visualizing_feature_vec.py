
#This code is used for t-SNE visualization of extracted featute vectors

from myfunctions import Action_recognition_class as arc
import numpy as np
import matplotlib.pyplot as plt
import random
from sklearn.manifold import TSNE


all_actions, labels = arc.read_UTkinect_aligned()

all_action_prepared = arc.zero_padding(arc, all_actions)
all_action_prepared = [x[3:] for x in all_action_prepared]

num_of_classes = len(set(labels))
#=============================================================================================

ref_samples = []
for i in range(0,num_of_classes):
    while(len(ref_samples) < (i + 1)):
        j = random.randint(0, len(all_action_prepared) - 1)
        if (labels[j] == i):
            ref_samples.append(all_action_prepared[j])

#=============================================================================================
ref_indexes = [x for x in range(len(ref_samples[0]))]
ref_indexes = [ref_indexes for i in range(num_of_classes)]

#If you want to use the dimension reduction, uncomment the code below, number_of_joints represent the number of used feature 

'''
def most_info_feature_ref():
    
    ref_indexes = []
    for i in range(len(ref_samples)):
        ref_indexes.append(arc.most_info_feature(arc, ref_samples[i], number_of_joints = 20))
    return ref_indexes

ref_indexes = most_info_feature_ref()
'''
            
def feature_extractor_one_dim(data):

    feature_vector = []
    c = 0
    for j in range(0,len(data)):
        feature_vector.append([])
        
        
        for w in range(len(ref_samples)):
            for d in ref_indexes[w]:
                   
                feature_vector[c].append(arc.my_kernel(arc, data[j][d], ref_samples[w][d], learning_rate=0.01))
        c = c + 1
    
    return feature_vector
#=============================================================================================

all_features_vec = feature_extractor_one_dim(all_action_prepared)

#=======================================================
 
X = np.array(all_features_vec)
X_embedded = TSNE(n_components=2, perplexity = 10, early_exaggeration = 10 ,n_iter = 1000).fit_transform(X)


x = [[] for a in range(num_of_classes)]
y = [[] for a in range(num_of_classes)]

for i in range(len(all_features_vec)):  
    x[i%num_of_classes].append(X_embedded[i][0])
    y[i%num_of_classes].append(X_embedded[i][1])

plt.scatter(x[0], y[0], c='C0', marker='o', label = 'walk')
plt.scatter(x[1], y[1], c='C1', marker='o', label = 'sit down')
plt.scatter(x[2], y[2], c='C2', marker='o', label = 'stand up')
plt.scatter(x[3], y[3], c='C3', marker='o', label = 'pick up')
plt.scatter(x[4], y[4], c='C4', marker='o', label = 'carry' )
plt.scatter(x[5], y[5], c='C5', marker='o', label = 'throw' )
plt.scatter(x[6], y[6], c='C6', marker='o', label = 'push')
plt.scatter(x[7], y[7], c='C7', marker='o', label = 'pull')
plt.scatter(x[8], y[8], c='C8', marker='o', label = 'wave hands')
plt.scatter(x[9], y[9], c='C9', marker='o', label = 'clap hands')

plt.legend()
plt.show()


