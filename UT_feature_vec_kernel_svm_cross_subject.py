
from myfunctions import Action_recognition_class as arc
from sklearn.svm import SVC
import random
import numpy as np

#reading sata
all_actions, labels = arc.read_UTkinect_aligned()

#preparing data
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

'''
two functions that extract feature
            first one is based on multi dimentional kernel
            second one is based on one dimentional kernel
'''

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

#==================================================================

#training and testing for every subject
#for UTKinect dataset, 10 is number of subjects and 20 is number of performed action by each subject
    
test_s = []
train_s = []

for v in range(10):
    
    print('fold_number', v)

    fold1 = all_action_prepared[0 : v * 20]
    fold2 = all_action_prepared[v * 20 : (v + 1) * 20]
    fold3 = all_action_prepared[(v + 1) * 20 : int(len(all_action_prepared))]
    
    label_fold1 = labels[0 : v * 20]
    label_fold2 = labels[v * 20 : (v + 1) * 20]
    label_fold3 = labels[(v + 1) * 20 : int(len(all_action_prepared))]
    
    
    training_data = fold1 + fold3
    training_label = label_fold1 + label_fold3
    
    test_data = fold2
    test_label = label_fold2
    

    feature_vector_train = feature_extractor_one_dim(training_data)
    
    feature_vector_test = feature_extractor_one_dim(test_data)
    
 
    clf = SVC(C = 500)
    
    
    clf.fit(feature_vector_train,training_label)
    
    train_s.append(clf.score(feature_vector_train,training_label))
    
    test_acc = clf.score(feature_vector_test,test_label)

    #each element of test_s indicate accuracy rate for each subject      
    test_s.append(test_acc)
    
    with open("results_UT", "a+") as f:
        f.write("Subject number: %d, %f\n" % (v, test_acc))  
 
    
with open("results_UT", "a+") as f:
    f.write("final_result: %f\n" % (np.mean(test_s)))  