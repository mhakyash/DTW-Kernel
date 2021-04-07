import scipy.io
import numpy as np
import fastdtw
import math

class Action_recognition_class:

    def read_tst():

        mat = scipy.io.loadmat('Datasets/TST.mat')

        my_numpy = mat['complete_skeleton']

        all_actions1 = []
        labels = []

        c = -1
        for i in range(my_numpy.shape[0]):

            for j in range(my_numpy.shape[1]):
                for w in range(my_numpy.shape[2]):
                    c = c + 1
                    all_actions1.append([])
                    labels.append(w)
                    for k in range(len(my_numpy[0][0][0])):
                        all_actions1[c].append(my_numpy[i][j][w][k])

        all_actions = []

        for w in range(my_numpy.shape[0] * my_numpy.shape[1] * my_numpy.shape[2]):
            c = -1
            all_actions.append([])
            for j in range(my_numpy.shape[4]):
                for i in range(my_numpy.shape[5]):
                    c = c + 1
                    all_actions[w].append([])
                    for n in range(my_numpy.shape[3]):
                        all_actions[w][c].append(all_actions1[w][n][j][i])

        return all_actions, labels


    def read_UTkinect_aligned():

        import scipy.io
        mat = scipy.io.loadmat('Datasets/all_aligned_joints_gathered_UTKinect.mat')
        my_numpy = mat['all_aligned_joints_gathered']

        all_actions1 = []
        labels = []

        c = -1
        for i in range(my_numpy.shape[0]):

            for j in range(my_numpy.shape[1]):
                for w in range(my_numpy.shape[2]):
                    c = c + 1
                    all_actions1.append([])
                    labels.append(w)
                    for k in range(len(my_numpy[0][0][0])):
                        all_actions1[c].append(my_numpy[i][j][w][k])

        all_actions = []

        for w in range(my_numpy.shape[0] * my_numpy.shape[1] * my_numpy.shape[2]):
            c = -1
            all_actions.append([])
            for j in range(my_numpy.shape[4]):
                for i in range(my_numpy.shape[5]):
                    c = c + 1
                    all_actions[w].append([])
                    for n in range(my_numpy.shape[3]):
                        all_actions[w][c].append(all_actions1[w][n][j][i])

        return all_actions, labels


    def read_UTkinect():
        
        import scipy.io
        mat = scipy.io.loadmat('Datasets/UTKinect.mat')
        my_numpy = mat['complete_skeleton']

        all_actions1 = []
        labels = []

        c = -1
        for i in range(my_numpy.shape[0]):

            for j in range(my_numpy.shape[1]):
                for w in range(my_numpy.shape[2]):
                    c = c + 1
                    all_actions1.append([])
                    labels.append(w)
                    for k in range(len(my_numpy[0][0][0])):
                        all_actions1[c].append(my_numpy[i][j][w][k])

        all_actions = []

        for w in range(my_numpy.shape[0] * my_numpy.shape[1] * my_numpy.shape[2]):
            c = -1
            all_actions.append([])
            for j in range(my_numpy.shape[4]):
                for i in range(my_numpy.shape[5]):
                    c = c + 1
                    all_actions[w].append([])
                    for n in range(my_numpy.shape[3]):
                        all_actions[w][c].append(all_actions1[w][n][j][i])

        return all_actions, labels


     
    def read_UCFKinect_aligned():
        
        import scipy.io
        mat = scipy.io.loadmat('Datasets/all_aligned_joints_gathered_UCFKinect.mat')
        my_numpy = mat['all_aligned_joints_gathered']

        all_actions1 = []
        labels = []

        c = -1
        for i in range(my_numpy.shape[0]):

            for j in range(my_numpy.shape[1]):
                for w in range(my_numpy.shape[2]):
                    c = c + 1
                    all_actions1.append([])
                    labels.append(w)
                    for k in range(len(my_numpy[0][0][0])):
                        all_actions1[c].append(my_numpy[i][j][w][k])

        all_actions = []

        for w in range(my_numpy.shape[0] * my_numpy.shape[1] * my_numpy.shape[2]):
            c = -1
            all_actions.append([])
            for j in range(my_numpy.shape[4]):
                for i in range(my_numpy.shape[5]):
                    c = c + 1
                    all_actions[w].append([])
                    for n in range(my_numpy.shape[3]):
                        all_actions[w][c].append(all_actions1[w][n][j][i])

        return all_actions, labels
    
    
    
    def triangle_area(a, b, c):
        s = (a + b + c) / 2

        area = np.sqrt(abs(s * (s - a) * (s - b) * (s - c)))

        return area

    def distance(xi, yi, xii, yii):
        sq1 = (xi - xii) * (xi - xii)
        sq2 = (yi - yii) * (yi - yii)
        return np.sqrt(sq1 + sq2)


    def normalizer(X):
        X = [x for x in X if x!=0]
        X_norm = (X - np.min(X)) / (np.max(X) - np.min(X))
        X_norm = 2*(X_norm - .5)
        return X_norm

    def mean(a, b):
        return ((a + b) / 2)


    def my_kernel(self, X, Y, learning_rate=0.01):
        triangle_area_list = []

        dtw_dis, warping_path = fastdtw.fastdtw(X, Y)

        L = len(warping_path)

        for i in range(1, L):

            if (warping_path[i][0] == warping_path[i - 1][0] + 1) and (warping_path[i][1] == warping_path[i - 1][1] + 1):
                # Area of a square should be calculated

                a = self.distance(warping_path[i - 1][0],
                             X[warping_path[i - 1][0]],
                             warping_path[i - 1][1],
                             Y[warping_path[i - 1][1]])
                b = self.distance(warping_path[i][1], Y[warping_path[i][1]],
                             warping_path[i - 1][1],
                             Y[warping_path[i - 1][1]])
                c = self.distance(warping_path[i-1][0], X[warping_path[i-1][0]],
                             warping_path[i][1],
                             Y[warping_path[i][1]])

                area1 = self.triangle_area(a, b, c)

                a = self.distance(warping_path[i][0], X[warping_path[i][0]],
                             warping_path[i][1], Y[warping_path[i][1]])
                b = self.distance(warping_path[i][0], X[warping_path[i][0]],
                             warping_path[i - 1][0],
                             X[warping_path[i - 1][0]])
                c = self.distance(warping_path[i-1][0], X[warping_path[i-1][0]],
                             warping_path[i][1],
                             Y[warping_path[i][1]])

                area2 = self.triangle_area(a, b, c)

                area = area1 + area2

            if (warping_path[i][0] == warping_path[i - 1][0]) or (warping_path[i][1] == warping_path[i - 1][1]):

                if warping_path[i][0] == warping_path[i - 1][0]:
                    a = self.distance(warping_path[i - 1][0],
                                 X[warping_path[i - 1][0]],
                                 warping_path[i - 1][1],
                                 Y[warping_path[i - 1][1]])
                    b = self.distance(warping_path[i - 1][0],
                                 X[warping_path[i - 1][0]],
                                 warping_path[i][1],
                                 Y[warping_path[i][1]])
                    c = self.distance(warping_path[i][1],
                                 Y[warping_path[i][1]],
                                 warping_path[i - 1][1],
                                 Y[warping_path[i - 1][1]])

                    area = self.triangle_area(a, b, c)

                if (warping_path[i][1] == warping_path[i - 1][1]):
                    a = self.distance(warping_path[i - 1][0],
                                 X[warping_path[i - 1][0]],
                                 warping_path[i - 1][1],
                                 Y[warping_path[i - 1][1]])
                    b = self.distance(warping_path[i][0],
                                 X[warping_path[i][0]],
                                 warping_path[i - 1][0],
                                 X[warping_path[i - 1][0]])
                    c = self.distance(warping_path[i][0],
                                 X[warping_path[i][0]],
                                 warping_path[i - 1][1],
                                 Y[warping_path[i - 1][1]])

                    area = self.triangle_area(a, b, c)

            # triangle_area_list.append(np.float32(area))
            
            triangle_area_list.append(math.exp(-learning_rate * area))

        # final_kernel = math.exp(-learning_rate * (np.sum(triangle_area_list)/L))
        final_kernel = np.sum(triangle_area_list)  / (L-1)

        return final_kernel

    def zero_padding(self,all_actions):

        all_actions_not_zeropad = []
        for i in range(len(all_actions)):
            all_actions_not_zeropad.append([])
            for j in range(len(all_actions[0])):
                all_actions_not_zeropad[i].append([])
                all_actions_not_zeropad[i][j] = [x for x in all_actions[i][j] if x != 0]
        return all_actions_not_zeropad
    
    
    
    def feature_to_joint(feature_number):
        
        return(int(feature_number / 3) + 1)


    def feature_to_joint_ext(feature_number):
        
        if feature_number % 3 == 0:
            c = 'x'
        if feature_number % 3 == 1:
            c = 'y'
        if feature_number % 3 == 2:
            c = 'z'
        
        c = str(int(feature_number / 3) + 1) + c
        
        return c
    
    
    def most_info_feature(self, action, number_of_joints = 20):
        
        a = [np.var(x) for x in action]
        a_sorted = a.copy()
        a_sorted.sort()
        
        indexes = []
        
        b = a_sorted[-number_of_joints:]
        b.reverse()
        
        indexes = [a.index(x) for x in b]

        return indexes
    
    
    def dim_dis(a, b):
        
        my_list = []
        for i in range(len(a)):        
        
            my_list.append((a[i]-b[i])**2)
        return(np.sqrt(np.sum(my_list)))
  
    
    def prepare_data_tst(self, all_action):
        
        all_action_new = []
        
        for c in range(len(all_action)):
            all_action_new.append([])       
            for k in range(125):
                all_action_new[c].append([])
                for j in range(60):
                    all_action_new[c][k].append(all_action[c][j][k])
        
        
        all_action_new_zero = self.zero_padding(self, all_action_new)
        
                    
        for i in range(len(all_action_new_zero)):
            for j in range(len(all_action_new_zero[i])):
                if all_action_new_zero[i].count([]) != 0:
                    all_action_new_zero[i].remove([])
                    
                    
        return(all_action_new_zero)
          

        
        
        
