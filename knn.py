import math
import csv
import random

#-------------------------------------------------------------
def normalisation(dataset,missing_value='?'):
    min_vals = [float(10000)] * len(dataset[0])
    max_vals = [float(10000)] * len(dataset[0])
    for row in dataset:
        for i in range(len(row)):
            if isinstance(row[i], (int, float)):
                if row[i] != missing_value:
                    if row[i] < min_vals[i]:
                        min_vals[i] = row[i]
                    if row[i] > max_vals[i]:
                        max_vals[i] = row[i]

    for row in dataset:
        for i in range(len(row)):
            if  isinstance(row[i], (int, float)):
                if row[i] != missing_value and max_vals[i] != min_vals[i]:
                    row[i] = (row[i] - min_vals[i]) / (max_vals[i] - min_vals[i])
                else:
                    row[i] = 0.5 
    return dataset
#--------------------------------------------------------------
def evaluation(test_set,predicate_total):
    correct = 0
    i=0
    while(i<len(test_set)):
        if test_set[i][-1] == predicate_total[i]:
            correct = correct + 1
        i=i+1    
    incorrect = len(test_set) - correct
    return correct, incorrect
#------------------------------------------------------------

def load(filename):
    with open(filename) as file:
        reader = csv.reader(file)
        header = next(reader)
        dataset = []
        for row in reader:
            data_row = []
            for val in row:
                try:
                    data_row.append(float(val))
                except ValueError:
                    data_row.append(val)
            dataset.append(data_row)
    return dataset
#--------------------------------------------------------------
def euclide(x1, x2):
    distance = 0
    for i in range(len(x1)):
        if i >= len(x2):
            break
        if isinstance(x1[i], float) and isinstance(x2[i], float):
            distance += (x1[i] - x2[i]) ** 2
        else:
            distance += int(x1[i] != x2[i])
    return math.sqrt(distance)
#-----------------------------------------------
def get_neighbors(dataset, inst, k):
    j = 0
    distances = []
    while j < len(dataset):
         distance = euclide(inst, dataset[j][0:-1])
         distances.append((j, distance))
         j =j+ 1
    distances.sort(key=lambda x: x[1])
    neighbors = []
    d = 0
    while d < k:
        neighbors.append(dataset[distances[d][0]])
        d = d+1
    return neighbors
#-----------------------------------------------
def predicate(neighbors):
    class_votes = {}
    for N in neighbors:
        class_label = N[-1]
        #print(N[-1])
        if class_label in class_votes:
            class_votes[class_label] += 1
            #print(class_votes[class_label])
        else:
            class_votes[class_label] = 1
    predicted_class = max(class_votes, key=class_votes.get)
    return predicted_class
#------------------------------------------------
def calculate_P_R_F(test_set, predicate_total):
    positive_label = predicate_total[0]
    true_positives = 0.0
    false_positives = 0.0
    false_negatives = 0.0
    i=0
    while (i <len(test_set)):
        true_label = test_set[i][-1]
        predicted_label = predicate_total[i]

        if true_label == predicted_label:
            if true_label == positive_label:
                true_positives = true_positives + 1.0
        else:
            if predicted_label == positive_label:
                false_positives = false_positives+1.0

            else:
                false_negatives =false_negatives+ 1.0
        i=i+1

    precision = 0
    if true_positives + false_positives > 0:
        precision = true_positives / (true_positives + false_positives)

    recall = 0
    if true_positives + false_negatives > 0:
        recall = true_positives / (true_positives + false_negatives)

    f_measure = 0
    if precision + recall > 0:
        f_measure = 2 * ((precision * recall) / (precision + recall))

    return precision, recall, f_measure



#------------------------------------------------
def split(dataset,percentage,k):
    results = {0: [], 1: [], 2: [],3 :[],4 : []}
    perc =float(float(percentage)/100)

    train_len=int(perc*len(dataset))
    train_set = []
    i=0
    while( i < train_len):
        train_set.append(dataset[i])
        i=i+1
    test_set=[]
    j=len(train_set)
    while(j<len(dataset)):
        test_set.append(dataset[j])
        j=j+1
    i=0
    predicate_total=[]
    while(i<len(test_set)):
        neighbors = get_neighbors(train_set, test_set[i], k)
        predica = predicate(neighbors)
        predicate_total.append(predica)    
        i=i+1
    precision, recall, f_measure=calculate_P_R_F(test_set, predicate_total)
    correct, incorrect = evaluation(test_set, predicate_total)

    results[0].append(correct)

    results[1].append(incorrect)


    results[2].append(precision)


    results[3].append(recall)


    results[4].append(f_measure)
    
    return results

   
#------------------------------------------------------------
def bagging(dataset, percentage, k, t):
    results = {0: [], 1: [], 2: [], 3: [], 4: []}
    perc = float(percentage) / 100
    train_len = int(perc * len(dataset))
    test_set=[]
    j=train_len
    while(j<len(dataset)):
        test_set.append(dataset[j])
        j=j+1

    #print(train_len)
    
    models = []
    for i in range(t):
        subset = []
        i1 = 0
        while i1 < train_len:
            inst_rand=random.choice(dataset)
            subset.append(inst_rand)
            i1 = i1 + 1
        train_set = subset
        #print(inst_rand)
        #print(subset)
        model = {"train": train_set, "test": test_set}
        models.append(model)
    #print(train_set)
    #print(models)
    predicate_final = []
    for instance in test_set:
        neighbors = []
        for model in models:
               neighbors.extend(get_neighbors(model["train"], instance, k))
        predica = predicate(neighbors)
        predicate_final.append(predica)

    precision, recall, f_measure = calculate_P_R_F(test_set, predicate_final)
    correct, incorrect = evaluation(test_set, predicate_final)
    results[0].append(correct)
    results[1].append(incorrect)
    results[2].append(precision)
    results[3].append(recall)
    results[4].append(f_measure)

    return results





#--------------------------------------------------------


#main partie

#------------------------------------------------------------
filename = 'path/of/your/file'
dataset = load(filename)
k = 1
perc=66
fold=10




results1 = split(dataset,perc,k)


print("les resultat percentage split de dataset avant utiliser bagging pour k=",k,"  et de Percentage = ",perc, ":")
print("Correctly Classified Instances:", results1[0])
print("Incorrectly Classified Instances:", results1[1])
print("Precision:" ,results1[3])
print("Recall:",results1[3])
print("F-measure:",results1[4])


results2 = bagging(dataset,perc,k,3)



print("les resultat percentage split de dataset apres utiliser bagging pour k=",k,"  et de Percentage = ",perc, ":")
print("Correctly Classified Instances:", results2[0])
print("Incorrectly Classified Instances:", results2[1])
print("Precision:" ,results2[2])
print("Recall:",results2[3])
print("F-measure:",results2[4])


