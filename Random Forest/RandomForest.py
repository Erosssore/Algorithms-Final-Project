import pandas as pd
df = pd.read_csv('final_dataset.csv', header = None)
df = df.sample(frac = 1)
data_list = df.values.tolist()
# Ratio of training data vs testing data
training_data = data_list[0:800] # Training 90%
testing_data = data_list[800:891] # Testing 10%


#####################
# Boiler-plate code #
#####################

# Testing for unique values (duplicate values)
def unique_vals(rows, col):
    return set([row[col] for row in rows])


# Counting the number of rows that apply to a certain feature
# EX: Number of first class passengers
def class_counts(rows):
    counts = {}
    for row in rows:
        label = row[-1] 
        if label not in counts:
            counts[label] = 0
        counts[label] += 1
    return counts


# Determining whether or not this is a number or a string
def is_num(value):
    return isinstance(value, int) or isinstance(value, float)


# Class that represents the decision nodes; asks the questions
class Question:
    # Initiation
    def __init__(self, col, val):
        self.column = col
        self.value = val

    # Returns string/number
    def match(self, example):
        val = example[self.column]
        if is_num(val):
            return val >= self.value
        else:
            return val == self.value

# Separating rows that are true vs rows that are false
# EX: Did this person survive Y/N
def partition(rows, question):
    true_rows, false_rows = [], []
    for row in rows:
        if question.match(row):
            true_rows.append(row)
        else:
            false_rows.append(row)
    return true_rows, false_rows


########################
# STATISTICAL ANALYSIS #
########################

# This can use entropy but this uses gini
# Gini measures a features impurity
def gini(rows):
    counts = class_counts(rows)
    impurity = 1
    for lbl in counts:
        prob_of_lbl =  counts[lbl] / float(len(rows))
        impurity -= prob_of_lbl ** 2
    return impurity


# Uses the measure of impurity that it got from gini to determine how much the feature helps us make an accurate prediction
# Impurity question
def information_gain(left, right, current_impurity):
    prob = float(len(left)) / (len(left) + len(right))
    return current_impurity - prob * gini(left) - (1 - prob) * gini(right)


# Uses information_gain to determine which feature to choose in a split
def best_split(rows):
    best_gain = 0 # Starts with 0
    best_question = None
    current_impurity = gini(rows)
    n_features = len(rows[0]) - 1 # Reviews questions its been given to determine best information gain 
                                  # By comparing each questions information_gain

    for col in range(n_features):
        values = set([row[col] for row in rows])
        for val in values:
            question = Question(col, val) # Splitting between true and false rows with partitioning
            true_rows, false_rows = partition(rows, question)

            if len(true_rows) == 0 or len(false_rows) == 0:
                continue

            gain = information_gain(true_rows, false_rows, current_impurity) # Determining information gain from each feature

            # If current gain is better than the best information gain
            # Updates best gain and best question with current feature
            if gain >= best_gain:
                best_gain, best_question = gain, question
    return best_gain, best_question


#################
# DECISION TREE #
#################

# Leaf class
class Leaf:
    def __init__(self, rows):
        self.predictions = class_counts(rows) # Number of classifications
        

# Decision node class
class DecisionNode:
    def __init__(self, question, true_branch, false_branch):
        self.question = question            # Is passenger female
        self.true_branch = true_branch      # True
        self.false_branch = false_branch    # False


# Building the tree recursively; O(nlogn)
def build_tree(rows):
    gain, question = best_split(rows) # Determine where to split data based on gain and question

    # If there are no more questions return a classification
    if gain == 0:
        return Leaf(rows) 

    # Whether or not the answer is true or false
    true_rows, false_rows = partition(rows, question)

    true_branch = build_tree(true_rows)
    false_branch = build_tree(false_rows)

    # Recursion
    return DecisionNode(question, true_branch, false_branch)


# Separating decision nodes from leaf nodes
def classify(row, node):
    if isinstance(node, Leaf):
        return node.predictions

    if node.question.match(row):
        return classify(row, node.true_branch)
    else:
        return classify(row, node.false_branch)

my_tree = build_tree(training_data)

test = testing_data[0:1][0][0:-1]

classify(test, my_tree)

# Building the tree
correct_count = 0
for row in testing_data:
    prediction = list(classify(row[0:-1], my_tree).keys())[0] # Make a 2D prediction array
    actual_value = row[-1] # Put the actual value next to the predicted value 
    
    # If prediction is accurate reward the tree
    if prediction == actual_value:
        correct_count += 1
    print("Actual: %s. Predicted: %s" %(actual_value, prediction))

# Based on correct predictions/actual values = accuracy
accuracy = correct_count / len(testing_data) * 100
print("Accuracy is : ", accuracy)
