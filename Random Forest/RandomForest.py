import pandas as pd
df = pd.read_csv('final_dataset.csv', header = None)
df = df.sample(frac = 1)
data_list = df.values.tolist()
training_data = data_list[0:800]
testing_data = data_list[800:891]

def unique_vals(rows, col):
    return set([row[col] for row in rows])

def class_counts(rows):
    counts = {}
    for row in rows:
        label = row[-1]
        if label not in counts:
            counts[label] = 0
        counts[label] += 1
    return counts

def is_num(value):
    return isinstance(value, int) or isinstance(value, float)


class Question:
    def __init__(self, col, val):
        self.column = col
        self.value = val

    def match(self, example):
        val = example[self.column]
        if is_num(val):
            return val >= self.value
        else:
            return val == self.value

def partition(rows, question):
    true_rows, false_rows = [], []
    for row in rows:
        if question.match(row):
            true_rows.append(row)
        else:
            false_rows.append(row)
    return true_rows, false_rows

def gini(rows):
    counts = class_counts(rows)
    impurity = 1
    for lbl in counts:
        prob_of_lbl =  counts[lbl] / float(len(rows))
        impurity -= prob_of_lbl ** 2
    return impurity

def information_gain(left, right, current_impurity):
    prob = float(len(left)) / (len(left) + len(right))
    return current_impurity - prob * gini(left) - (1 - prob) * gini(right)


def best_split(rows):
    best_gain = 0
    best_question = None
    current_impurity = gini(rows)
    n_features = len(rows[0]) - 1

    for col in range(n_features):
        values = set([row[col] for row in rows])
        for val in values:
            question = Question(col, val)
            true_rows, false_rows = partition(rows, question)

            if len(true_rows) == 0 or len(false_rows) == 0:
                continue

            gain = information_gain(true_rows, false_rows, current_impurity)

            if gain >= best_gain:
                best_gain, best_question = gain, question
    return best_gain, best_question

class Leaf:
    def __init__(self, rows):
        self.predictions = class_counts(rows)

class DecisionNode:
    def __init__(self, question, true_branch, false_branch):
        self.question = question
        self.true_branch = true_branch
        self.false_branch = false_branch


def build_tree(rows):
    gain, question = best_split(rows)

    if gain == 0:
        return Leaf(rows)

    true_rows, false_rows = partition(rows, question)

    true_branch = build_tree(true_rows)
    false_branch = build_tree(false_rows)

    return DecisionNode(question, true_branch, false_branch)


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

correct_count = 0
for row in testing_data:
    prediction = list(classify(row[0:-1], my_tree).keys())[0]
    actual_value = row[-1]
    if prediction == actual_value:
        correct_count += 1
    print("Actual: %s. Predicted: %s" %(actual_value, prediction))
accuracy = correct_count / len(testing_data) * 100
print("Accuracy is : ", accuracy)