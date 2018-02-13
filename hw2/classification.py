import sys
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import StandardScaler

def read_data(datapath):
    result = []
    with open(datapath, 'r') as f:
        for line in f.readlines():
            xx = []
            for num in line.split(','):
                xx.append(float(num))
            result.append(xx)
    return result

def write_result(result):
    with open('predict.csv', 'w') as f:
        counter = 0
        for i in range(len(result)):
            counter += 1
            f.write(str(result[i]))
            if counter < len(result):
                f.write('\n')

def split(Data):
    label = [ int(Data[row][-1]) for row in range(len(Data)) ]
    data = [ Data[row][:-1] for row in range(len(Data)) ]
    return train_test_split(data, label, test_size=0.2)

def model_generator(token):
    if token == 'R':
        model = LogisticRegression(penalty='l2', solver='newton-cg', multi_class='ovr', verbose=0, n_jobs=1)
    elif token == 'S':
        model = SVC(C=10.0, kernel='rbf', degree=2, gamma='auto', decision_function_shape='ovr')
    elif token == 'D':
        model = DecisionTreeClassifier(criterion='gini', splitter='best', max_depth=None, min_samples_split=2, max_features=None, max_leaf_nodes=None, min_impurity_decrease=0.0)
    elif token == 'N':
        model = MLPClassifier(hidden_layer_sizes=(100,), activation='relu', solver='adam', alpha=0.0001, batch_size='auto', learning_rate='constant', learning_rate_init=0.001, max_iter=1000, shuffle=True, momentum=0.9)
    return model

def voting(val_data, val_label, model_list, scaler_list, token, val=True):
    result = []
    for i in range(len(val_data)):
        vote = [0, 0]
        for j in range(len(model_list)):
            if token == 'N':
                ans = model_list[j].predict(scaler_list[j].transform([val_data[i]]))
            else:
                ans = model_list[j].predict([val_data[i]])
            vote[ans[0]] += 1
        result.append(vote.index(max(vote)))
    if val:
        return accuracy_score(val_label, result)
    else:
        return result

def classification(train_data, test_data, token):
    models = []
    scores = []
    scalers = []
    for i in range(11):
        X_train, Y_train, X_label, Y_label = split(train_data)
        if token == 'N':
            scaler = StandardScaler(copy=True, with_mean=True, with_std=True)
            scaler.fit(X_train)
            scalers.append(scaler)
            X_train = scaler.transform(X_train)
            Y_train = scaler.transform(Y_train)
        model = model_generator(token)
        model.fit(X_train, X_label)
        models.append(model)
        scores.append(model.score(Y_train, Y_label))
    voting_score = voting(Y_train, Y_label, models, scalers, token)
    if voting_score > max(scores):
        predict = voting(test_data, None, models, scalers, token, False)
    else:
        if token == 'N':
            test_data = scalers[scores.index(max(scores))].transform(test_data)
        predict = models[scores.index(max(scores))].predict(test_data)
    write_result(predict)

def argument_parser(L):
    token = L[1]
    training_data = read_data(L[2])
    testing_data = read_data(L[3])
    classification(training_data, testing_data, token)

if __name__ == '__main__':
    argument_parser(sys.argv)
