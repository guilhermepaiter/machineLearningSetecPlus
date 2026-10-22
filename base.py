import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.neighbors import KNeighborsClassifier
from sklearn import metrics

def data():
    data = pd.read_csv("satd_sentiment_3000.csv")

    x = data['abstract']
    y = data['sentiment_final']

    return x, y
# print(data['abstract'])

def encoder(x, y):
    labelEncoder = LabelEncoder();
    yEncoded = labelEncoder.fit_transform(y);

    xTrain, xTest, yTrain, yTest = train_test_split(x, yEncoded, test_size=0.3);

    return xTrain, xTest, yTrain, yTest

def vectorize(xTrain, xTest):
    vectorizer = CountVectorizer();

    xTrainVectorize = vectorizer.fit_transform(xTrain);
    xTestVectorize = vectorizer.transform(xTest);

    return xTrainVectorize, xTestVectorize

def nearestNeighbor(yTrain, xTrainVectorize, xTestVectorize):
    classifierKNN = KNeighborsClassifier(n_neighbors=1);

    classifierKNN.fit(xTrainVectorize, yTrain);

    resultsPredict = classifierKNN.predict(xTestVectorize);

    return resultsPredict

def metric(yTest, resultsPredict):
    accuracy = metrics.accuracy_score(yTest, resultsPredict);
    precision = metrics.precision_score(yTest, resultsPredict, average='binary');
    recall = metrics.recall_score(yTest, resultsPredict, average='binary');
    f1score = metrics.f1_score(yTest, resultsPredict, average='binary');

    print("Acúracia = " + str(accuracy))
    print("Precisão = " + str(precision))
    print("Revocação = " + str(recall))
    print("F1-Score = " + str(f1score))

def main():
    date = data()
    encode = encoder()
    vector = vectorize()
    knn = nearestNeighbor()
    score = metric()
    

'''
import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score, classification_report

CSV_PATH = "satd_sentiment_3000.csv"
TEXT_COLUMN = "abstract"
LABEL_COLUMN = "sentiment_final"
TEST_SIZE = 0.3
RANDOM_STATE = 42
N_NEIGHBORS = 1


def load_data(path, text_col, label_col):
    data = pd.read_csv(path)
    return data[text_col], data[label_col]


def encode_labels(y):
    encoder = LabelEncoder()
    return encoder.fit_transform(y), encoder


def split_data(x, y, test_size, random_state):
    return train_test_split(x, y, test_size=test_size, random_state=random_state)


def vectorize(x_train, x_test):
    vectorizer = CountVectorizer()
    x_train_vec = vectorizer.fit_transform(x_train)
    x_test_vec = vectorizer.transform(x_test)
    return x_train_vec, x_test_vec, vectorizer


def train_model(x_train_vec, y_train, n_neighbors):
    model = KNeighborsClassifier(n_neighbors=n_neighbors)
    model.fit(x_train_vec, y_train)
    return model


def evaluate(model, x_test_vec, y_test, class_names):
    predictions = model.predict(x_test_vec)
    print("Acurácia:", accuracy_score(y_test, predictions))
    print(classification_report(y_test, predictions, target_names=class_names))
    return predictions


def main():
    x, y = load_data(CSV_PATH, TEXT_COLUMN, LABEL_COLUMN)
    y_encoded, encoder = encode_labels(y)
    x_train, x_test, y_train, y_test = split_data(x, y_encoded, TEST_SIZE, RANDOM_STATE)
    x_train_vec, x_test_vec, vectorizer = vectorize(x_train, x_test)
    model = train_model(x_train_vec, y_train, N_NEIGHBORS)
    evaluate(model, x_test_vec, y_test, encoder.classes_)


if __name__ == "__main__":
    main()
'''