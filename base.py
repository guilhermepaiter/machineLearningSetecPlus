# import pandas as pd
# from sklearn.preprocessing import LabelEncoder
# from sklearn.model_selection import train_test_split
# from sklearn.feature_extraction.text import CountVectorizer
# from sklearn.neighbors import KNeighborsClassifier
# from sklearn import metrics

# def data():
#     data = pd.read_csv("satd_sentiment_3000.csv")

#     x = data['abstract']
#     y = data['sentiment_final']

#     return x, y
# # print(data['abstract'])

# def encoder(x, y):
#     labelEncoder = LabelEncoder();
#     yEncoded = labelEncoder.fit_transform(y);

#     xTrain, xTest, yTrain, yTest = train_test_split(x, yEncoded, test_size=0.3);

#     return xTrain, xTest, yTrain, yTest

# def vectorize(xTrain, xTest):
#     vectorizer = CountVectorizer();

#     xTrainVectorize = vectorizer.fit_transform(xTrain);
#     xTestVectorize = vectorizer.transform(xTest);

#     return xTrainVectorize, xTestVectorize

# def nearestNeighbor(yTrain, xTrainVectorize, xTestVectorize):
#     classifierKNN = KNeighborsClassifier(n_neighbors=1);

#     classifierKNN.fit(xTrainVectorize, yTrain);

#     resultsPredict = classifierKNN.predict(xTestVectorize);

#     return resultsPredict

# def metric(yTest, resultsPredict):
#     accuracy = metrics.accuracy_score(yTest, resultsPredict);
#     precision = metrics.precision_score(yTest, resultsPredict, average='binary');
#     recall = metrics.recall_score(yTest, resultsPredict, average='binary');
#     f1score = metrics.f1_score(yTest, resultsPredict, average='binary');

#     print("Acúracia = " + str(accuracy))
#     print("Precisão = " + str(precision))
#     print("Revocação = " + str(recall))
#     print("F1-Score = " + str(f1score))

# def main():
#     date = data()
#     encode = encoder()
#     vector = vectorize()
#     knn = nearestNeighbor()
#     score = metric()
    
# main()


import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.neighbors import KNeighborsClassifier
from sklearn import metrics


def load_data(path, text_col, label_col):
    data = pd.read_csv(path).dropna(subset=[text_col, label_col])
    return data[text_col], data[label_col]


def encode_labels(y):
    labelEncoder = LabelEncoder()
    yEncoded = labelEncoder.fit_transform(y)
    return yEncoded, labelEncoder


def split_data(x, y, test_size=0.3, random_state=42):
    return train_test_split(
        x, y, test_size=test_size, random_state=random_state, stratify=y
    )


def vectorize(xTrain, xTest):
    vectorizer = CountVectorizer()
    xTrainVec = vectorizer.fit_transform(xTrain)  # aprende o vocabulário
    xTestVec = vectorizer.transform(xTest)        # só aplica
    return xTrainVec, xTestVec


def nearest_neighbor(xTrainVec, yTrain, xTestVec, k=1):
    classifier = KNeighborsClassifier(n_neighbors=k)
    classifier.fit(xTrainVec, yTrain)
    return classifier.predict(xTestVec)


def metric(yTest, predictions, class_names):
    print(f"Acurácia = {metrics.accuracy_score(yTest, predictions):.4f}")
    print(metrics.classification_report(
        yTest, predictions, target_names=class_names
    ))


def main():
    x, y = load_data("satd_sentiment_3000.csv", "abstract", "sentiment_final")
    yEncoded, labelEncoder = encode_labels(y)
    xTrain, xTest, yTrain, yTest = split_data(x, yEncoded)
    xTrainVec, xTestVec = vectorize(xTrain, xTest)
    predictions = nearest_neighbor(xTrainVec, yTrain, xTestVec, k=1)
    metric(yTest, predictions, labelEncoder.classes_)


if __name__ == "__main__":
    main()
