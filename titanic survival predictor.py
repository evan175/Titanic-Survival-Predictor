import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score
from sklearn.svm import SVC
from sklearn.neural_network import MLPClassifier
from sklearn.ensemble import RandomForestClassifier
import matplotlib.pyplot as plt

def kernel(X_train_split, y_train_split, X_val_split, y_val_split, X_test):
    #train svm model with RBF kernel on training data
    svm_model = SVC(C=2.0, kernel='rbf', random_state=42)
    svm_model.fit(X_train_split, y_train_split)

    #predict target variable from validation data
    y_val_pred = svm_model.predict(X_val_split)

    #print accuracy of model
    accuracy = accuracy_score(y_val_split, y_val_pred)
    print("Kernal accuracy:", accuracy)

    #predict target variable from test data
    y_test_pred = svm_model.predict(X_test)

    #add prediction to test_data
    test_data['Survived'] = y_test_pred
    test_data[['PassengerId', 'Survived']].to_csv('predictions_kernel.csv', index=False)

    return accuracy

def NN(X_train_split, y_train_split, X_val_split, y_val_split, X_test):
    #train neural network model using MLP classifier on training data
    mlp_model = MLPClassifier(hidden_layer_sizes=(64, 32), max_iter=1000, random_state=42)
    mlp_model.fit(X_train_split, y_train_split)
   
    #predict target variable from validation data
    y_val_pred = mlp_model.predict(X_val_split)

    #print accuracy of model
    accuracy = accuracy_score(y_val_split, y_val_pred)
    print("Neural network accuracy:", accuracy)

    #predict target variable from test data
    y_test_pred = mlp_model.predict(X_test)

    #add prediction to test_data
    test_data['Survived'] = y_test_pred
    test_data[['PassengerId', 'Survived']].to_csv('predictions_neural_network.csv', index=False)

    return accuracy

def tree(X_train_split, y_train_split, X_val_split, y_val_split, X_test):
    #train tree model with random forest classifier on training data
    rf_model = RandomForestClassifier(n_estimators=200, random_state=42)
    rf_model.fit(X_train_split, y_train_split)

    #predict target variable from validation data
    y_val_pred = rf_model.predict(X_val_split)

    #print accuracy of model
    accuracy = accuracy_score(y_val_split, y_val_pred)
    print("Tree accuracy:", accuracy)

    #predict target variable from test data
    y_test_pred = rf_model.predict(X_test)

    #add prediction to test_data
    test_data['Survived'] = y_test_pred
    test_data[['PassengerId', 'Survived']].to_csv('predictions_random_forest.csv', index=False)

    return accuracy

if __name__ == '__main__':
    #load in data with pandas
    train_data = pd.read_csv('train.csv')
    test_data = pd.read_csv('test.csv')

    #create variable for features and variable for targets from the training data
    X_train = train_data[['Pclass', 'Sex', 'Age', 'SibSp', 'Parch', 'Fare', 'Embarked']]
    y_train = train_data['Survived']

    #seperate features in test data
    X_test = test_data[['Pclass', 'Sex', 'Age', 'SibSp', 'Parch', 'Fare', 'Embarked']]

    #convert categorical variables to numerical
    X_train = pd.get_dummies(X_train, columns=['Sex', 'Embarked'], drop_first=True)
    X_test = pd.get_dummies(X_test, columns=['Sex', 'Embarked'], drop_first=True)

    #replace NaN values
    X_train.fillna(X_train.mean(), inplace=True)
    X_test.fillna(X_test.mean(), inplace=True)

    #standard normalization of data
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)

    #split training data into training and validation data
    X_train_split, X_val_split, y_train_split, y_val_split = train_test_split(X_train, y_train, test_size=0.2, random_state=42)

    kernal_accuracy = kernel(X_train_split, y_train_split, X_val_split, y_val_split, X_test)
    NN_accuracy = NN(X_train_split, y_train_split, X_val_split, y_val_split, X_test)
    tree_accuracy = tree(X_train_split, y_train_split, X_val_split, y_val_split, X_test)

    accuracies = [kernal_accuracy, NN_accuracy, tree_accuracy]

    models = ['Kernel SVM', 'Neural Network', 'Random Forest']

    plt.bar(models, accuracies, color=['blue', 'orange', 'green'])
    plt.xlabel('Models')
    plt.ylabel('Accuracy')
    plt.title('Model Performance Comparison')
    plt.ylim([0, 1])  # Set the y-axis limit to be between 0 and 1 for accuracy
    plt.show()
