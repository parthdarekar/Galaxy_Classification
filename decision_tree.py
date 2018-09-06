# Decision Tree Classifier to classify the galaxy shape.
# Mergers were the hardest to classify correctly as observed from the confusion matrix
import numpy as np
import plot_confusion_matrix as pcm
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import confusion_matrix
from matplotlib import pyplot as plt

def splitdata_train_test(data, fraction_training):
  # complete this function
  np.random.shuffle(data)
  split_index = int(fraction_training*len(data))
  return data[:split_index], data[split_index:]


def generate_features_targets(data):

    targets = data['class']
    features = np.empty(shape=(len(data), 13))
    # print(data.shape)
    # print(len(data))
    features[:, 0] = data['u-g']
    features[:, 1] = data['g-r']
    features[:, 2] = data['r-i']
    features[:, 3] = data['i-z']
    features[:, 4] = data['ecc']
    features[:, 5] = data['m4_u']
    features[:, 6] = data['m4_g']
    features[:, 7] = data['m4_r']
    features[:, 8] = data['m4_i']
    features[:, 9] = data['m4_z']

    #concentrations in the u, r and z filters
    # concentration in u filter
    features[:, 10] = (data['petroR50_u']) / (data['petroR90_u'])
    # concentration in r filter
    features[:, 11] = (data['petroR50_r']) / (data['petroR90_r'])
    # concentration in z filter
    features[:, 12] = (data['petroR50_z']) / (data['petroR90_z'])

    return features, targets


    #splitting the data set and training a decision tree classifier
def dtc_predict_actual(data):
     # split the data into training and testing sets using a training fraction of 0.7

     training_fraction = 0.7
     training, testing = splitdata_train_test(data, training_fraction)
     # generate the feature and targets for the training and test sets
     # i.e. train_features, train_targets, test_features, test_targets
     train_features, train_targets = generate_features_targets(training)
     test_features, test_targets = generate_features_targets(testing)

     # instantiate a decision tree classifier
     dtc = DecisionTreeClassifier()
     # train the classifier with the train_features and train_targets
     dtc.fit(train_features, train_targets)
     # get predictions for the test_features
     predictions = dtc.predict(test_features)
     # return the predictions and the test_targets
     return predictions, test_targets

#Function to calculate the accuracy of the model
def calculate_accuracy(predicted, actual):
    total_predictions = len(predicted)
    correct_predictions = 0
    for i in range(total_predictions):
        if predicted[i] == actual[i]:
            correct_predictions = correct_predictions + 1
    accuracy = correct_predictions / total_predictions
    # print (correct_predictions)
    # return accuracy
    #More efficient calculation-
    return sum(predicted == actual) / len(actual)


if __name__ == '__main__':
    data1 = np.load('galaxy_catalogue.npy')
    features,targets=generate_features_targets(data1)
    # print(data)
    predicted_class, actual_class = dtc_predict_actual(data1)

    # Print some of the initial results
    print("Some initial results...\n   predicted,  actual")
    for i in range(10):
        print("{}. {}, {}".format(i, predicted_class[i], actual_class[i]))

    model_score=calculate_accuracy(predicted_class,actual_class)
    print("The accuracy of the model using a Decision Tree Classifier is ",model_score)
    # calculate the models confusion matrix using sklearns confusion_matrix function
    class_labels = list(set(targets))

    model_cm = confusion_matrix(y_true=actual_class, y_pred=predicted_class, labels=class_labels)

    # Plot the confusion matrix using the provided functions.
    plt.figure()
    pcm.plot_confusion_matrix(model_cm, classes=class_labels, normalize=False)
    plt.show()

