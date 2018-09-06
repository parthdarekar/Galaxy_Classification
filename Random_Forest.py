#Random Forest Classifier with different number of trees to predict the shape of the galaxy
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import cross_val_predict

def generate_features_targets(data):
    targets = data['class']
    features = np.empty(shape=(len(data), 13))

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

    # concentration in u filter
    features[:, 10] = (data['petroR50_u']) / (data['petroR90_u'])
    # concentration in r filter
    features[:, 11] = (data['petroR50_r']) / (data['petroR90_r'])
    # concentration in z filter
    features[:, 12] = (data['petroR50_z']) / (data['petroR90_z'])

    return features, targets

#getting predictions from a random forest classifier
def rf_predict_actual(data, n_estimators):
  # generate the features and targets
  features,targets=generate_features_targets(data)
  # instantiate a random forest classifier using n estimators
  rfc=RandomForestClassifier(n_estimators=n_estimators)
  # get predictions using 10-fold cross validation with cross_val_predict
  predicted=cross_val_predict(rfc,features,targets,cv=10)
  # return the predictions and their actual classes
  return predicted,targets

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



if __name__ == "__main__":
  data = np.load('galaxy_catalogue.npy')

  # get the predicted and actual classes
  number_estimators = [10,50,100,500]              # Number of trees
  for i in range(4):
      predicted, actual = rf_predict_actual(data, number_estimators[i])


      # calculate the model score
      accuracy = calculate_accuracy(predicted, actual)
      print("Accuracy score using a Random Forest Classifier with ",number_estimators[i]," trees:", accuracy)


