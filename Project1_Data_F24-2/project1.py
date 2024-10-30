import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib.colors import Normalize
import pandas as pd;
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix

def read_csv_convert_to_numpy(fileName='groupB.txt'):

    df = pd.read_csv(fileName)
    #print(df)

    numpy_cost = df[['cost']].values
    numpy_weight = df[['weight']].values
    numpy_carOrTruck = df[['carOrTruck']].values

    return numpy_cost, numpy_weight, numpy_carOrTruck

def normalize(costs_list, numpy_weight, labels):
    sample_amount = len(costs_list)
    max_cost = max(costs_list)
    min_cost = min(costs_list)
    max_weight = max(weights_list)
    min_weight = min(weights_list)
    num_car = 0
    num_truck = 0
    normalized_car_cost = []
    normalized_car_weight = []
    normalized_truck_cost = []
    normalized_truck_weight = []

    # Loop through sample amount and normalize cost and weights from given normalization formula z(i) = (x(i) – min(x)) / (max(x) – min(x))
    for i in range(sample_amount):
        costs_list[i] = ((costs_list[i] - min_cost) / (max_cost - min_cost))
        numpy_weight[i] = ((numpy_weight[i] - min_weight) / (max_weight - min_weight))

        # Split cars and trucks into two different arrays to get different colors on graph
        if(labels[i] == 0):
            normalized_car_cost.append(costs_list[i])
            normalized_car_weight.append(numpy_weight[i])
        else:
            normalized_truck_cost.append(costs_list[i])
            normalized_truck_weight.append(numpy_weight[i])
    #print(f"normalized cost: {numpy_cost} ")
    #print(f"normalized weight: {numpy_weight}")
    return normalized_car_cost, normalized_car_weight, normalized_truck_cost, normalized_truck_weight


def plot_dataset(normalized_car_cost, normalized_car_weight, normalized_truck_cost, normalized_truck_weight):

    # Plot dataset
    plt.scatter(normalized_car_cost, normalized_car_weight, color = 'green')
    plt.scatter(normalized_truck_cost, normalized_truck_weight, color = 'red')
    plt.xlabel('Cost')
    plt.ylabel('Weight')

    plt.show()

def linear_separator(cost, weight, label):

    X = np.column_stack((cost, weight))
    y = label

    # Split the dataset into training and test sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Train
    model = LogisticRegression()
    model.fit(X_train, y_train)

    # Get the weights and threshold
    weights = model.coef_[0]
    threshold = model.intercept_[0]

    # Pass data to confusion function
    confusion_and_stats(model, X, y)

    print(f"Weights: {weights}")
    print(f"Threshold: {threshold}")

def confusion_and_stats(model, X_test, y_test):

    # Predict the labels on the test set
    y_predicted = model.predict(X_test)

    # Compute the confusion matrix
    conf_matrix = confusion_matrix(y_test, y_predicted)

    # Extract the true negative, false positive, false negative, and true positive from the confusion matrix
    TN, FP, FN, TP = conf_matrix.ravel()

    # Calculate Metrics
    accuracy = (TP + TN) / (TP + TN + FP + FN)
    error_rate = 1 - accuracy
    tpr = TP / (TP + FN)  # True Positive Rate
    tnr = TN / (TN + FP)  # True Negative Rate
    fpr = FP / (FP + TN)  # False Positive Rate
    fnr = FN / (FN + TP)  # False Negative Rate

    print(f"Accuracy: {accuracy}")
    print(f"Error Rate: {error_rate}")
    print(f"True Positive Rate: {tpr}")
    print(f"True Negative Rate: {tnr}")
    print(f"False Positive Rate: {fpr}")
    print(f"False Negative Rate: {fnr}")

    print("Confusion Matrix:")
    print(conf_matrix)



print("1. GroupA.txt\n2. GroupB.txt\n3. GroupC.txt\n")
file_choice = int(input("Enter the file number to open: "))

if file_choice == 1:
    file_name = "GroupA.txt"
elif file_choice == 2:
    file_name = "GroupB.txt"
elif file_choice == 3:
    file_name = "GroupC.txt"
else:
    print("Invalid File Option")
    exit()

# Read file content
with open(file_name) as f:
    data_lines = f.readlines()

weights_list, costs_list, categories, normalized_prices, normalized_weights = [], [], [], [], []
# Parse file content
for line in data_lines:
    parts = line.strip().split(',')
    weights_list.append(float(parts[1]))
    costs_list.append(float(parts[0]))
    categories.append(int(parts[2]))

normalized_car_cost, normalized_car_weight, normalized_truck_cost, normalized_truck_weight = normalize(costs_list, weights_list, categories);
plot_dataset(normalized_car_cost, normalized_car_weight, normalized_truck_cost, normalized_truck_weight)
normalized_car_cost.extend(normalized_truck_cost)
normalized_car_weight.extend(normalized_truck_weight)
linear_separator(normalized_car_cost, normalized_car_weight, categories)