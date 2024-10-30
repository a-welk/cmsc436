# Alex Welk - Ethan Scott - CMSC 436 - Project 2
from matplotlib.pyplot import subplots, show
import random
import math
from sys import maxsize


# Normalize values between a minimum and maximum
def normalize(value, min_val, max_val):
    return (value - min_val) / (max_val - min_val)


# Applies a unipolar activation
def activation_function(net, k=1e-2):
    return 1 / (1 + math.exp(-k * net))


# Simple sign activation function
def sign(net):
    if net >=0:
        return 1
    else:
        return 0

# Split data for training and testing
def split_data(car_list):
    train_data, test_data = [], []
    for i, car in enumerate(car_list):
        (train_data if i % 4 != 0 else test_data).append([*car[:2], 1, car[2]])
    return train_data, test_data


# Hard activation training for neuron
def hard_activation_training(group, training_patterns, size):
    if group == 1: # group A
        learning_rate = 0.2 # alpha
        target_error_threshold = 10e-5
    elif group == 2: # group B
        learning_rate = 0.00118
        target_error_threshold = 40
    else: # group C
        if size == 1:
            learning_rate = 2e-5
            target_error_threshold = 700
        elif size == 2:
            learning_rate = 0.01
            target_error_threshold = 330
    
    minimum_error = maxsize
    total_iterations = 5000
    weights = [random.uniform(-0.5, 0.5) for _ in range(3)]
    desired_output = [pattern[3] for pattern in training_patterns]
    for iteration in range(total_iterations):
        actual_output = [0] * len(training_patterns)
        total_error = 0.0
        for pattern_index, pattern in enumerate(training_patterns):
            net_input = 0  # Initialize net input
            for i in range(len(weights)):  # Loop over each weight and input in the pattern
                net_input += weights[i] * pattern[i]  # Multiply weight and input, accumulate in net_input
            actual_output[pattern_index] = sign(net_input)
            error = desired_output[pattern_index] - actual_output[pattern_index]
            total_error += error ** 2
            learning_rate_adjustment = learning_rate * error
            for i in range(len(weights)):  # Loop over each weight and input in the pattern
                weights[i] = weights[i] + learning_rate_adjustment * pattern[i]  # Update each weight

        if total_error < target_error_threshold: # if error is already below threshold return, otherwise keep looping
            return [total_error, weights, iteration, total_error]
        minimum_error = min(minimum_error, total_error)

    return [total_error, weights, total_iterations, minimum_error]


# Soft activation training for neuron
def soft_activation_training(group, training_patterns, size):
    if group == 1:
        learning_rate = 0.2
        k = 0.2
        target_error_threshold = 10e-5
    elif group == 2:
        learning_rate = 0.2
        k = 0.01
        target_error_threshold = 40
    else:
        if size == 1:
            learning_rate = 2e-5
            k = 0.2
            target_error_threshold = 700
        elif size == 2:
            learning_rate = 0.01
            k = 5
            target_error_threshold = 177

    minimum_error = maxsize
    weights = [random.uniform(-0.5, 0.5) for j in range(3)]
    desired_output = [pattern[3] for pattern in training_patterns]
    total_iterations = 5000

    for iteration in range(total_iterations):
        actual_output = [0] * len(training_patterns)
        total_error = 0.0

        for pattern_index, pattern in enumerate(training_patterns):
            net_input = 0  # Initialize net input
            for i in range(len(weights)):  # Loop over each weight and input in the pattern
                net_input += weights[i] * pattern[i]  # Multiply weight and input, accumulate in net_input
            actual_output[pattern_index] = activation_function(net_input, k)
            error = desired_output[pattern_index] - actual_output[pattern_index]
            total_error += error ** 2
            learning_rate_adjustment = learning_rate * error
            for i in range(len(weights)):  # Loop over each weight and input in the pattern
                weights[i] = weights[i] + learning_rate_adjustment * pattern[i]  # Update each weight


        if total_error < target_error_threshold: # if error is already below threshold return, otherwise keep looping
            return [total_error, weights, iteration, total_error]
        minimum_error = min(minimum_error, total_error)

    return [total_error, weights, total_iterations, minimum_error]


# End Utility Functions


# Initialize variables
weights_list, costs_list, categories, normalized_prices, normalized_weights = [], [], [], [], []
fig, ax = subplots()


# Allows user to input file choice rather than hardcoding file name
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

# Prompt user for further instruction
activation_choice = int(input("Enter 1 for Hard Activation. Enter 2 for Soft Activation\n"))
training_split_choice = int(input("Enter 1 to train 75% of data. Enter 2 to train 25% of data\n"))
plot_type_choice = int(input("Enter 1 to plot trained points. Enter 2 to plot test points\n"))

# Parse file content
for line in data_lines:
    parts = line.strip().split(',')
    weights_list.append(float(parts[1]))
    costs_list.append(float(parts[0]))
    categories.append(int(parts[2]))

# Set min/max for normalization
max_price = max(costs_list)
min_price = min(costs_list)
max_weight = max(weights_list)
min_weight = min(weights_list)

large_cars, small_cars = [], []

# Normalize data and separate by category
for weight, price, label in zip(weights_list, costs_list, categories):
    norm_weight = normalize(weight, min_weight, max_weight)
    norm_price = normalize(price, min_price, max_price)
    if label == 1:
        large_cars.append([norm_weight, norm_price, label])
    else:
        small_cars.append([norm_weight, norm_price, label])

# Split data into 75/25 
large_75, large_25 = split_data(large_cars)
small_75, small_25 = split_data(small_cars)

if training_split_choice == 1: # train 75%
    train_data = large_75 + small_75
else:  # train 25%
    train_data = large_25 + small_25
random.shuffle(train_data)


if activation_choice == 1: # train hard activation
    output = hard_activation_training(file_choice, train_data, training_split_choice)
else: # train soft activation
    output = soft_activation_training(file_choice, train_data, training_split_choice)

# Display training results
final_weights = output[1]
print(f"\nFinal Error: {output[0]}\n")
print(f"Trained Weights: {final_weights}\n")
print(f"Iterations: {output[2]}\n")
print(f"Minimum Error: {output[-1]}")


# Plot classification line
x_intercept = final_weights[2] / final_weights[0]
y_intercept = final_weights[2] / final_weights[1]
ax.plot([abs(x_intercept), 0], [0, abs(y_intercept)])

# Scatter plot based on user choices
large_x, large_y, small_x, small_y = [], [], [], []
if plot_type_choice == 1: # plot trained
    if training_split_choice == 1: # trained 75% so plot the trained 75%
        for small_car, large_car in zip(small_75, large_75):
            large_x.append(large_car[0])
            large_y.append(large_car[1])
            small_x.append(small_car[0])
            small_y.append(small_car[1])
    else: # trained 25% so plot the trained 25%
        for small_car, large_car in zip(small_25, large_25):
            large_x.append(large_car[0])
            large_y.append(large_car[1])
            small_x.append(small_car[0])
            small_y.append(small_car[1])
    ax.scatter(large_x, large_y, c='pink')
    ax.scatter(small_x, small_y, c = 'green')
    
elif plot_type_choice == 2: # plot test
    if training_split_choice == 1: #trained 75% so plot test 25%
        for small_car, large_car in zip(small_25, large_25):
            large_x.append(large_car[0])
            large_y.append(large_car[1])
            small_x.append(small_car[0])
            small_y.append(small_car[1])
    else: # trained 25% so plot test 75%
        for small_car, large_car in zip(small_75, large_75):
            large_x.append(large_car[0])
            large_y.append(large_car[1])
            small_x.append(small_car[0])
            small_y.append(small_car[1])

    ax.scatter(large_x, large_y, c='pink')
    ax.scatter(small_x, small_y, c = 'green')

# Configure plot
ax.set_title(f'Normalized Weight vs Price for {file_name}')
ax.set_xlabel('Weight (pounds)')
ax.set_ylabel('Price (USD)')

# Initialize confusion matrix variables
true_pos, true_neg, false_pos, false_neg = 0, 0, 0, 0

reg_line = final_weights

# Compute confusion matrix values for large cars
for car_weight, cost, label in zip(large_x, large_y, [1]*len(large_y)):
    total = car_weight * reg_line[0] + cost * reg_line[1] + reg_line[2]

    if total >= 0: 
        if label == 1:
            true_pos += 1 
        else:
            false_neg += 1  
    else:
        if label == 0:
            true_neg += 1  
        else:
            false_pos += 1 

# Compute confusion matrix values for small cars
for car_weight, cost, label in zip(small_x, small_y, [0]*len(small_x)):
    total = car_weight * reg_line[0] + cost * reg_line[1] + reg_line[2]

    if total >= 0: 
        if label == 1:
            true_pos += 1 
        else:
            false_neg += 1 
    else: 
        if label == 0:
            true_neg += 1
        else:
            false_pos += 1

# Print confusion matrix results
print(f"Confusion Matrix Values: \n")
print(f"True Positive: {true_pos}\n")
print(f"True Negative: {true_neg}\n")
print(f"False Positive: {false_pos}\n")
print(f"False Negative: {false_neg}\n")

# Calculate rates based on the confusion matrix
if float(true_pos + true_neg + false_neg + false_pos) == 0:
    accuracy = 0
else:
    accuracy = float(true_pos + true_neg) / float(true_pos + true_neg + false_neg + false_pos)

error = 1.0 - accuracy

#  Ensures clean division in cases that denominator is 0
if float(true_pos + false_neg) == 0:
    true_pos_rate = 0
else:
    true_pos_rate = float(true_pos) / float(true_pos + false_neg)
if float(true_neg + false_pos) == 0:
    true_neg_rate = 0
else:
    true_neg_rate = float(true_neg) / float(true_neg + false_pos)
if float(true_neg+false_pos) == 0:
    false_pos_rate = 0
else:
    false_pos_rate = float(false_pos) / float(true_neg + false_pos)
if float(true_pos + false_neg) == 0:
    false_neg_rate = 0
else:
    false_neg_rate = float(false_neg) / float(true_pos + false_neg)

# Print calculated rates
print(f"Accuracy Rate: {accuracy*100.0}%\n")
print(f"Error Rate: {error*100.0}%\n")
print(f"True Positive Rate: {true_pos_rate*100.0}%\n")
print(f"True Negative Rate: {true_neg_rate*100.0}%\n")
print(f"False Positive Rate: {false_pos_rate*100.0}%\n")
print(f"False Negative Rate: {false_neg_rate*100.0}%\n")

# Display plot
show()
