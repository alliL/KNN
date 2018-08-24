import matplotlib.pyplot as plt
from operator import itemgetter
import random as r
import math

# generate points within x and y range: [-10, 10]
def generate_points(num_points):
    return [[r.randrange(start = -10, stop = 11),
             r.randrange(start = -10, stop = 11)] for i in range(num_points)]

# classifies training points as above (True) or below (False) the sine wave
def classify(points):
    return [[x, y, math.sin(x * 2 * math.pi / 10) * 5 < y]
            for [x, y] in points]

# returns k indices of the train point distances closest to the given test point
def knn_classifier(k, train_points, test_point):
    test_x = test_point[0]
    test_y = test_point[1]
    distances = [math.sqrt((test_x - train_x1) ** 2 + (test_y - train_y1) ** 2)
                 for train_x1, train_y1 in train_points]

    # find k indices with the shortest distance to the test point
    indexed_distances = list(enumerate(distances))
    indexed_distances.sort(key = itemgetter(1))
    return [indexed_distances[index][0] for index in range(k)]

k = 5

# draw a graph with x-range [-10,10], y-range [-10,10]
# with a sin(x * 2pi/10)*5
x = list(range(-10, 11))
y = [math.sin(i * 2.0 * math.pi / 10.0) * 5.0 for i in x]
plt.plot(x, y)

# create the training points and test points
train_points = generate_points(9)
test_points = generate_points(9)

# Insert num_points randomly within [-10, 10],[-10,10]. Label them True or
# False depending on whether they're above or below the sine wave.
plt.scatter([x[0] for x in train_points], [y[1] for y in train_points])
plt.show()

classified_train_points = classify(train_points)
classified_test_points = classify(test_points)

# classifies each test point based on KNN
tested_classifications = []
for test_point in test_points:
    indices = knn_classifier(k, train_points, test_point)
    train_true_values = [classified_train_points[index][2] for index in indices]
    classification = sum(train_true_values) > k - sum(train_true_values)
    tested_classifications.append(classification)

# calculate percent of calculations that are correct
correct = sum([classified_test_points[index][2] == tested_classifications[index]
               for index in range(len(classified_test_points))])
percent_correct = correct / len(tested_classifications) * 100
print(correct / len(tested_classifications) * 100, "percent of the classifications are correct.")