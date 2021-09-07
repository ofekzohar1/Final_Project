import sys
import numpy as np
import csv
import timeit
import pandas as pd
#  import spkmeans

DEFAULT_ITER = 300
MIN_ARGUMENTS = 3
INVALID_INPUT_MSG = "Invalid Input!"
ERROR_MSG = "An Error Has Occured"
SECOND_FILE_INDEX_IN_ARGV = 3
COMMA = ','
GOALS = ["spk", "jacobi", "wam", "ddg", "lnorm"]


# Validates the arguments, assigns the vectors data and a list of the indexes for the first clusters centrals.
# Using fit function from the module we build in C
def main():
    k, goal, file = validate_and_assign_input_user()
    list_of_vectors = build_vectors_list(file)


# validates the users input (amount of arguments, receiving int when needed) and assigns it to its character
# using (MIN_ARGUMENTS + 1) - there is really minimum of 4 arguments but it is more comfortable and intuitive this way
def validate_and_assign_input_user():
    if len(sys.argv) < MIN_ARGUMENTS + 1 or (not sys.argv[1].isdigit()):
        print(INVALID_INPUT_MSG)
        exit()  # End program, min arguments/not valid type
    k = int(sys.argv[1])
    goal = sys.argv[2]
    file = sys.argv[3]
    if k < 0 or goal not in GOALS:
        print(INVALID_INPUT_MSG)
        exit()  # End program, not valid k/goal
    return k, goal, file


#  Validates file arguments and assigns them, combines both files in to one panda with inner join
#  Returns a panda that is holding the combined files data
def build_vectors_list(file):
    # read csv file as a list of lists
    try:
        with open(file, 'r') as read_file:
            return [[float(x) for x in row.split(COMMA)] for row in read_file]
    except FileNotFoundError as err:
        print(INVALID_INPUT_MSG)
        exit(err.errno)


# Receives the vectors pandas and amount of k clusters and builds a list of clusters
# Returns a list of clusters first central indexes
# Function uses the distances from a vector to the closest central
#   to pick the next central with random.choice with a probability by that distance
def choose_random_centrals(np_of_vectors, k, num_of_vectors):
    np.random.seed(0)
    list_random_init_centrals_indexes = [np.random.choice(num_of_vectors)]
    for i in range(k - 1):
        np_subtract = np_of_vectors - np_of_vectors[list_random_init_centrals_indexes[i]]
        np_norms = np.linalg.norm(np_subtract, axis=1) ** 2
        np_min_norms = np.minimum(np_min_norms, np_norms) if i > 0 else np_norms
        np_prob = np_min_norms / np_min_norms.sum()
        list_random_init_centrals_indexes.append(np.random.choice(num_of_vectors, p=np_prob))
    return list_random_init_centrals_indexes


# prints new central after adjusting for the relevant structure
def print_centrals(final_centroids_list):
    np_centroids = np.round(final_centroids_list, decimals=4)
    for central in np_centroids:
        print(*central, sep=COMMA)


if __name__ == '__main__':
    main()
