# Imports
import sys
import numpy as np
import spkmeans as spk

# Constants
MIN_ARGUMENTS = 3
INVALID_INPUT_MSG = "Invalid Input!"
ERROR_MSG = "An Error Has Occured"
COMMA = ','
NEG_ZERO_LOWER_BOUND = -0.00005
GOALS = ["spk", "jacobi", "wam", "ddg", "lnorm"]


# The main algorithm - Spectral clustering.
# Prints the corresponding result for each goal in GOALS
def main():
    # Read and valid user input
    k, goal, file = validate_and_assign_input_user()
    list_of_vectors = build_vectors_list(file)
    n_vectors = len(list_of_vectors)
    n_features = len(list_of_vectors[0])
    if k >= n_vectors:
        print(INVALID_INPUT_MSG)
        exit()  # End program k >= n

    try:
        if goal != "jacobi":
            calc_matrix = spk.calc_mat(list_of_vectors, goal, k, n_features, n_vectors)
            if goal == "spk":
                if k == 0:  # K not provided - The Eigengap Heuristic result == T's n_features
                    k = len(calc_matrix[0])
                # Kmeans++
                list_random_init_centrals_indexes = choose_random_centrals(calc_matrix, k)
                calc_matrix, vec_to_cluster_labeling = spk.kmeans(calc_matrix, n_vectors, k, k,
                                                                  list_random_init_centrals_indexes)
                print(*list_random_init_centrals_indexes, sep=COMMA)
            print_matrix(calc_matrix)  # Print matrix according to the goal
        else:  # goal == "jacobi"
            eigen_matrix, eigen_values = spk.jacobi(list_of_vectors, n_vectors)
            print_matrix([eigen_values] + eigen_matrix)
    except Exception:
        print(ERROR_MSG)
        exit(1)


# Validates and return the user input - k, goal and filename as tuple
def validate_and_assign_input_user():
    if len(sys.argv) < MIN_ARGUMENTS + 1 or (not sys.argv[1].isdigit()):
        print(INVALID_INPUT_MSG)
        exit()  # End program, min arguments/not valid type
    k = int(sys.argv[1])
    goal = sys.argv[2]
    file = sys.argv[3]
    if goal not in GOALS or (k < 0 and goal == "spk"):
        print(INVALID_INPUT_MSG)
        exit()  # End program, not valid k/goal
    return k, goal, file


# The function read from csv format file (extension .txt/.csv) into matrix.
# file - the csv filename/filepath
# return: Reading result as list of lists (matrix)
def build_vectors_list(file):
    try:
        with open(file, 'r') as read_file:
            return [[float(x) for x in row.split(COMMA)] for row in read_file]
    except IOError as err:
        print(ERROR_MSG)
        exit(err.errno)


# Choosing the first k indexes to be the initial clusters' centroids for the kmeans algorithm.
# The indexes are randomly selected according to the probability function,
#   Which is constructed from the Euclidean norm from all other centroids.
# k - number of clusters
# Return: list of clusters first central indexes
def choose_random_centrals(list_of_vectors, k):
    np.random.seed(0)
    n_vectors = len(list_of_vectors)
    np_of_vectors = np.array(list_of_vectors)
    list_rand_init_centrals_indexes = [np.random.choice(n_vectors)]  # chose the first index randomally
    for i in range(k - 1):
        # Calculate the euclidean norm from the last chosen index to all other points
        np_subtract = np_of_vectors - np_of_vectors[list_rand_init_centrals_indexes[i]]
        np_norms = np.linalg.norm(np_subtract, axis=1) ** 2
        np_min_norms = np.minimum(np_min_norms, np_norms) if i > 0 else np_norms
        np_prob = np_min_norms / np_min_norms.sum()  # Normalize to get probability function
        list_rand_init_centrals_indexes.append(np.random.choice(n_vectors, p=np_prob))
    return list_rand_init_centrals_indexes


# Print matrix in csv format where floats formatted to 4 digits after the decimal point
def print_matrix(matrix):
    for row in matrix:
        print(*[f"{neg_zero(x):.4f}" for x in row], sep=COMMA)


# Print format -0.0000 into 0.0000 present
# x - A float number
# return: -x if in required range
def neg_zero(x):
    if NEG_ZERO_LOWER_BOUND < x < 0:
        return -x
    return x


# Define main() as the main function
if __name__ == '__main__':
    main()
