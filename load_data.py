import torch
import numpy as np
import os

'''
Loads the data from all days for a given area at a given predefined zoom level.
E.g. For zoom level 10, the United States has top-left x,y of [159,351], and 
bottom-right x,y of [324,438]:

159,351              -->
     ___________________
    |                   |
    |                   |
    |                   |
    |                   |
 |  |                   |
 v  |___________________|
                        324,438
'''


# GLOBAL VARIABLES
TOP_LEFT_CORNER_X = 159
TOP_LEFT_CORNER_Y = 351
BOTTOM_RIGHT_CORNER_X = 324
BOTTOM_RIGHT_CORNER_Y = 438
ZOOM_LEVEL = 10
X_WIDTH  = BOTTOM_RIGHT_CORNER_X - TOP_LEFT_CORNER_X # 165
Y_HEIGHT = BOTTOM_RIGHT_CORNER_Y - TOP_LEFT_CORNER_Y # 87


'''
    Finds max count across all files. This is used to normalize all other counts
    between 0 and 1.
'''
def find_max_count(directory):
    print("Finding max count for normalization...")

    index = 0
    max_count = 0
    files = os.listdir(directory)  # Sort by date
    for filename in files:
        if ((index+1)%10) == 0:
            print("\e[\rProcessing file %d/%d" % (index+1, 2096))
        elif index == 0:
            print("Processing file 0/2096")


        filename = directory + filename
        with open(filename, 'r') as f:
            for line in f:
                vals = [int(x) for x in line.split(',')]
                if vals[4] > max_count:
                    max_count = vals[4]
        index += 1


    print('Max count value of %d found.' % max_count)
    return max_count

'''
    Reads a file, containing with zoom,x,y,quadhash,count
    E.g. 10,159,352,0212211111,107

    Produces a matrix of float counts for entire area with data from file.
'''
def get_counts(filename, max_count):
    # Initialize matrix for all possible tiles within area
    #  specified by Y_HEIGHT and X_WIDTH (rows, cols)
    counts = np.zeros((Y_HEIGHT, X_WIDTH), dtype=float)

    with open(filename, 'r') as f:
        for line in f:
            line = f.readline().strip()

            if len(line) > 0:
                vals = [int(x) for x in line.split(',')] # Capture counts
                if vals[0] == ZOOM_LEVEL:
                    x = vals[1] - TOP_LEFT_CORNER_X - 1
                    y = vals[2] - TOP_LEFT_CORNER_Y - 1
                    count = float(vals[4]) / max_count # Normalize count with max_count

                    counts[y][x] = count

        return counts


'''
    Creates as input, a vector of 2D matrices.
    A single 2D matrix represents all the counts for a single day.
    For each day, window_size matrices k..k+n are added to the vector.
    For each vector, the label consists of a single matrix for day k+n+1.

    Returns torch stacked numpy arrays for train/validation/test sets.
'''
def get_data_sets(directory, window_size, max_count):
    cuda_available = torch.cuda.is_available()
    type = torch.FloatTensor
    if (cuda_available):
        type = torch.cuda.FloatTensor

    if cuda_available:
        print("CUDA is available")
    else:
        print("CUDA is not available")
    #print("CUDA %s available." % 'is' if cuda_available else 'is not')

    train_features = []
    train_targets = []
    validation_features = []
    validation_targets = []
    test_features = []
    test_targets = []

    window_start = 0
    window_end = window_size

    files = os.listdir(directory)
    files.sort()  # Sort by date
    while window_end < (len(files) - 1):

        if (window_start+1) % 10 == 0:
            print("Processing file %d/%d" % (window_start+1, 2096))

        # List of numpy matrices      
        input_vector = []

        # Read input_size files
        for i in range(window_start, window_end):
            file = directory + files[i]
            counts = get_counts(file, max_count)
            input_vector.append(counts)

        numpy_input = np.array(input_vector)
        tensor_input = torch.from_numpy(numpy_input).type(type)

        file = directory + files[window_end]
        counts = get_counts(file, max_count)
        numpy_output = np.array(counts)
        tensor_output = torch.from_numpy(numpy_output).type(type)

        year = int(file[23:27])

        if year < 2016:
            train_features.append(tensor_input)
            train_targets.append(tensor_output)
        elif year < 2018:
            validation_features.append(tensor_input)
            validation_targets.append(tensor_output)
        else:
            test_features.append(tensor_input)
            test_targets.append(tensor_output)

        window_start += 1
        window_end += 1

    return torch.stack(train_features), torch.stack(train_targets), torch.stack(validation_features), torch.stack(validation_targets), torch.stack(test_features), torch.stack(test_targets)
