import torch
import numpy as np
import os

def get_counts(filename):
	line_patterns = [
		'1,0,0,0,',
		'1,0,1,2,',
		'1,1,0,1,',
		'1,1,1,3,'
	]

	# Read first four lines of file, producing [c_i0, c_i2, c_i1, c_i3]
	with open(filename, 'r') as f:
		counts = []
		for tile_entry in range(4):
			line = f.readline().strip()

			# Handle bad data
			if line[:8] == line_patterns[tile_entry]:
				vals = [int(x) for x in line.split(',')[4:]] # Only capture counts
				counts.append(vals[0])
			else:
				counts.append(0)

		return counts


'''
Returns training, validation, and test sets, each of which contain:
[
	[
		tensor([ <-- window_size (i) * count vectors
			[c_00, c_02, c_01, c_03],
			[c_10, c_12, c_11, c_13],
			...
			[c_i0, c_i2, c_i1, c_i3]
		]), 
		tensor([c_(i+1)0, c_(i+1)2, c_(i+1)1, c_(i+1)3]) <-- label for i+1 count vector
	],
	...
	[]
]
'''
def get_data_sets(window_size):
	train_features = []
	train_targets = []
	validation_features = []
	validation_targets = []
	test_features = []
	test_targets = []

	window_start = 0
	window_end = window_size


	files = os.listdir('./zoom_1_subset')  # Sort by date
	files.sort()
	while window_end < (len(files) - 1):

		input_vector = []

		# Read input_size files
		for i in range(window_start, window_end):
			file = './zoom_1_subset/' + files[i]
			input_vector.append(get_counts(file))

		# Create tensor from input_size vectors?
		numpy_input = np.array(input_vector)
		tensor_input = torch.from_numpy(numpy_input).float()

		file = './zoom_1_subset/' + files[window_end]
		numpy_output = np.array(get_counts(file))
		tensor_output = torch.from_numpy(numpy_output).float()

		year = int(file[22:26])

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
