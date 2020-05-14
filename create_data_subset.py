import os

# Global variables
ZOOM_LEVEL = 10
TOP_LEFT_X = 159
TOP_LEFT_Y = 351
BOTTOM_RIGHT_X = 324
BOTTOM_RIGHT_Y = 438

def is_within_bounds(vals):
    return vals[1] >= TOP_LEFT_X and vals[1] <= BOTTOM_RIGHT_X and vals[2] >= TOP_LEFT_Y and vals[2] <= BOTTOM_RIGHT_Y

'''
    Iterates over OSM_Processed_Data, creating a subset
    belonging only to United States tiles at the defined
    zoom level.
'''
def main():
     
    
    input_path  = '/s/lattice-64/a/nobackup/galileo/OSM_Processed_Data/'
    output_path = './zoom_{0}_subset/'.format(ZOOM_LEVEL)
    files = os.listdir(input_path)

    if not os.path.exists(output_path):
        os.makedirs(output_path)    

        count = 0
        for filename in files:
            print('Processing file %d: %s...' % (count, filename))
            with open(input_path + filename, 'r') as file_in:
                with open(output_path + filename, 'w') as file_out:
        
                    for line in file_in:
                        vals = [int(element) for element in line.split(',')]
                        if vals[0] == ZOOM_LEVEL and is_within_bounds(vals):
                            file_out.write(line)
            
            count += 1 

    else:
        print('%s already exists!' % output_path)

if __name__ == '__main__':
    main()
