import os


def initial_tile(zoom_level):
    return '0'*zoom_level

'''
    Replaces a positional character within a string.
'''
def change_char(string, position, replacement):
    return string[:position]+replacement+string[position+1:]

'''
    Increments a tile id.
    I.e. 1123333 -> 1130000
'''
def increment(tile_id):
    for i in range(len(tile_id)-1, -1, -1):
        quad_digit = int(tile_id[i])
        if quad_digit < 3:
            tile_id = change_char(tile_id, i, str(quad_digit+1))
            break
        else:
            tile_id = change_char(tile_id, i, '0')
        
    return tile_id

'''
    Replaces missing counts for tiles with 0.
'''
def replace_empty_records(tiles, zoom_level):
    print('Replacing empty records')
    tile_id = initial_tile(zoom_level)
    for i in range(0, 4**zoom_level):
        if tile_id not in tiles:
            tiles[tile_id] = 0
        
        tile_id = increment(tile_id)
    
'''
    Reads a file and returns tile dictionary mapping
    tile id -> count. I.e. { '121113' : 84 }
'''
def read_tiles(filename, zoom_level):
    print('Reading %s' % filename)
    tiles = {} 
    with open(filename, 'r') as f:
        for line in f:
            vals = line.split(',')
            if int(vals[0]) == zoom_level:
                tiles[vals[3]] = int(vals[4])
    
    return tiles

'''
    Outputs all tile count records in sorted order from
    tiles dictionary to file.
'''
def output_tiles(filename, zoom_level, tiles):
    print('Writing %s' % filename)
    
    with open(filename, 'w') as f:
        tile_id = initial_tile(zoom_level) 
        
        for i in range(0, 4**zoom_level):
            if tile_id not in tiles:
                f.write('{0},{1}'.format(tile_id, '0'))
            else:
                f.write('{0},{1}'.format(tile_id, tiles[tile_id]))

            tile_id = increment(tile_id)


'''
    Extracts all tile counts for a single given zoom level.
'''
def preprocess_data(zoom_level):
    input_path  = '/s/lattice-64/a/nobackup/galileo/OSM_Processed_Data/'
    output_path = './zoom_{0}_subset/'.format(zoom_level)
    files = os.listdir(input_path)
    
    if not os.path.exists(output_path):
        os.makedirs(output_path)    

        count = 0
        for filename in files:
            print('Processing file %d: %s...' % (count, filename))
            tiles = read_tiles(input_path + filename, zoom_level)
            # replace_empty_records(tiles, zoom_level)      
            output_tiles(output_path + filename, zoom_level, tiles) 
            
            if count == 3:
                break

            count += 1

    else:
        print('%s already exists!' % output_path)




def main():
    ZOOM_LEVEL = 16 
    preprocess_data(ZOOM_LEVEL)
  


if __name__ == '__main__':
    main()
