## Network analysis. Use graph theory to model each test subject's eye gaze for a single image, where vertices represent objects present in image, and edges represent
## eye-gaze transitions (saccades) between two distinct objects.

## TO DO: Code up the eigenvector centrality measure. The centrality measures serve as a method to quantify object significance in images, which corresponds to viewing
## patterns and preferences for individuals with and without ASD diagnoses.

from skimage.io import imread, imshow
from random import randint

color_list = [[0, 0, 0], [224, 224, 192], [128, 0, 0], [128, 0, 128], [192, 128, 128], [128, 192, 0], [0, 128, 0],
              [192, 0, 128], [192, 0, 0], [64, 0, 128], [128, 128, 0], [128, 64, 0], [64, 0, 0], [64, 128, 0],
              [0, 64, 128], [64, 128, 128], [0, 128, 128], [0, 0, 128], [128, 128, 128], [192, 128, 0], [0, 64, 0],
              [0, 192, 0]]
class_list = ['other', 'outline', 'aeroplane', 'bottle', 'person', 'train', 'bicycle', 'horse', 'chair', 'dog', 'bird',
              'sheep', 'cat', 'cow', 'TV/monitor', 'motorbike', 'bus', 'boat', 'car', 'dining table', 'potted plant',
              'sofa']

# Create an adjacency matrix based on the data in coordinates and image #
def adjacency_matrix(image, coordinates, output=False):
    '''
    Computes adjacency matrix from eye-gaze data, coordinates, for test subject viewing image, img
    
    adjacency_matrix(image, coordinates, output=False) takes in a numpy array, image, and list of 2-tuples, coordinates
    Returns matrix representation of graph obtained from the eye-gaze data, coordinates, and image
    
    Parameters:
      image: numpy array
      coordinates: list of 2-tuples of int
      output: boolean (initially "False")
    
    Returns:
    list of lists of int
      adjacency matrix representation of graph obatined from coordinates and image
    '''
    
    objects = [] # Store object class labels here
    matrix = [] # Adjacency matrix to return
    previous_node = 0  # Keep track of the node from which connection is formed
    for coord in coordinates:
        current_color = image[coord[0], coord[1]].tolist()[:-1] # Get the color of the given pixel
        current_object = class_list[color_list.index(current_color)] # Current object class is current node
        if len(objects) == 0:  # This means image is viewed for the first time
            matrix.append([0])  # Initialize adjacency matrix
            objects.append(current_object)
        else:
            if current_object not in objects: # We come across a new type of object in image
                current_node = len(objects)  # Obtain node which corresponds to current object class
                for row in matrix:
                    row.append(0)
                matrix.append([0] * len(matrix[0]))  # Add another row of length equal to the other rows in matrix
                objects.append(current_object)
            else:
                current_node = objects.index(current_object)
            matrix[previous_node][current_node] += 1
            if current_node != previous_node: # If each dimension of matrix is greater than 1
                matrix[current_node][previous_node] += 1
            previous_node = current_node
    if output: # 'output' is a parameter that can be set to 'True' if matrix is to be printed out
        print('Objects:')
        for o in objects:
            print('=>', o)
        print('\nAdjacency Matrix:')
        for row in matrix:
            print(row)

    return objects, matrix


def degree_centrality(image, coordinates, output=False):
    '''
    Computes degree centrality for each node in the graph representation of coordinates and image
    
    degree_centrality(image, coordinates, output=False) sums up the number of connections to other nodes for each node in the graph
    (The graph is based on image and coordinates)
    Returns a dictionary, where each key is a node in the graph, and the value for each key is the degree centrality of the corresponding node
    
    Parameters:
      image: numpy array
      coordinates: list of 2-tuples of int
      output: boolean (initially "False")
    
    Returns:
    dict
      dictionary containing degree centrality values for each node in the graph obtained from coordinates and image

    '''
    
    values = {}
    object_classes, graph = adjacency_matrix(image, coordinates, output=False)
    n = len(object_classes)
    if n >= 2:
        for o in object_classes:
            values[o] = sum(graph[object_classes.index(o)])
    if output:
        print('\nDegree Centralities:')
        for v in values:
            print('=>', v, ':', values[v])

    return values


def get_path_list(g, start, end, l, m):
    '''
    Appends all paths from start to end in graph g to list m
    
    get_path_list(g, start, end, l, m) takes in a graph, g, two nodes in g (start, end), and two lists (l, m), and computes all paths in g from start to end
    Returns list m, which is a list of all paths
    
    Parameters:
      g: list of lists of int
      start: int
      end: int
      l: list of int
      m: list of lists of int
    
    Returns:
    list of lists of int
      list of all possible paths between nodes start and end in graph g

    '''
    
    if g[start][end] >= 1:
        l.append(str(g[start][end])) # Store the frequency of this connection, along with the connection itself
        m.append(l)
    else:
        c = 0
        for s in g[start]:
            if c not in l and c != start and s >= 1:
                h = l.copy()
                h.append(c)
                get_path_list(g, c, end, h, m)
            c += 1


def betweenness_centrality(image, coordinates, output=False):
    '''
    Computes betweenness centrality for each node in the graph representation of coordinates and image
    
    betweenness_centrality(image, coordinates, output=False) computes the number of times that each node lies along the shortest path between two other nodes in the graph
    (The graph is based on image and coordinates)
    Returns a dictionary, where each key is a node in the graph, and the value for each key is the betweenness centrality of the corresponding node
    
    Parameters:
      image: numpy array
      coordinates: list of 2-tuples of int
      output: boolean (initially "False")
    
    Returns:
    dict
      dictionary containing betweenness centrality values for each node in the graph obtained from coordinates and image
      
    '''
    values = {}
    object_classes, graph = adjacency_matrix(image, coordinates, output=False)
    n = len(object_classes)
    for node in range(0, n):
        bc_value = 0
        for i in range(0, n):
            for j in range(0, n):
                if i != node and node != j and i != j:
                    l = [i]
                    m = []
                    get_path_list(graph, i, j, l, m)
                    if m is not []:
                        current_min = len(m[0])
                        for path in m[1:]:
                            if len(path) < current_min:
                                current_min = len(path)
                        for path in m:
                            if len(path) > current_min:
                                m.remove(path)
                            elif node in path:
                                bc_value += int(path[-1]) # This value is the string that we stored earlier
        o = object_classes[node]
        values[o] = bc_value
    if output:
        print('\nBetweenness Centralities:')
        for v in values:
            print('=>', v, ':', values[v])
    return values


def closeness_centrality(image, coordinates, output=False):
    '''
    Computes closeness centrality for each node in the graph representation of coordinates and image
    
    closeness_centrality(image, coordinates, output=False) computes the total distance from the other nodes in the graph for each node
    (The graph is based on image and coordinates)
    Returns a dictionary, where each key is a node in the graph, and the value for each key is the closeness centrality of the corresponding node
    Note: the closeness centrality values can be rescaled by dividing by (n-1), to get the average geodesic distance instead of the total distance
    
    Parameters:
      image: numpy array
      coordinates: list of 2-tuples of int
      output: boolean (initially "False")
    
    Returns:
    dict
      dictionary containing closeness centrality values for each node in the graph obtained from coordinates and image

    '''
    values = {}
    object_classes, graph = adjacency_matrix(image, coordinates, output=False)
    n = len(object_classes)
    for node in range(0, n):
        total_distance = 0
        for i in range(0, n):
            if i != node:
                l = [node]
                m = []
                get_path_list(graph, node, i, l, m)
                current_min = len(m[0])
                for path in m[1:]:
                    if len(path) < current_min:
                        current_min = len(path)
                total_distance += current_min - 1 # (Due to the way our get_path_list function is coded)
        o = object_classes[node]
        values[o] = total_distance
    if output:
        print('\nCloseness Centralities:')
        for v in values:
            print('=>', v, ':', values[v])
    return values

# The following code demonstrates how the centrality measures work with a sample image and set of pseudorandomly generated coordinates #

# coordinate_list = []
# img = imread(r'C:\Users\hmlee\Documents\VOC2012\train_masks\2008_000074.png')

# for i in range(0, 10):
#     x = randint(1, 300)
#     y = randint(1, 300)
#     coordinate_list.append((x, y))

# adjacency_matrix(img, coordinate_list, output=True)
# degree_centrality(img, coordinate_list, output=True)
# betweenness_centrality(img, coordinate_list, output=True)
# closeness_centrality(img, coordinate_list, output=True)
