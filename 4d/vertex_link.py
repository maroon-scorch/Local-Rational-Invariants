from pointn import *

def vertex_set(squares):
    vertex = set()
    for sq in squares:
        for pt in sq:
            vertex.add(pt)
    return vertex

def find_neighbors(v, squares):
    record = []
    for center, sq in squares:
        for pt in sq:
            if pt == v:
              record.append((center, sq))
              continue
    return record  

def vertex_link(v, neighbors, n):
    # n is the dimension of the cube
    # Each face in the link is represented by the center of the cube
    # containing the face, and a vector pointing to it
    link = []
    for center, sq in neighbors:
        # reduced_sq = list(filter(lambda x: x != v, sq))
        diff = -1*(v.vec - center.vec)
        indices = np.nonzero(diff)
        
        for i in np.nditer(indices):
            template = np.zeros(n)
            template[i] = diff[i]
            link.append((center, template))

    print("Link: ", link)
    print(len(link))
    return link

if __name__ == "__main__":
    
    cube = [PointN([0, 0, 0]),
                    PointN([0, 0, 1]), PointN([1, 0, 0]), PointN([0, 1, 0]),
                    PointN([0, 1, 1]), PointN([1, 1, 0]), PointN([1, 0, 1]),
                    PointN([1, 1, 1])]
    new_cube = []
    for p in cube:
        q = translate(p, [0, 0, 1])
        new_cube.append(q)
    
    # A "square" is given by its center and its vertex types
    square_list = [(PointN([0.5, 0.5, 0.5]), cube), (PointN([0.5, 0.5, 1.5]), new_cube)]
    cube_list = [elt[1] for elt in square_list]
    vertices = vertex_set(cube_list)
    for v in vertices:
        if v == PointN([1, 1, 1]):
            neighbor = find_neighbors(v, square_list)
            print("Vertex: ", v)
            print("Number of Neighbors: ", len(neighbor))
            
            vertex_link(v, neighbor, 3)
    
    