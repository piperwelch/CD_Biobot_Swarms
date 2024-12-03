'''
class for a single anthrobot
'''

import numpy as np
import pickle 

class Bot():

    def __init__(self):

        self.body = self.make_sphere()
        # self.set_cilia_forces()
        

    def make_sphere(self):
        #TODO
        pass


    def get_surface_cell_coords(self, arr):
        # Get the coordinates of cells on the surface of the bot
        xs = []
        ys = []
        zs = []
        neigh = self.get_neighbors(arr)

        for x in range(arr.shape[0]):
            for y in range(arr.shape[1]):
                for z in range(arr.shape[2]):
                    if arr[x,y,z]==1 and np.sum(neigh[x,y,z,:])<6:
                        xs.append(x)
                        ys.append(y)
                        zs.append(z)

        return xs,ys,zs

    def get_neighbors(self, a):
        b = np.pad(a, pad_width=1, mode='constant', constant_values=0)
        neigh = np.concatenate((
            b[2:, 1:-1, 1:-1, None], b[:-2, 1:-1, 1:-1, None],
            b[1:-1, 2:, 1:-1, None], b[1:-1, :-2, 1:-1, None],
            b[1:-1, 1:-1, 2:, None], b[1:-1, 1:-1, :-2, None]), axis=3)
        return neigh

    def get_empty_neighbor_positons(self,body,pos):
        empty_neigh = []
        for direction in [self.front,self.back,self.left,self.right]:
            neigh_pos = direction(pos[0],pos[1],pos[2])

            # Checking if the neighboring voxels is in array bounds
            if neigh_pos[0]>=0 and neigh_pos[0]<body.shape[0] and neigh_pos[1]>=0 and neigh_pos[1]<body.shape[1]:

                if body[neigh_pos] == 0: # in bounds and empty
                    empty_neigh.append(neigh_pos)

            else: # out of array bounds so by default there is an empty neighbor
                empty_neigh.append(neigh_pos)

        return empty_neigh


    def generate_restricted_cilia_forces(self):
        '''
        Assumes the body array is a matrix of 0s (empty), 1s (passive voxels), and 2s (ciliated cells)
        '''
        # Assumes voxels do not have cilia forces in the z-direction
        # Straight --> cilia force will be at a 90 degree angle from the surface (i.e. just chooses
        # a unit vector to an empty neighbor)
        # if align_vec is not None and straight=True, and a cell has more than one empty neighbor
        # the straight vector is chosen as determined by the align_vec

        self.cilia = np.zeros((self.body.shape[0], self.body.shape[1], self.body.shape[2], 3))

        rad45 = (45/180)*np.pi
        
        # iterate through ciliated cells
        for r in range(self.body.shape[0]):
            for c in range(self.body.shape[1]):
                for z in range(self.body.shape[2]):
                    
                    if self.body[r,c,z]==2:

                        curr_pos = (r,c,z)
                        
                        # Get neighboring empty voxel locations
                        empty_neigh = self.get_empty_neighbor_positons(self.body,curr_pos)

                        # Compute vectors to directions of the empty neighbors
                        vectors = []
                        for empty_neigh_pos in empty_neigh:

                            x_comp = curr_pos[1] - empty_neigh_pos[1]
                            y_comp = curr_pos[0] - empty_neigh_pos[0]
                            z_comp = curr_pos[2] - empty_neigh_pos[2]# should always be 0
                            assert z_comp==0
                            
                            # by default all of the vectors are unit vectors because the distance between voxels is 1
                            vectors.append([x_comp,y_comp,z_comp])
                        
                        
                        # Compute range of angles the cilia force vector can lie in
                        # +/- 45 degrees of the vector to each empty neighboring voxel
                        if len(vectors)>0:
                            bounds = []
                            for vector in vectors:

                                # print("VECTOR:",vector)

                                angle_in_radians = np.arctan2(vector[1],vector[0])
                                lb = angle_in_radians-rad45 # lower bound
                                ub = angle_in_radians+rad45 # upper bound
                                
                                bounds.append([lb,ub])


                            # first choose a random range to choose from if more than one 
                            # important if the ranges are not touching
                            # i.e. missing voxels to the left and right but not up and down
                            range_index = np.random.randint(len(bounds))
                            angle_range = bounds[range_index]

                            # Choose a random angle in the range (on the unit circle)
                            cilia_force_angle = (angle_range[1] - angle_range[0]) * np.random.random() + angle_range[0]
                            
                            # Compute the x and y components of the unit vector given the chosen angle
                            cilia_x_comp = np.cos(cilia_force_angle) # row is really the y comp 
                            cilia_y_comp = np.sin(cilia_force_angle) # c is really the x comp
                            cilia_z_comp = 0
                            cilia_force_vec = [cilia_x_comp, cilia_y_comp, cilia_z_comp]                      
                            
                            self.cilia[r,c,z,:] = cilia_force_vec

    def front(self, x,y,z,d=1):
        return x,y-d,z 

    def back(self, x,y,z,d=1):
        return x,y+d,z

    def left(self, x,y,z,d=1):
        return x-d,y,z

    def right(self, x,y,z,d=1):
        return x+d,y,z