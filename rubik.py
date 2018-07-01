import numpy as np


class Rubik:
    #Environment for Rubik's Cube
    def __init__(self):
        # F, B, R, L, U, D
        self.cube = np.zeros((6, 3, 3), dtype = np.int32)
        self.reset()

    def move_face(self, face, direction):
        #Just rotating face
        # 6 Faces F, B, R, L, U, D
        # 2 directions clockwise and anticlockwise
        copy_face = np.copy(self.cube[face])
        for i in range(3):
            self.cube[face][i] = copy_face[:, 2-i] if direction == 0 else copy_face[::-1, i]

    def move_ring(self, face, ring, direction):
        #For a face there are 3 rings
        #Rotate face layer == move_face(face, dir) + move_ring(face, 0, dir)
        #Rotate middle layer == move_ring(face, 1, dir)
        strip = np.zeros((4,3))
        if face == 0:
            strip[0] = np.copy(self.cube[4][2-ring,:])
            strip[1] = np.copy(self.cube[2][:,ring])
            strip[2] = np.copy(self.cube[5][ring,::-1])
            strip[3] = np.copy(self.cube[3][::-1,2-ring])

            add = 1 if direction == 0 else -1
            self.cube[4][2-ring,:] = strip[(0+add)%4]
            self.cube[2][:,ring] = strip[(1+add)%4]
            self.cube[5][ring,::-1] = strip[(2+add)%4]
            self.cube[3][::-1,2-ring] = strip[(3+add)%4]

        if face == 1:
            strip[0] = np.copy(self.cube[4][ring,::-1])
            strip[1] = np.copy(self.cube[3][:,ring])
            strip[2] = np.copy(self.cube[5][2-ring,:])
            strip[3] = np.copy(self.cube[2][::-1,2-ring])

            add = 1 if direction == 0 else -1
            self.cube[4][ring,::-1] = strip[(0+add)%4]
            self.cube[3][:,ring] = strip[(1+add)%4]
            self.cube[5][2-ring,:] = strip[(2+add)%4]
            self.cube[2][::-1,2-ring] = strip[(3+add)%4]

        if face == 2:
            strip[0] = np.copy(self.cube[4][::-1,2-ring])
            strip[1] = np.copy(self.cube[1][:,ring])
            strip[2] = np.copy(self.cube[5][::-1,2-ring])
            strip[3] = np.copy(self.cube[0][::-1,2-ring])

            add = 1 if direction == 0 else -1
            self.cube[4][::-1,2-ring] = strip[(0+add)%4]
            self.cube[1][:,ring] = strip[(1+add)%4]
            self.cube[5][::-1,2-ring] = strip[(2+add)%4]
            self.cube[0][::-1,2-ring] = strip[(3+add)%4]

        if face == 3:
            strip[0] = np.copy(self.cube[4][:,ring])
            strip[1] = np.copy(self.cube[0][:,ring])
            strip[2] = np.copy(self.cube[5][:,ring])
            strip[3] = np.copy(self.cube[1][::-1,2-ring])

            add = 1 if direction == 0 else -1
            self.cube[4][:,ring] = strip[(0+add)%4]
            self.cube[0][:,ring] = strip[(1+add)%4]
            self.cube[5][:,ring] = strip[(2+add)%4]
            self.cube[1][::-1,2-ring] = strip[(3+add)%4]
            
        if face == 4:
            strip[0] = np.copy(self.cube[1][ring,::-1])
            strip[1] = np.copy(self.cube[2][ring,::-1])
            strip[2] = np.copy(self.cube[0][ring,::-1])
            strip[3] = np.copy(self.cube[3][ring,::-1])

            add = 1 if direction == 0 else -1
            self.cube[1][ring,::-1] = strip[(0+add)%4]
            self.cube[2][ring,::-1] = strip[(1+add)%4]
            self.cube[0][ring,::-1] = strip[(2+add)%4]
            self.cube[3][ring,::-1] = strip[(3+add)%4]

        if face == 5:
            strip[0] = np.copy(self.cube[0][2-ring,:])
            strip[1] = np.copy(self.cube[2][2-ring,:])
            strip[2] = np.copy(self.cube[1][2-ring,:])
            strip[3] = np.copy(self.cube[3][2-ring,:])

            add = 1 if direction == 0 else -1
            self.cube[0][2-ring,:] = strip[(0+add)%4]
            self.cube[2][2-ring,:] = strip[(1+add)%4]
            self.cube[1][2-ring,:] = strip[(2+add)%4]
            self.cube[3][2-ring,:] = strip[(3+add)%4]

    def rotate(self, axis = -1, direction = 0):
        #placing the cube in a different orientation
        # 3 axis of rotations front, top, left
        # 2 directions clockwise and anticlockwise
        if axis == -1: #randomly choose an axis and direction
            axis = np.random.choice(3)
            direction = np.random.choice(2)
        face = axis * 2
        self.move_face(face, direction)
        self.move_face(face + 1, 1 - direction)
        for ring in range(3):
            self.move_ring(face, ring, direction)

    def move(self, face, direction):
        #rotating a given face
        self.move_face(face, direction)
        self.move_ring(face, 0, direction)

    def scramble(self, steps):
        #scrambling cube
        for step in range(steps):
            r = np.random.choice(10) #randomly rotating the cube 10 times
            for i in range(r):
                self.rotate()
            face = np.random.choice(6)
            direction = np.random.choice(2)
            self.move(face, direction)

    def is_solved(self):
        #If the cube is solved
        for face in range(6):
            if not np.all(self.cube[face] == self.cube[face][0][0]):
                return False
        return True    
        
    def reward(self):
        #Rewards
        if self.is_solved():
            return 100.0
        return -1.0

    def print_cube(self):
        print(self.cube)

    def get_state(self):
        #One hot encoding and flattening to get the state
        value = self.cube.flatten()
        b = np.zeros((value.size, max(value)+1))
        b[np.arange(value.size), value] = 1
        return b.flatten()

    def reset(self):
        for i in range(6):
            self.cube[i].fill(i)            


