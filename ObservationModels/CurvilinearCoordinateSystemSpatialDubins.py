import numpy as np

def getCurvilinearCoordinates(state, pt_ref):
    
    x, y, psi = state[0], state[1], state[2]
    x_ref, y_ref, psi_ref, distance_along = pt_ref[0], pt_ref[1], pt_ref[2], pt_ref[3]

    cross_track_error = -(x-x_ref)*np.sin(psi_ref) + (y-y_ref)*np.cos(psi_ref)
    heading_error = psi-psi_ref

    return np.array([cross_track_error, heading_error])

def getHMatrix(state, pt_ref):

    x, y, psi = state[0], state[1], state[2]
    xc , yc = 5, 0

    R = 5
    
    return np.array([[-np.sin(pt_ref[2]), np.cos(pt_ref[2]), 0], [0, 0, 1]])