import numpy as np

def getCurvilinearCoordinates(state, pt_ref):
    
    x, y, psi, v, delta = state[0], state[1], state[2], state[3], state[4]
    x_ref, y_ref, psi_ref, distance_along = pt_ref[0], pt_ref[1], pt_ref[2], pt_ref[3]

    cross_track_error = -(x-x_ref)*np.sin(psi_ref) + (y-y_ref)*np.cos(psi_ref)
    lateral_error = (x-x_ref)*np.cos(psi_ref) + (y-y_ref)*np.sin(psi_ref)
    heading_error = psi-psi_ref

    centres = [[5+4.9*np.cos(3*np.pi/4), 0+4.9*np.sin(3*np.pi/4)]]
    radii = [0.2]

    min_dist = np.inf

    for (centre, radius) in zip(centres, radii):
        # import pdb; pdb.set_trace()
        if (np.linalg.norm(centre-state[0:2]) <= min_dist):
            min_dist = np.linalg.norm(centre-state[0:2])

    # np.exp(-min_dist*0.5*min_dist)

    return np.array([v, cross_track_error, 0])

def getHMatrix(state, pt_ref):

    x, y, psi, v, delta = state[0], state[1], state[2], state[3], state[4]
    xc , yc = 5, 0

    R = 5
    # np.array([[-np.sin(pt_ref[2]), np.cos(pt_ref[2]), 0, 0, 0], [0, 0, 1, 0, 0], [0, 0, 0, 1, 0], [0, 0, 0, 0, 0]])
    
    return np.array([[0, 0, 0, 1, 0], [((x - xc)*((x - xc)**2 + (y - yc)**2)**(7/2)*(R*(y - yc)**2 + (-R + np.sqrt((x - xc)**2 + (y - yc)**2))*(x - xc)**2 + (y - yc)*(R*(y - yc) + (-y + yc)*np.sqrt((x - xc)**2 + (y - yc)**2))) - (x - xc)*((x - xc)**2 + (y - yc)**2)**3*(R*(x - xc)**2*np.sqrt((x - xc)**2 + (y - yc)**2) - R*((x - xc)**2 + (y - yc)**2)**(3/2) + ((x - xc)**2 + (y - yc)**2)**2) + (R*(x - xc) + (-x + xc)*np.sqrt((x - xc)**2 + (y - yc)**2))*((x - xc)**2 + (y - yc)**2)**(9/2))/((x - xc)**2 + (y - yc)**2)**(11/2), (y - yc)*((-R + np.sqrt((x - xc)**2 + (y - yc)**2))*((x - xc)**2 + (y - yc)**2)**(9/2) + ((x - xc)**2 + (y - yc)**2)**(7/2)*(-R*(x - xc)**2 - (x - xc)*(R*(x - xc) + (-x + xc)*np.sqrt((x - xc)**2 + (y - yc)**2)) + (y - yc)*(R*(y - yc) + (-y + yc)*np.sqrt((x - xc)**2 + (y - yc)**2))) + ((x - xc)**2 + (y - yc)**2)**3*(R*(y - yc)**2*np.sqrt((x - xc)**2 + (y - yc)**2) - R*((x - xc)**2 + (y - yc)**2)**(3/2) + ((x - xc)**2 + (y - yc)**2)**2))/((x - xc)**2 + (y - yc)**2)**(11/2), 0, 0, 0], [0, 0, 0, 0, 0]])