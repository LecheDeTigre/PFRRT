from ObservationModels.ObservationModel import ObservationModel
import numpy as np

class CurvilinearCoordinateSystemDubins(ObservationModel):
    def getObservation(self, state, ref):
        x, y, psi, v, delta = state[0], state[1], state[2], state[3], state[4]
        x_ref, y_ref, psi_ref, distance_along = ref[0], ref[1], ref[2], ref[3]

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

        return np.array([v, cross_track_error, distance_along])

    def getHMatrix(self, state, ref):

        x, y, psi, v, delta = state[0], state[1], state[2], state[3], state[4]
        xc , yc = 5, 0

        R = 5

        s_f = 5*np.pi/2
        # np.array([[-np.sin(ref[2]), np.cos(ref[2]), 0, 0, 0], [0, 0, 1, 0, 0], [0, 0, 0, 1, 0], [0, 0, 0, 0, 0]])

        return np.array([[0, 0, 0, 1, 0], [-np.sin(ref[2]), np.cos(ref[2]), 0, 0, 0], [0, 0, 0, 0, 0]])