import cv2

def _cv_key_points_to_list(cv_key_points):
    converted = []
    for cv_kp in cv_key_points:
        kp = [cv_kp.pt[0], cv_kp.pt[1],
              cv_kp.size,
              cv_kp.angle,
              cv_kp.response,
              float(cv_kp.octave),
              float(cv_kp.class_id)]
        converted.append(kp)
    return converted

def _list_to_cv_key_points(key_points):
    converted = []
    for kp in key_points:
        cv_kp = cv2.KeyPoint(kp[0], kp[1], # pt
                             kp[2], # size
                             kp[3], # angle
                             kp[4], # response
                             int(kp[5]),  # octave
                             int(kp[6]))  # class_id
        converted.append(cv_kp)
    return converted

