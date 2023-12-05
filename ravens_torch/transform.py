# import numpy as np 
# from scipy.spatial.transform import Rotation as R
# import argparse
# # measured by moving the robot to the origin and then moving the robot to the x and y offsets
# origin = np.array([ 0.27142034, -0.26961078,  0.0036798 ])
# x_offset = np.array([ 0.4320755,  -0.27482935,  0.00345093])
# y_offset = np.array([ 0.27731327, -0.07600204,  0.00431739])


# transform_rob2world = None 
# transform_world2rob = None

# def world2rob(p):
#     p_h = np.ones((4,1))
#     p_h[:3,0] = p
#     return transform_world2rob@p_h

# def rob2world(p):
#     p_h = np.ones((4,1))
#     p_h[:3,0] = p

#     return transform_rob2world@p_h

# def compute_all():
#     global transform_rob2world, transform_world2rob
#     # get normalized vectors for x and y offsets, compute z offset as cross product
#     x_offset_normalized = (x_offset - origin)/np.linalg.norm(x_offset - origin)
#     y_offset_normalized = (y_offset - origin)/np.linalg.norm(y_offset - origin)
#     z_offset_normalized = np.cross(x_offset_normalized,y_offset_normalized)

#     # compute rotation matrix and translation vector
#     R_world2rob = np.array([x_offset_normalized,y_offset_normalized,z_offset_normalized]).T
#     t_world2rob = origin

#     # compute world2rob transform
#     transform_world2rob = np.eye(4)
#     transform_world2rob[:3,:3] = R_world2rob
#     transform_world2rob[:3,3] = t_world2rob

#     # compute rob2world transform
#     transform_rob2world = np.eye(4)
#     transform_rob2world[:3,:3] = R_world2rob.T
#     transform_rob2world[:3,3] = -R_world2rob.T@t_world2rob



# if __name__=='__main__':
#     np.set_printoptions(suppress=True)
#     compute_all()
#     print('origin in world frame is {}, {}, {} in robot frame'.format(world2rob(origin)[0][0],world2rob(origin)[1][0],world2rob(origin)[2][0]))
#     print('origin in robot frame is {}, {}, {} in world frame'.format(rob2world(np.array([0,0,0]))[0][0],rob2world(np.array([0,0,0]))[1][0],rob2world(np.array([0,0,0]))[2][0]))

#     print('x_offset in world frame is {}, {}, {} in world frame'.format(rob2world(x_offset)[0][0],rob2world(x_offset)[1][0],rob2world(x_offset)[2][0]))
#     print('y_offset in robot frame is {}, {}, {} in world frame'.format(rob2world(y_offset)[0][0],rob2world(y_offset)[1][0],rob2world(y_offset)[2][0]))

#     parser = argparse.ArgumentParser()
#     parser.add_argument("-x",default=0.0)
#     parser.add_argument("-y",default=0.0)
#     parser.add_argument("-z",default=0.0)
#     args = parser.parse_args()
    
#     query_point = np.array([args.x,args.y,args.z])
#     rob_frame = world2rob(query_point)
#     print(rob_frame)


import numpy as np 
from scipy.spatial.transform import Rotation as R
import argparse
# measured by moving the robot to the origin and then moving the robot to the x and y offsets
origin = np.array([ 0.27142034, -0.26961078,  0.0036798 ])
x_offset = np.array([ 0.4320755,  -0.27482935,  0.00345093])
y_offset = np.array([ 0.27731327, -0.07600204,  0.00431739])


transform_rob2world = None 
transform_world2rob = None

def compute_all():
    global transform_rob2world, transform_world2rob
    # get normalized vectors for x and y offsets, compute z offset as cross product
    x_offset_normalized = (x_offset - origin)/np.linalg.norm(x_offset - origin)
    y_offset_normalized = (y_offset - origin)/np.linalg.norm(y_offset - origin)
    z_offset_normalized = np.cross(x_offset_normalized,y_offset_normalized)

    # compute rotation matrix and translation vector
    R_world2rob = np.array([x_offset_normalized,y_offset_normalized,z_offset_normalized]).T
    t_world2rob = origin

    # compute world2rob transform
    transform_world2rob = np.eye(4)
    transform_world2rob[:3,:3] = R_world2rob
    transform_world2rob[:3,3] = t_world2rob

    # compute rob2world transform
    transform_rob2world = np.eye(4)
    transform_rob2world[:3,:3] = R_world2rob.T
    transform_rob2world[:3,3] = -R_world2rob.T@t_world2rob

def world2rob(p):
    if p.shape[0] == 4:
        p = p[:3]
    

    if transform_world2rob is None:
        compute_all()
        
    p_h = np.ones((4,1))
    p_h[:3,0] = p
    return (transform_world2rob@p_h)[:3].reshape((3,))

def rob2world(p):
    if p.shape[0] == 4:
        p = p[:3]
    
    if p.ndim >1:
        p = p.reshape((-1))

   
    if transform_rob2world is None:
        compute_all()
    p_h = np.ones((4,1))
    p_h[:3,0] = p

    return (transform_rob2world@p_h)[:3].reshape((3,))




if __name__=='__main__':
    np.set_printoptions(suppress=True)
    compute_all()
    print('origin in world frame is {}, {}, {} in robot frame'.format(world2rob(origin)[0][0],world2rob(origin)[1][0],world2rob(origin)[2][0]))
    print('origin in robot frame is {}, {}, {} in world frame'.format(rob2world(np.array([0,0,0]))[0][0],rob2world(np.array([0,0,0]))[1][0],rob2world(np.array([0,0,0]))[2][0]))

    print('x_offset in world frame is {}, {}, {} in world frame'.format(rob2world(x_offset)[0][0],rob2world(x_offset)[1][0],rob2world(x_offset)[2][0]))
    print('y_offset in robot frame is {}, {}, {} in world frame'.format(rob2world(y_offset)[0][0],rob2world(y_offset)[1][0],rob2world(y_offset)[2][0]))

    parser = argparse.ArgumentParser()
    parser.add_argument("-x",default=0.0)
    parser.add_argument("-y",default=0.0)
    parser.add_argument("-z",default=0.0)
    args = parser.parse_args()
    
    query_point = np.array([args.x,args.y,args.z])
    rob_frame = world2rob(query_point)
    print(rob_frame)
