"""Util functions for converting the KITTI lidar scans.

Based on Lucas Caccia's code (https://github.com/pclucas14/lidar_generation)
"""
import numpy as np
from pathlib import Path
from lidar_compression.util.lidarexceptions import PointException, InvalidInputException, LineNumberException

def get_quadrant(point):
    if point[0] >= 0. and point[1] >= 0. :
        return 0
    elif point[0] <= 0. and point[1] >= 0. : 
        return 1
    elif point[0] <= 0. and point[1] <= 0. : 
        return 2
    elif point[0] >= 0. and point[1] <= 0. : 
        return 3
    else :
        raise InvalidInputException({'message':'Invalid input: ' + str(point),
                                    'point': str(point),
                                    'shape': str(point.shape)})

def passed_origin(x_t, x_t1):
    if get_quadrant(x_t1) == 3 and get_quadrant(x_t) == 0: 
        return True
    else : 
        return False

def fit_quadrant(points, quadrant, desired_amt):
    points = np.asarray(points)
    slots = []
    slot_size = np.pi / (2 * desired_amt)
    for i in range(int(desired_amt)):
        slots.append([])
    try:
        if quadrant == 0: 
            points = points[::-1]
        elif quadrant == 1 : 
            #import pdb; pdb.set_trace()
            points[:, 0] = - points[:, 0]
        elif quadrant == 2 :
            points = points[::-1] 
            points[:, 0] = - points[:, 0]
            points[:, 1] = - points[:, 1]
        elif quadrant == 3 : 
            points[:, 1] = - points[:, 1]
    except IndexError:                
        raise PointException({'message':'Invalid points shape: ' + str(points.shape),
                                'points':str(points),
                                'quadrant':str(quadrant),
                                'shape':str(points.shape)})

    # import pdb; pdb.set_trace()
    for point in points : 
        angle = np.arctan(point[1] / point[0])
        index = min(int(angle / slot_size), desired_amt - 1)
        slots[index].append(point)

    for i in range(len(slots)):
        if len(slots[i]) == 0 : 
            slots[i] = np.array([0., 0., 0., 0.])
        else :
            full_slot = np.asarray(slots[i])
            slots[i] = full_slot.mean(axis=0)

    points = np.asarray(slots)
    if quadrant == 0: 
        points = points[::-1]
    elif quadrant == 1 : 
        points[:, 0] = - points[:, 0]
    elif quadrant == 2 : 
        points = points[::-1]
        points[:, 0] = - points[:, 0]
        points[:, 1] = - points[:, 1]
    elif quadrant == 3 : 
        points[:, 1] = - points[:, 1]

    return points

def parse_velo(velo):
    """ Creates an array of "lines" for the 2d representation.
    Each line represents one lidar rotation.

    Returns an H x 4 (Quadrants) x ? array, split into quadrants"""

    # Points closer to the origin (0,0,0) are at the end of the point cloud.
    # So invert the point cloud such that we begin near the origin. 
    velo = velo[::-1]
    lines = []
    current_point = velo[0]
    current_quadrant = get_quadrant(current_point)
    current_line = [[], [], [], []]    
    for point in velo:
        point_quadrant = get_quadrant(point)
        
        if passed_origin(current_point, point):
            lines.append(current_line)
            current_line = [[], [], [], []]

        current_line[point_quadrant].append(point)
        current_quadrant = point_quadrant
        current_point = point     
    # print("Number of lines: " + str(len(lines)))
    # print("Number of elements per line: " + str(len(lines[0][0])))
    # print("Number of channels: " + str(len(lines[0])))
    # print("Elements at line[0][0][0]: " + str(lines[0][0][0]))    
    return lines

def process_velo_kitti(velo, points_per_layer, stop=False):
    lines = parse_velo(velo)
    inverse = quad_to_pc_inv(lines)    
    lines = lines[2:-1]
    if len(lines) != 60 :
        raise LineNumberException({'message':'Invalid number of lines.',
                                    'lines': str(len(lines))})
    out_tensor = np.zeros((60, points_per_layer, 4))
    if stop:
        import pdb; pdb.set_trace()
        x = 1
    for j in range(len(lines)):
        line = lines[j]
        out_line = np.zeros((points_per_layer, 4))
        for i in range(len(line)):            
            gridded = fit_quadrant(line[i], i, int(points_per_layer / 4))
            out_tensor[j][i*int(points_per_layer/4):(i+1)*int(points_per_layer/4), :] = gridded[::-1]
    return out_tensor, inverse

def process_velo_waymo(velo, points_per_layer, stop=False):
    """Transform a waymo point cloud into a 2d representation with 4 Channels. 
    The resulting Tensor has shape (height, width, 4).

    Does not work currently
    """
    lines = parse_velo(velo)
    inverse = quad_to_pc_inv(lines)      
    out_tensor = np.zeros((len(lines), points_per_layer, 4))
    if stop:
        import pdb; pdb.set_trace()
        x = 1
    for j in range(len(lines)):
        line = lines[j]
        out_line = np.zeros((points_per_layer, 4))
        for i in range(len(line)):
            gridded = fit_quadrant(line[i], i, int(points_per_layer / 4))
            out_tensor[j][i*int(points_per_layer/4):(i+1)*int(points_per_layer/4), :] = gridded[::-1]
    return out_tensor, inverse

def waymo_to_polar(pc_cart):
    for point in pc_cart:
        pass

def quad_to_pc_inv(lines, th=3.):
    # lines is a 63 x 4 array, where each slot has an array of 4d/3d points
    # goal : get an array of points that fills empty spaces
    points = []
    for i in range(len(lines)) :
        line = lines[i] 
        distance = []
        for quad in line : 
            for point in quad : 
                x, y, z = point[:3]
                distance.append(x**2 + y**2)
        distance = np.array(distance)
        std = distance.std()
        sorted_indices = np.argsort(distance)
        median_index = sorted_indices[int(sorted_indices.shape[0]*0.95)]
        median = distance[median_index]

        for quad in line : 
            for point in quad : 
                x, y, z = point[:3]
                dist = x ** 2 + y ** 2 
                if dist < median and (median/dist-1.) > th:#*std : 
                    # blocked point --> scale to get real pt
                    scale = np.sqrt(median / dist)
                    scaled = scale * point
                    points.append(scaled)
    return np.array(points)

def read_raw_velo(path:Path):
    cloud = np.fromfile(path, dtype=np.float32).reshape((-1, 4))
    return cloud