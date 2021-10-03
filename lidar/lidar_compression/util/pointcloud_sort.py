import numpy as np
import numpy.ma as ma
from math import pi, floor

def add_spherical_coordinates(cartesian_cloud:np.ndarray)->np.ndarray:
    """Adds the spherical coordinates r, theta, phi to each point in the point cloud.

    Parameters
    ----------
    cartesian_cloud : np.ndarray
        The point cloud in cartesian coordinates with shape (num_points, 3), each point ordered [x,y,z]

    Returns
    -------
    np.ndarray
        Array of shape (num_points, 6), each point having [x,y,z,r,theta,phi]
    """
    if cartesian_cloud is None or cartesian_cloud.size == 0:    
        raise ValueError("Point cloud must not be empty")
    x = cartesian_cloud[:,0]
    y = cartesian_cloud[:,1]
    z = cartesian_cloud[:,2]
    r = np.sqrt(x**2 + y**2 + z**2).reshape((cartesian_cloud.shape[0], 1))
    # Same as the "theta = arctan2(...)" formula
    # We need the where-condition in divide to catch cases where r=0 -> arccos(x) should be 0 there -> 0 = arccos(1)
    theta = np.arccos(np.divide(z, r[:,0], out=np.ones_like(z, dtype=np.float), where=r[:,0]!=0)).reshape((cartesian_cloud.shape[0], 1))
    phi = np.arctan2(y,x).reshape((cartesian_cloud.shape[0], 1))
    # Scale phi from [-pi,pi] to [0,2pi], which is needed for sorting it later
    phi[phi<0] = phi[phi<0] + 2*pi
    return np.concatenate((cartesian_cloud, r, theta, phi), axis=1)

def remove_distant_and_close_points(cloud:np.ndarray, min_percentile:int=1, max_percentile:int=99)->np.ndarray:
    """Removes points that are outside the interval determined by [min_percentile, max_percentile]. The percentiles are computed and applied to each axis (x,y,z) individually.

    Parameters
    ----------
    cloud : np.ndarray
        Point Cloud with shape(num_points, n), where n>=3 and shape[2][:3] consists of (x,y,z).
    min_percentile : int, optional
        Percentile that will determine the lower bound of values on each axis, by default 1
    max_percentile : int, optional
        Percentile that will determine the upper bound of values on each axis, by default 99

    Returns
    -------
    np.ndarray
        A reduced array.
    """
    cloud = cloud.reshape(-1,3)

    min_x = np.percentile(cloud[:,0], min_percentile, interpolation='nearest')
    max_x = np.percentile(cloud[:,0], max_percentile, interpolation='nearest')
    min_y = np.percentile(cloud[:,1], min_percentile, interpolation='nearest')
    max_y = np.percentile(cloud[:,1], max_percentile, interpolation='nearest')
    min_z = np.percentile(cloud[:,2], min_percentile, interpolation='nearest')
    max_z = np.percentile(cloud[:,2], max_percentile, interpolation='nearest')

    # condition_x = np.any([cloud[:,0] < min_x, cloud[:,0] > max_x], axis=0).reshape((cloud.shape[0], 1))
    # condition_y = np.any([cloud[:,1] < min_y, cloud[:,1] > max_y], axis=0).reshape((cloud.shape[0], 1))
    # condition_z = np.any([cloud[:,2] < min_z, cloud[:,2] > max_z], axis=0).reshape((cloud.shape[0], 1))
    # condition = np.concatenate([condition_x, condition_y, condition_z], axis=1)
    # TODO Try this
    condition_xyz = np.any([cloud[:,0] < min_x,
                           cloud[:,0] > max_x,
                           cloud[:,1] < min_y,
                           cloud[:,1] > max_y,
                           cloud[:,2] < min_z,
                           cloud[:,2] > max_z], axis=0).reshape((cloud.shape[0], 1))
    condition = np.concatenate([condition_xyz, condition_xyz, condition_xyz], axis=1)


    masked_cloud = ma.masked_where(condition, cloud, copy=False)
    # ~masked_cloud.mask removes the values where the mask is True.
    # With True meaning that the condition applied, this means that we want to remove those values. 
    return masked_cloud[~masked_cloud.mask].data.reshape((-1,3))

def sort_points(cloud:np.ndarray, num_cols:int=512, num_rows:int=64)->np.ndarray:
    """Sorts the cloud by theta(rows) and phi (columns) by binning all points into num_rows*num_cols cells and calculating the mean x,y,z for each cell.

    Parameters
    ----------
    cloud : np.ndarray
        Point cloud of shape=(num_points, 6), where shape[2]=(x,y,z,r,theta,phi) and theta, phi are in radians
    num_cols : int, optional
        The number of columns that the resulting array should have, by default 512
    num_rows : int, optional
        The number of rows that the resulting array should have, by default 60

    Returns
    -------
    np.ndarray
        Sorted and binned array of shape=(num_rows, num_cols, 3), where shape[2] is ordered x,y,z
    """    
    max_theta = cloud[:,4].max()
    min_theta = cloud[:,4].min()
    max_phi = cloud[:,5].max()
    min_phi = cloud[:,5].min()    
    row_step = (max_theta - min_theta) / num_rows
    col_step = (max_phi - min_phi) / num_cols
    
    sorted_list = [[ [] for _ in range(num_cols)] for _ in range(num_rows)]
    for point in cloud:
        # point[4] is theta
        # We must subtract theta_min before division, because the interval does not start at 0, but at theta_min.
        # Clamp the index to min(index,num_rows-1) to avoid idx getting out of 
        # bounds when (point[4] - min_theta) / row_step) > num_rows - 0,5 (because we round up there)
        idx_row = min(floor((point[4] - min_theta) / row_step), num_rows-1)
        idx_col = min(floor((point[5] - min_phi) / col_step), num_cols-1)
        sorted_list[idx_row][idx_col].append(point)

    sorted = np.zeros((num_rows, num_cols, 3), dtype=float)
    row_idx = 0
    for row in sorted_list:
        col_idx = 0        
        for col in row:            
            if len(col) > 0:
                arr = np.asfarray(col)[:, :3]
                sorted[row_idx][col_idx][0] = arr[:,0].mean() # x
                sorted[row_idx][col_idx][1] = arr[:,1].mean() # y
                sorted[row_idx][col_idx][2] = arr[:,2].mean() # z            
            col_idx += 1
        row_idx += 1
    return sorted
