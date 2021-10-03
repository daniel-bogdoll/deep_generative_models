import numpy as np
from math import floor

def sort_points_theta_phi_naive(cloud:np.ndarray, num_cols:int=512, num_rows:int=64)->np.ndarray:
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
    
    rows = [[] for _ in range(num_rows)]    
    for point in cloud:
        # point[4] is theta
        # We must subtract theta_min before division, because the interval does not start at 0, but at theta_min.
        # Clamp the index to min(index,num_rows-1) to avoid idx getting out of 
        # bounds when (point[4] - min_theta) / row_step) > num_rows - 0,5 (because we round up there)
        idx = min(floor((point[4] - min_theta) / row_step), num_rows-1)
        rows[idx].append(point)

    # rows is now sorted by theta and contains num_rows lists - one for each row.
    # We must sort and bin each of those by phi.
    sorted = np.ndarray((num_rows, num_cols, 3), dtype=float)
    sorted_list = [[ [] for _ in range(num_cols)] for _ in range(num_rows)]
    row_idx = 0    
    idx = 0
    for row in rows:
        for point in row:
            idx = min(floor((point[5] - min_phi) / col_step), num_cols-1)
            sorted_list[row_idx][idx].append(point)            
        row_idx += 1

    row_idx = 0
    for row in sorted_list:
        col_idx = 0        
        for col in row:
            value = np.zeros(3, dtype=np.float)
            if len(col) > 0:
                arr = np.asfarray(col)[:, :3]
                value[0] = arr[:,0].mean() # x
                value[1] = arr[:,1].mean() # y
                value[2] = arr[:,2].mean() # z         
            sorted[row_idx][col_idx] = value
            col_idx += 1
        row_idx += 1
    return sorted

def sort_points_theta_phi_optimized_intermediate(cloud:np.ndarray, num_cols:int=512, num_rows:int=64)->np.ndarray:
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
    
    # sorted_list[4] = number of elements that were summed in this cell
    # Used for averaging later
    sorted_list = np.zeros((num_rows, num_cols, 4), dtype=float)    
    for point in cloud:
        # point[4] is theta
        # We must subtract theta_min before division, because the interval does not start at 0, but at theta_min.
        # Clamp the index to min(index,num_rows-1) to avoid idx getting out of 
        # bounds when (point[4] - min_theta) / row_step) > num_rows - 0,5 (because we round up there)
        idx_row = min(floor((point[4] - min_theta) / row_step), num_rows-1)
        idx_col = min(floor((point[5] - min_phi) / col_step), num_cols-1)
        sorted_list[idx_row][idx_col] += np.concatenate((point[:3], [1]))

    sorted = np.zeros((num_rows, num_cols, 3), dtype=float)
    row_idx = 0
    for row in sorted_list:
        col_idx = 0
        for col in row:      
            sorted[row_idx][col_idx][0] = np.divide(col[0], col[3], out=np.zeros(1), where=col[3]!=0) # x
            sorted[row_idx][col_idx][1] = np.divide(col[1], col[3], out=np.zeros(1), where=col[3]!=0) # y
            sorted[row_idx][col_idx][2] = np.divide(col[2], col[3], out=np.zeros(1), where=col[3]!=0) # z
            col_idx += 1
        row_idx += 1    
    return sorted

def sort_points_theta_phi_optimized_numpy_only(cloud:np.ndarray, num_cols:int=512, num_rows:int=64)->np.ndarray:
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
    
    # sorted_list[4] = number of elements that were summed in this cell
    # Used for averaging later
    sorted_list = np.zeros((num_rows, num_cols, 6), dtype=float)
    # divisors = np.zeros((num_rows, num_cols, 3), dtype=np.ushort)
    for point in cloud:
        # point[4] is theta
        # We must subtract theta_min before division, because the interval does not start at 0, but at theta_min.
        # Clamp the index to min(index,num_rows-1) to avoid idx getting out of 
        # bounds when (point[4] - min_theta) / row_step) > num_rows - 0,5 (because we round up there)
        idx_row = min(floor((point[4] - min_theta) / row_step), num_rows-1)
        idx_col = min(floor((point[5] - min_phi) / col_step), num_cols-1)
        sorted_list[idx_row][idx_col] += np.concatenate((point[:3], np.ones(3)))
        # divisors[idx_row][idx_col] += np.ones(3, dtype=np.ushort)

    # sorted_list = np.divide(sorted_list, divisors, out=np.zeros_like(sorted_list), dtype=float, where=divisors!=0)
    # sorted = np.zeros((num_rows, num_cols, 3), dtype=float)
    # row_idx = 0
    # for row in sorted_list:
    #     col_idx = 0
    #     for col in row:      
    #         sorted[row_idx][col_idx] = np.divide(col[:3], col[3], out=np.zeros(3), where=col[3]!=0) # x
    #         # sorted[row_idx][col_idx][1] = np.divide(col[1], col[3], out=np.zeros(1), where=col[3]!=0) # y
    #         # sorted[row_idx][col_idx][2] = np.divide(col[2], col[3], out=np.zeros(1), where=col[3]!=0) # z
    #         col_idx += 1
    #     row_idx += 1    
    return np.divide(sorted_list[:,:,:3], sorted_list[:,:,3:6], out=np.zeros_like(sorted_list[:,:,:3]), dtype=float, where=sorted_list[:,:,3:6]!=0)