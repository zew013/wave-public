import numpy as np
from scipy.interpolate import griddata
from scipy.linalg import lstsq
from scipy.ndimage import binary_erosion
from sklearn.cluster import DBSCAN
from datetime import datetime
def parallel_f(tup, pixel_ratio, dt):
    def edge_detect(mat):
            result = [[True if mat[row, column] < max([
                mat[row-1, column],
                mat[row, column-1], mat[row, column+1],
                mat[row+1, column]]  
            ) else False for column in range(1, mat[row].shape[0] - 1)] for row in range(1, mat.shape[0] - 1)]

            return np.pad(result, 1, 'edge')
    
    def vector_field(mat, n=7):
        # X = np.array([[0, 0], [0, 1], [0,2], [1,0], [1, 1], [1,2], [2, 0], [2, 1], [2, 2]])
        # X = np.hstack((X, np.ones((len(X), 1))))
        # mat = np.pad(mat, 1, mode='edge')
        # result = [[lstsq(X, np.array([
        #     mat[row+1, column-1], mat[row, column-1], mat[row-1, column-1],
        #     mat[row+1, column], mat[row, column], mat[row-1, column],
        #     mat[row+1, column+1], mat[row, column+1], mat[row-1, column+1]]  
        # ))[0][:2]
        #     if not (np.isnan([
        #     mat[row+1, column-1], mat[row, column-1], mat[row-1, column-1],
        #     mat[row+1, column], mat[row, column], mat[row-1, column],
        #     mat[row+1, column+1], mat[row, column+1], mat[row-1, column+1]]  
        # ).any()) else np.array([np.nan, np.nan]) for column in range(1, mat[row].shape[0] - 1)] for row in range(1, mat.shape[0] - 1)]
        # return np.array(result)

        t_field = np.empty((*(mat.shape), 2))
        # Create the array using list comprehension
        X = np.array([[j, i] for i in reversed(range(n)) for j in (range(n))])
        X = np.hstack((X, np.ones((len(X), 1))))
        pad_width = (n-1)//2
        mat = np.pad(mat, pad_width, mode='edge')

        for i in range(mat.shape[0] - n + 1):
            for j in range(mat.shape[1] - n + 1):
                kernel_lstsq = mat[i:i+n, j:j+n].flatten()
                if np.isnan(kernel_lstsq).any():
                    t_field[i, j] = np.array([np.nan, np.nan])
                else:
                    t_field[i, j] = np.array(lstsq(X, kernel_lstsq)[0][:2])
        return t_field
    
    def create_matrix_from_indices(indices, shape):
        """ Create a matrix from indices. """
        matrix = np.zeros(shape, dtype=int)
        matrix[indices[:, 1], indices[:, 0]] = 1

        return matrix

    def apply_erosion(matrix, kernel_size=7):
        """ Apply binary erosion to the matrix. """
        structuring_element = np.ones((kernel_size, kernel_size), dtype=int)
        matrix = binary_erosion(matrix, structure=structuring_element)
        return matrix# np.argwhere(matrix==1)

    activation_map, activation_map_index_dict = tup
    # for each activation map, if available pixels are less than one-third of the entire pixels, then skip
    if (np.isnan(activation_map).sum()) > (activation_map.shape[0] * activation_map.shape[1] / 2):
        # continue
        return
    # extract the edge of the activation map
    edge = edge_detect(activation_map)
    
    frame = np.nanmin(activation_map)
    end_frame = np.nanmax(activation_map)

    # interpolate the activation map
    iso_idx = np.nonzero(edge)
    isocontour_points = np.array(list(zip(*iso_idx)))
    isocontour_values = activation_map[iso_idx]
    grid_x, grid_y = np.mgrid[0:edge.shape[0], 0:edge.shape[1]]
    grid_z = griddata(isocontour_points, isocontour_values, (grid_x, grid_y), method='linear')
    
    # calculate the vector field using least square
    v_field = vector_field(grid_z)
    t = (v_field**2).sum(axis = 2)
    
    notnanzero_position = ((~np.isnan(t)) & (~(t <= 0)))
    # v = (np.sqrt(t)[notnanzero_position] / t[notnanzero_position]) * self.pixel_ratio / self.dt
    
    v = np.full(t.shape, np.nan)
    v = np.divide(np.sqrt(t), t, out=v, where=notnanzero_position) * pixel_ratio / dt
    # vfield = v 

    activation_map_none = np.isnan(activation_map)
    v = np.where(activation_map_none, np.nan, v)
    # wave_t_to_filter = set(activation_map[(v > 50) & (~activation_map_none)])
    all_index_to_filter = []
    for each_t in activation_map_index_dict:
        
        idx_t = activation_map_index_dict[each_t]
        clustering_label = DBSCAN(eps=32, min_samples=32, n_jobs=1).fit_predict(idx_t)
        clusters = np.unique(clustering_label)
        for i in clusters:
            idx_t_per_cluster = idx_t[clustering_label == i]
            # Determine the size of the matrix
            matrix_size = v.shape
            # Create the matrix from indices
            matrix = create_matrix_from_indices(idx_t_per_cluster, matrix_size)
            # Apply erosion 
            matrix = apply_erosion(matrix)
            if np.any(matrix * v > 40):
                all_index_to_filter.append(idx_t_per_cluster)
    if len(all_index_to_filter) > 0:
        all_index_to_filter = np.vstack(all_index_to_filter)
        v[all_index_to_filter[:, 1], all_index_to_filter[:, 0]] = np.nan
    v = np.where(v > 30, np.nan, v)
    
    # can also use gradient
    # t = np.reciprocal(v_field, where=v_field!=0, dtype=float)
    # t = (t**2).sum(axis = 2)
    # notnanzero_position = ((~np.isnan(t)) & (~(t == 0)))
    # v = (np.sqrt(t)[notnanzero_position]).flatten() * self.pixel_ratio / self.dt

    # dtdx = sobel(grid_z,axis=1,mode='nearest').astype(float)
    # dtdy = sobel(grid_z,axis=0,mode='nearest').astype(float)
    # t = dtdx**2 + dtdy**2
    # notnanzero_position = ((~np.isnan(t)) & (~(t == 0)))
    # v = np.reciprocal((np.sqrt(t)), where=notnanzero_position).flatten() * self.pixel_ratio / self.dt

    # t = (np.array(np.gradient(grid_z))**2).sum(axis=0)
    # notnanzero_position = ((~np.isnan(t)) & (~(t == 0)))
    # v = np.reciprocal((np.sqrt(t)), where=notnanzero_position).flatten() * self.pixel_ratio / self.dt

    # save the 25, 50, 75 percentile of the v in each activation map
    vs = v
    v = v.flatten()
    median = np.nanmedian(v)
    if np.isnan(median):
        median = 0
    v = np.where(0.5 * median < v, v, np.nan) 
    mean = np.nanmean(v)

    return frame, end_frame, median, mean, vs# , vfield
