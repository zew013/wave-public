import numpy as np
import tifffile
from scipy.interpolate import griddata
from scipy.linalg import lstsq
from tqdm.notebook import tqdm
import os
from sklearn.cluster import DBSCAN
from scipy.spatial.distance import cdist
from itertools import islice
import pickle
import matplotlib.pyplot as plt
import pandas as pd
# from scipy.ndimage import sobel
from multiprocessing import Pool, cpu_count
from functools import partial
from parallel_f import parallel_f

# LinkedList data structure node class
class Node:
    def __init__(self, data, frame, label):
        self.data = data
        self.frame = frame
        self.label = label
        self.next = None

    def __repr__(self):
        return f'({self.data}, {self.frame}, {self.label})'
# Linkedlist class
class LinkedList:
    def __init__(self, start_frame, head = None, tail = None):
        self.head = head
        self.tail = tail
        self.start_frame = start_frame
    def __repr__(self):
        node = self.head
        nodes = []
        while node is not None:
            nodes.append(repr(node))
            node = node.next
        nodes.append("None")
        return " -> ".join(nodes)
    def __iter__(self):
        node = self.head
        while node is not None:
            yield node
            node = node.next

class SpeedMeasure:
    # Class for measuring the speed
    def __init__(self, tiff_file='wave_mask.tif', image_range = None, min_pixel_required=500, max_continuous_frames=None, pixel_ratio=1906 / 512, dt = 15):
        self.path = tiff_file.split("/")
        self.path.pop(-1)
        self.path = '/'.join(self.path)
        # read in image
        if image_range:
            self.mov =  tifffile.imread(tiff_file, key = range(image_range[0], image_range[1]))
            self.start = image_range[0]
        else:
            self.mov =  tifffile.imread(tiff_file)
            self.start = 0
        
        # if the labeled pixels for a frame is less than min_pixel_required, then skip that frame
        self.idx = {}
        self.min_pixel_required = min_pixel_required
        wave = []
        for i in range(len(self.mov)):
            if np.sum(self.mov[i]==255) < self.min_pixel_required:
                self.idx[i - len(wave)] = wave
                wave = []
            else:
                if max_continuous_frames is not None:
                    if len(wave) > max_continuous_frames:
                        self.idx[i - len(wave)] = wave
                        wave = []
                    else:
                        wave.append(i)
                else:
                    wave.append(i)
                
            
                
        self.idx[len(self.mov) - len(wave)] = wave
        self.dim = self.mov[0].shape
        self.pixel_ratio = pixel_ratio
        self.dt = dt

    def gen_activation_map(self, epss=[(0,100)], cache='dbscan.pkl'):
        # create activation maps for the video

        def _gen_cluster(self):
            # cluster each frame using DBSCAN. Features used for clustering are x,y coordinates
            cluster_info = {}
            pbar = tqdm(self.idx.items(), total=len(self.idx))
            for start, frames in pbar:
                if len(frames) == 0:
                    continue
                cluster_info[start] = {}
                
                for frame in tqdm(frames, desc=''):
                    if len(epss) > 0:
                        if frame+self.start >= epss[0][0]:
                            eps = epss[0][1]
                            epss.pop(0)
                    y, x = np.where(self.mov[frame] != 0)
                    X = np.array(list(zip(x, y)))
                    
                    clustering = DBSCAN(eps=eps, min_samples=self.min_pixel_required, n_jobs=-1).fit(X)
                    cluster_info[start][frame] = clustering
                    pbar.set_postfix({'latest_frame': frame})
            return cluster_info
        
        def _gen_cord(self, cluster_info):
            # {402: {402: [array([375.79695431, 505.55160745])],
            # 403: [array([386.45316253, 499.1184948 ])],
            # 404: [array([399.39223839, 493.30561331])],...
            # get the center coordinate of each clustering
            cord_info = {}
            for start, frames in cluster_info.items():
                cord_info[start] = {}
                for frame, cluster in frames.items():
                    n_cluster = cluster.labels_.max() + 1
                    y, x = np.where(self.mov[frame] != 0)
                    X = np.array(list(zip(x, y)))
                    if len(X) <= 0:
                        continue
                    cord_info[start][frame] = []
                    for i in range(n_cluster):
                        cord_info[start][frame].append(X[cluster.labels_ == i].mean(axis = 0))
            return cord_info
        
    
        def _gen_match(self, cord_info):
            # match the clusters based on nearest distance from clusters in last frame

            def match(cords1, cords2):
                return cdist(cords1, cords2)
            
            match_info = {}    
            # loop through the center coordinates from _gen_cord
            for start, frames in cord_info.items():
                n = len(frames)
                match_info[start] = {}
                finished = []
                unfinished = []

                # initialize the linkedlist in unfinished
                # linkedlist is used to connect the same wave through different frames
                for i, cord in enumerate(frames[start]):
                    head = Node(cord, start, i)
                    unfinished.append(LinkedList(start, head = head, tail = head))

                # loop through the rest of frames starting at index 1
                for frame, cords in islice(frames.items(), 1, None):
                    # if the last frame has no cluster (coordinates), initialize new linkedlist in unfinished
                    if len(frames[frame-1]) == 0:
                        for i, cord in enumerate(frames[frame-1]):
                            head = Node(cord, frame-1, i)
                            unfinished.append(LinkedList(frame-1, head = head, tail = head))
                        continue
                    # if the current frame has no cluster, then finish the linkedlist 
                    if len(cords) == 0:
                        for ll in unfinished:
                            ll.tail.next = None
                            ll.tail = None
                            unfinished.remove(ll)
                            finished.append(ll)
                        continue

                    # connect waves from last frame and this frame by comparing their distances
                    distance = match(frames[frame-1], cords)
                    a = set(zip(range(max(distance.shape)), np.argmin(distance, axis = 1)))
                    b = set(zip(np.argmin(distance, axis = 0), range(max(distance.shape))))
                    update = a.intersection(b)
                    remove = a.difference(b)
                    add = b.difference(a)
                    # finish the linkedlist if the label of cluster of the tail is in remove
                    for j in remove:
                        for ll in unfinished:
                            if j[0] == ll.tail.label:
                                ll.tail.next = None
                                ll.tail = None
                                unfinished.remove(ll)
                                finished.append(ll)
                    buffer = []
                    # update the linkedlist if the label of cluster of the tail is in update
                    for i in update:
                        for ll in unfinished:
                            if i[0] == ll.tail.label:
                                tail = Node(cords[i[1]], frame, i[1])
                                buffer.append((ll, tail))
                    for ll, tail in buffer:
                        ll.tail.next = tail
                        ll.tail = tail
                    # add new linkedlist if there is a new cluster in add
                    for k in add:
                        head = Node(cords[k[1]], frame, k[1])
                        unfinished.append(LinkedList(frame, head = head, tail = head))
                
                for ll in unfinished:
                    ll.tail.next = None
                    ll.tail = None
                match_info[start] = finished + unfinished
            return match_info
        
        
        def _gen_map(self, match_info, cluster_info):
            # generate activate map based on the linked clusters
            map_info = []
            map_info_index = []
            for start, all_ll in match_info.items():
                for ll in all_ll:
                    m = np.array(self.dim[0] * [self.dim[1]*[np.nan]])
                    index_dict = {}
                    for node in ll:
                        model = cluster_info[start][node.frame]
                        indices = model.components_[model.labels_[model.core_sample_indices_] == node.label]
                        
                        m[indices[:,1], indices[:,0]] = node.frame
                        index_dict[node.frame] = indices
                    map_info.append(m)
                    map_info_index.append(index_dict)
            return map_info, map_info_index
        
        # if there already exist DBSCAN resulting file, then you can use it directly as long as the parameters match
        cache = self.path + '/' + cache
        if os.path.exists(cache):
            with open(cache, 'rb') as f:
                print(f'loaded cached cluster model {cache}')
                
                cluster_info = pickle.load(f)
                
                idx_count = 0
                for start, frames in self.idx.items():
                    if len(frames) == 0:
                        continue
                    for frame in frames:
                        idx_count += 1

                cached_count = 0
                for key1, value1 in cluster_info.items():
                    for key2, value2 in value1.items():
                        cached_count += 1
                if idx_count != cached_count:
                    raise Exception("please make sure to use the cluster that match the range selected")
        else:
            cluster_info = _gen_cluster(self)
            with open(cache, 'wb') as f:
                pickle.dump(cluster_info, f)
        # gen coordinates -> match the coordinates -> create map
        cord_info = _gen_cord(self, cluster_info)
        match_info = _gen_match(self, cord_info)
        self.map_info, self.map_info_index = _gen_map(self, match_info, cluster_info)
            
        return 
    
    def gen_speed(self):
        output = []
        p = Pool(cpu_count())
        
        output = list(
            tqdm(
                p.imap(
                        partial(parallel_f, pixel_ratio = self.pixel_ratio, dt = self.dt), zip(self.map_info, self.map_info_index)
                    )
                , total=len(self.map_info))
            )
        p.close()
        p.join()
        self.frame = []
        self.end_frame = []
        self.vs = []
        self.v_median = []
        self.v_mean = []
        self.vfield = []
        self.grid_z = []
        for i in output:
            if i is  None:
                continue
            self.frame.append(i[0])
            self.end_frame.append(i[1])
            self.v_median.append(i[2])
            self.v_mean.append(i[3])
            self.vs.append(i[4])
            # self.vfield.append(i[5])

        self.frame = np.array(self.frame)
        """
        # calculate the velocity from the activation map
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
        
        
        self.frame = []
        self.end_frame = []
        self.vs = []
        self.v_median = []
        self.v_25 = []
        self.v_75 = []
        self.v_mean = []
        self.vfield = []
        pbar = tqdm(range(len(self.map_info)))

        for i in pbar:

            activation_map = self.map_info[i]
            activation_map_index_dict = self.map_info_index[i]
            # for each activation map, if available pixels are less than one-third of the entire pixels, then skip
            if (np.isnan(activation_map).sum()) > (activation_map.shape[0] * activation_map.shape[1] / 3):
                continue
            # extract the edge of the activation map
            edge = edge_detect(activation_map)
            self.frame.append(np.nanmin(activation_map))
            self.end_frame.append(np.nanmax(activation_map))

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
            
            v = np.empty_like(t).fill(np.nan)
            v = np.divide(np.sqrt(t), t, out=v, where=notnanzero_position) * self.pixel_ratio / self.dt
            self.vfield.append(v)  
            # v = v[50:self.dim[0]-50, 50:self.dim[1]-50]
            #v = np.where(v<20, v, np.nan)
            activation_map_none = np.isnan(activation_map)
            v = np.where(activation_map_none, np.nan, v)
            wave_t_to_filter = set(activation_map[(v > 30) & (~activation_map_none)])
            all_index_to_filter = []
            for each_t in wave_t_to_filter:
                all_index_to_filter.append(activation_map_index_dict[each_t])
            all_index_to_filter = np.vstack(all_index_to_filter)
            v[all_index_to_filter[:, 1], all_index_to_filter[:, 0]] = np.nan
            
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
            self.vs.append(v)
            v = v.flatten()
            median = np.nanmedian(v)
            if np.isnan(median):
                median = 20
            self.v_median.append(median)
            v = np.where(0.5 * median < v, v, np.nan) 
            # v = np.sort(v)[round(len(v)/3):round(len(v)*2/3)]
            self.v_mean.append(np.nanmean(v))
            
            # self.v_25.append(v[round(len(v)/4)])
            # self.v_75.append(v[round(len(v)/4*3)])
            self.v_25.append(0)
            self.v_75.append(0)
            pbar.set_postfix({'median': round(median), 'q1': round(self.v_25[-1]), 'q3': round(self.v_75[-1])})
            pbar.update()

        self.frame = np.array(self.frame)
        """
    
    # export the result to csv
    def export_result(self, name):
        pd.DataFrame({
            'frame': self.frame + self.start, 
            'speed_median': self.v_median,
            'speed_mean': self.v_mean
            }).to_csv(self.path + '/' + name, index=False)
    
    # plot the v across the frames, you can choose to display 25, 50 or 75 percentile using first, second, third. You can
    # also skip the activation map which has v that is larger than skip.
    def plot(self, mean=True, median=True, skip=None, name='new.png'):

        if skip is not None:
            v_median_ = []
            v_mean_ = []
            frames = []
            for i in range(len(self.v_median)):
                if self.v_median[i] < skip:
                    frames.append(self.frame[i]+self.start)
                    v_median_.append(self.v_median[i])
                    v_mean_.append(self.v_mean[i])

            plt.figure(figsize=(24,10))
            if median:
                plt.plot(frames, v_median_, label='median')
            if mean:
                plt.plot(frames, v_mean_, label='mean')
            plt.legend()
            plt.show()
            return 
        plt.figure(figsize=(24,10))
        if median:
            plt.plot(self.frame + self.start, self.v_median, label='median')
        if mean:
            plt.plot(self.frame + self.start, self.v_mean, label='mean')
        
        plt.legend()
        plt.savefig(self.path + '/' + name)
        plt.show()