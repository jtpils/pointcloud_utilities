"""
pointcloud

Author: Chad Stainbank (ucfacms@ucl.ac.uk)

A module to read in, manipulate and plot ALS and TLS data.
Designed for use in tree crown modelling of Wytham Forest.

Note that this file is not a final product, and is regularly altered and updated. 
"""

import laspy
from glob import glob
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib.colors import LogNorm
import warnings

class PointCloud(object):
    """Extract and store point cloud data from a `laspy.file.File` object.
    
    Attributes:
        x, y, z ::: numpy arrays of scaled x, y and z coordinates
            (individual storage is preferred to a single xyz array out of consideration for memory)
    """
    # Get minimum and maximum values
    
    def __init__(self, input_data, label=None, colour=None):
        """ Args:
                input_data ::: any of the following:
                    : str path to .las, or laspy.file.File object, to read .las
                    : np array (3, n_points) to read xyz coordinate array
                    : dict conforming to _check_bounds to create bounds box object
                label ::: str label for object, used in plotting
                colour ::: str colour name (pyplot conforming) for object, used in plotting
                bounds ::: bounds dict to slice PointCloud to
        """
        
        if input_data is None:
            self._initialise_empty() 
        
        elif isinstance(input_data, str):
            self._initialise_from_filename(input_data)
        
        elif isinstance(input_data, list):
            self._initialise_from_list(input_data)
            
        elif isinstance(input_data, laspy.file.File):
            self._initialise_from_LAS(input_data)
        
        elif isinstance(input_data, np.ndarray):
            self._initialise_from_xyz_array(input_data)

        elif isinstance(input_data, dict) and _check_bounds(input_data):
            self._initialise_from_bounds(input_data)

        else: print "Invalid data, PointCloud empty"
        
        if label:
            self.label = label
        if colour:
            self.colour = colour
    """ Initialisation methods """

    def _initialise_empty(self):
        """ Initialise with nothing. """
        self.x = np.array([])
        self.y = np.array([]) 
        self.z = np.array([]) 

    def _initialise_from_LAS(self, laspy_file):
        """Extract and store point cloud data from a `laspy.file.File` object.
        
        Args:
            laspy_file ::: a `laspy.file.File` object from which to extract x,y and z arrays
            fname ::: relative path to file from which laspy_file was created
        """
        fpath = laspy_file.filename
        self.fname = fpath.split('/')[-1]
        self._data_dir = '/'.join(fpath.split('/')[:-1])
        self._extract_scaled_arrays(laspy_file) # assign x,y,z
        self._extract_header_info(laspy_file)
        self._set_descrips()
            
    def _extract_scaled_arrays(self, laspy_file):
        """Extract scaled point co-ordinates from `laspy.file` object to attributes x, y and z."""
        
        self.x = laspy_file.x
        self.y = laspy_file.y
        self.z = laspy_file.z

    def _extract_header_info(self, laspy_file):
        """Assign attributes to contain header information from `laspy.file` object.
        
        Sets:
            .fname ::: filename of data source
            ._scale ::: scaling factors applied to array values [x, y, z] 
            ._offset ::: offset applied to array values [x, y, z]    
        """
        
        self._scale = laspy_file.header.scale # factor applied to scale from integer coordinate values 
        self._offset = laspy_file.header.offset # coordinate offset
        # + whatever else you want

    def _initialise_from_filename(self, fname):
        """Initialise pointcloud object from a provided filename"""
        
        # Determine file extension
        filetype = fname.split(".")[-1].lower() # reduce to lower case

        # Call correct initialisation method
        if filetype == "las": # Initialise .las via laspy
            self._initialise_from_LAS(laspy.file.File(fname))
        elif filetype == "pcd":
            print "I can't read .%s files yet" %(filetype)
        else:
            print ".%s files unrecognised"%(filetype)
            
    def _initialise_from_list(self, fnames):
        """ Initialise PointCloud object from multiple files."""

        # Build from an empty PointCloud
        self._initialise_empty()      
        
        # Accrete all PointClouds in list
        for fname in fnames:
            self += PointCloud(fname)

        # Set descriptory attributes
        self._set_descrips()
        self.fname = fnames
    
    def _initialise_from_xyz_array(self, xyz_array):
        """Extract x, y and z coordinates from an array.
        
        Args:
            xyz_array ::: 3-row 2D numpy array of coordinates; [0] = x, [1] = y, [2] = z
        """
        
        assert xyz_array.shape[0] == 3, "Array needs exactly 3 rows (x, y, z)"
        # Assign x, y and z attributes
        self.x = xyz_array[0]
        self.y = xyz_array[1]
        self.z = xyz_array[2]

        self._set_descrips()

    def _initialise_from_bounds(self, bounds):
        """Assign vertices of bounds box as pointcloud.
        Args:
            bounds ::: dict of (min, max) values for each coord x, y and z 
        """
        # Convert bounds to array of coords
        vertices = vertices_from_bounds(bounds)
        vertices_array = np.array(vertices).T
        # Initialise from array
        self._initialise_from_xyz_array(vertices_array)

    def _set_descrips(self):
        """Assign descriptory attributes of pointcloud."""

        self.n_points = self._get_n_points() # length of data
        if self.n_points == 0:
           # print "There is no data"
            pass
        else:
            self.bounds = self._get_box_bounds()
    
    """ Magic methods """

    def __add__(self, other):
        """ Combine two PointClouds by concatenating respective coordinates, losing any other attributes """
        
        # Concatenate coordinates 
        xs = np.concatenate([self.x, other.x])
        ys = np.concatenate([self.y, other.y])
        zs = np.concatenate([self.z, other.z])
        
        # Package into xyz array
        xyz = np.vstack([xs, ys, zs])

        return PointCloud(xyz)  
 
    def __iadd__(self, other):
        """ Add another PointCloud to present by concatenating coordinates, preserving all other attributes """

        xs = np.concatenate([self.x, other.x])
        ys = np.concatenate([self.y, other.y])
        zs = np.concatenate([self.z, other.z])

        self.x = xs
        self.y = ys
        self.z = zs
        
        # Update descriptors
        self._set_descrips()
         
        return self

    def __getitem__(self, key):
        """ Slice by bounds dict, or apply any valid numpy slice to individual coordinates to generate new sliced PointCloud. """
        
        if isinstance(key, dict): # bounds dict slicing
            PC = self.slice(key)
        else: # np array slicing
            xs = self.x[key]
            ys = self.y[key]
            zs = self.z[key]

            xyz = np.vstack([xs, ys, zs])
            PC = PointCloud(xyz)
        
        return PC

    """ Getting methods """

    def _get_n_points(self):
        """Determine how many points the pointcloud has."""
        if self.check_alignment(): # ensure data is good
            return self.x.shape[0] # return number of points    

    def _get_box_bounds(self):
        """Return dict of the (min, max) bounds of the 'box' which surrounds the pointcloud."""
    
        # Check that data actually exists
        if not any(self.x):
            return None

        # Find min and max values in each coordinate
        bounds = {coord : _find_array_range(getattr(self, coord)) for coord in ['x', 'y', 'z']}
        return bounds

    def _get_box_ranges(self):
        """Return dict of the range (max-min) in each of x,y and z for pointcloud box."""
        
        # Find range for each coord
        ranges = {coord: c_bounds[1]-c_bounds[0] for coord, c_bounds in self.bounds.iteritems()}
        return ranges

    def _get_box_volume(self):
        """Return volume of pointcloud box."""

        ranges = self._get_box_ranges()
        volume = ranges['x'] * ranges['y'] * ranges['z']
        return volume
    
    def _get_box_areas(self):
        """Return dict of areas of faces of pointcloud box."""
        ranges = self._get_box_ranges()
        # 3 faces        
        area_tuples = (('x', 'y'), ('x', 'z'), ('y', 'z'))
        areas = {dim1+dim2 : (ranges[dim1]*ranges[dim2]) for dim1, dim2 in area_tuples}
        return areas    

    def _get_point_density_2D(self):
        """Return number of points per m^2 (in xy) """
        point_density = self.n_points/self._get_box_areas()['xy']
        return point_density

    def _get_point_density_3D(self):
        """Return number of points per m^3"""
        point_density = self.n_points/self._get_box_volume()
        return point_density

    """ Other methods """

    def rasterise(self, **kwargs):
        """Convert pointcloud to 2D raster image.
        
        Args:
            PC ::: PointCloud object
            x_axis, y_axis ::: coordinates to rasterize (default: x,y)
        Returns:
            img ::: 2D numpy array, image of pointcloud space where values are m^2 point density 
            x offset, y_offset ::: the offsets used to shift data to origin
            
        Adapted from SO post by 'Luke' (http://stackoverflow.com/a/6658307)
        """
        
        PC = self # give alternate name

        # Get plotting parameters
        x_axis, y_axis, bounds = _retrieve_pars(
        ['x_axis', 'y_axis', 'bounds'],
        PC, **kwargs)
       
        # Slice and subsample
        PC, = _slice_and_subsample(bounds, None, PC)
        
        # Get the selected data
        xx = getattr(PC, x_axis)
        yy = getattr(PC, y_axis)

        # floor data by conversion to int
        xx = xx.astype(int)
        yy = yy.astype(int)

        # Shift data to origin
        x_offset, y_offset = xx.min(), yy.min() # determine offsets from 0
        xx = xx - x_offset
        yy = yy - y_offset

        # Assign points to pixels
        data = np.vstack((xx, yy))
        img = np.zeros((xx.max()+1, yy.max()+1), dtype='int')  # blank image, with 1 pixel in each dim per possible value
        for i in xrange(data.shape[1]):  # increment over pixels
            img[data[0,i], data[1,i]] += 1
        img = np.rot90(img) # Rotate to right way round
        
        return img, (x_offset, y_offset)
        
    def check_alignment(self):
        """Return True if there is 1:1:1 alignment of x, y and z coordinates."""
        
        if self.x.shape == self.y.shape == self.z.shape:
            return True
        else:
            print "coordinates don't align x:%s y:%s z:%s" % (
            self.x.shape, self.y.shape, self.z.shape)
        
    def return_xyz(self):
        """Combine individual coordinates to a 3 row (x,y,z) array of point coordinates."""
        
        # Ensure data is good
        assert self.check_alignment(), "x %s, y %s and z %s do not align" % (self.x.shape, self.y.shape, self.z.shape)
        # Combine x, y and z into single array
        return np.array([self.x, self.y, self.z])
    
    def copy_pointcloud(self):
        """Return a copy of the current pointcloud (x, y and z only) 
        Returns:
            PC_copy ::: a new PointCloud object containing only copies of 
                        the x, y and z attributes of this instance.
        """
        
        # Initialise a new PC object from copies of current instances pointcloud arrays
        PC_copy = PointCloud(np.array(
            [getattr(self, coord).copy() for coord in ['x', 'y', 'z']]))
        return PC_copy
    
    def subsample(self, limit, scale = None):
        """Return a new PointCloud object with a random subsample of current pointcloud.

        Args:
            scale ::: Scale factor for size reduction (i.e int steps in slice indexing)
            limit ::: Set a maximum number of points to reduce to (takes precedence over scale)
        Returns:
            PC_subsampled ::: A PointCloud subsample of the pointcloud of this instance (x, y and z)
        """

        if scale and not limit:  # convert scale to limit
            limit = int(np.floor(self.n_points/(1.0*scale)))
        elif not scale and not limit: # if nothing provided
            limit = self.n_points
        
        if limit >= self.n_points: # avoid unecessary sampling in case of copy
            return self.copy_pointcloud()

        # Randomly choose indices
        indices = np.arange(self.n_points)
        selection = np.random.choice(indices, limit, False) # no replacement
        selection.sort() # maintain order
        
        # Generate new PointCloud object from array of copies of sample of x, y and z
        PC_subsampled = PointCloud(np.array(
                    [getattr(self, coord)[[selection]] for coord in ('x', 'y', 'z')]))
        
        return PC_subsampled
    
    def subsample_quick(self, scale=100, limit=None):
        """A quicker subsampling method, using less robust stepwise slicing.

        Args:
            scale ::: Scale factor for size reduction (i.e int steps in slice indexing)
            limit ::: Set a maximum number of points to reduce to (takes precedence over scale)
        Returns:
            PC_subsampled ::: A PointCloud subsample of the pointcloud of this instance (x, y and z)
        """
        
        # Determine scale from limit, if necessary
        if limit:
            scale = np.ceil((float(self.n_points)/limit)) # ratio of desired to actual points, rounded up
 
        # Generate new PointCloud object from array of copies of sample of x, y and z
        PC_subsampled = PointCloud(np.array(
                [getattr(self, coord)[::scale].copy() for coord in ('x', 'y', 'z')]))
        
        return PC_subsampled


    def store_xyz(self):
        """Store xyz array as attribute.
        
        Note that this is not recommended, as the object unnecessarily doubles the size.
        Most methods are designed to operate on x, y and z seperately due to speed and memory considerations.
        """
        self.xyz = self.return_xyz()
    
    """ Description methods / utils """
    

    def describe(self, human = False):
        """Return a summary of pointcloud values.

        Args:
            human ::: bool choice to print data in human readable form instead
        Returns:
            ::: a tuple summarising the pointcloud:
                (number of points, ground area, dict of x, y and z summaries (range, min, max))
            ::: (opt) A human readable description of the above
            """

        # Don't describe missing data
        if self.n_points == 0:
            if human:
                print "No data points"
            # Return an essentially empty set of values
            return (self.n_points, None, {coord: (None, None, None) for coord in ['x', 'y', 'z']})


        summary = {'n_points': self.n_points, 'point_density' : self._get_point_density_2D(), 'bounds': self.bounds,
        'ranges': self._get_box_ranges(), 'areas': self._get_box_areas(), 'volume': self._get_box_volume()}

        if human: # Print summary of pointcloud
            describe_summary(summary)
        else: return summary

        
    """ Slicing/ General bounds methods """
    


    def get_pointBox(self):
        """Return the vertices of pointcloud boundary Box.
        
        Returns:
            ::: PointCloud object specifying 8 points marking the vertices (A, B, ..., H) of pointBox;
                the conceptual axes-aligned cuboid which encloses the pointcloud of this instance
        """
        
        # Find vertices from bounds and convert to PC object
        return generate_pointcloud_from_coord_tuples(vertices_from_bounds(self.bounds))
    
    def slice(self, bounds):
        """Slice instance pointcloud by providing new bounds.
        
        Args:
            bounds ::: dict of (lower, upper) bound values of pointcloud slice in dimensions (x, y and z)
                             missing dims, or None values, will be replaced by existing bounds (i.e. updating)
        Returns:
            PC_slice ::: new PointCloud object with sliced pointcloud
        """

        # Fill out bounds
        bounds = _update_bounds(self.bounds, bounds)
        
        # Create slice
        PC_slice = slice_pointcloud(self, bounds)

        return PC_slice

    def slice_from_frac(self, xs_centre = 0.5, ys_centre = 0.5, zs_centre = 0.5,
                           xs_width = 1e-2, ys_width = 1e-2, zs_width = 0.5):
        """Slice instance pointcloud by fractional bounds.
        
        Args: 
            xs_centre, ys_centre, zs_centre ::: fractional central coordinates of desired slice 
            xs_width, ys_width, zs_width ::: desired widths of slice
        
        Returns:
            PC_slice ::: new PointCloud object with sliced pointcloud
        """
        # Translate generic fraction bounds to specific slice bounds
        frac_bounds = make_frac_bounds(xs_centre, ys_centre, zs_centre, xs_width, ys_width, zs_width)
        bounds = make_slice_bounds_from_fracs(self.bounds, frac_bounds)
        
        return self.slice(bounds)

""" Functions and utilities """

def describe_summary(summary):
    """Translate the summary dict into a human readable format."""
    
    # Flatten dicts to sorted k:v lists for printing        
    ranges = tuple([item for pair in sorted(summary['ranges'].items()) for item in pair])
    areas = tuple([item for pair in sorted(summary['areas'].items()) for item in pair])
    
    p_n = "Points: \n\t %s \n" % summary['n_points']
    p_d = "Point density: \n\t %s points per m^-2\n" % summary['point_density']
    p_r = "Coordinate Ranges:\n\t %s: %s \n\t %s: %s \n\t %s: %s \n" % ranges
    p_v = "Box Volume: \n\t %s \n" % summary['volume']
    p_a = "Box Face Areas: \n\t %s: %s \n\t %s: %s \n\t %s: %s \n" % areas
    
    print p_n, p_d, p_r, p_a, p_v

def _check_bounds(candidate):
    """Check if candidate is a valid bounds dict.

    Args:
        candidate ::: an object to be tested for validity as bounds dict
    Returns:
        True ::: if object is a bounds dict (note that frac_bounds is indistinguishable from bounds)
        (else) False
    """
    # Check candidate is dict
    is_dict = type(candidate) == dict
    if not is_dict:
        print "candidate is not a dict. \n\t type: %s"%type(candidate)
        return False
    # Check candidate has exactly 3 items
    has_3_items = len(candidate) == 3
    if not has_3_items:
        print "candidate does not have exactly 3 items. \n\t len: %s"%len(candidate)
        return False
    # Check candidate has 'x', 'y' and 'z'
    has_xyz = [coord in ['y', 'x', 'z'] for coord in candidate.iterkeys()]
    if not  all(has_xyz):
        print "candidate does not have x, y and z keys. \n\t keys: %s"%candidate.keys()
        return False
    # Check candidate tuples have exactly 2 elements
    have_2_elements = [len(value) == 2 for value in candidate.itervalues()]
    if not all(have_2_elements):
        print "candidate does not have 2 elements in each tuple. \n\t values: %s"%candidate.values()
        return False
    # Check candidate tuples represent valid bound order ((x1, x2) etc.)
    are_ordered = [value[0] <= value[1] for value in candidate.itervalues()]
    if not all(are_ordered):
        print "candidate tuples do not obey (min, max) order.  \n\t values: %s"%candidate.values()
        return False
    
    else:
        return True

def _find_array_range(array):
    """Return the (min, max) values of a numpy array."""
    return (array.min(), array.max())

def _unpack_AH(AH):
    """Construct a dict of coord bounds from AH."""
    
    # Loop over (x, y, z) to assign dict coords from A and H
    A, H = AH
    bounds = {coord: (A[i], H[i]) for i, coord in enumerate(['x','y','z'])}
    
    return bounds

def vertices_from_bounds(bounds):
    """Determine the points of the vertices of the box defined by bounds.
    Args:
        bounds ::: dict of (min, max) values for each coord in pointcloud
    Returns:
        vertices ::: The (x,y,z) coordinates of all box vertices (A,B,C,D,E,F,G,H)
    """

    # 'Unpack' bounds into full set of vertices coordinates
    vertices = [(bounds['x'][x], bounds['y'][y], bounds['z'][z]) # build coordinate tuple (x,y,z)
        for x in (0,1) for y in (0,1) for z in (0,1)] # from A(111) to to H (222)
    return vertices

def _update_bounds(bounds, update = {'x': None, 'y': None, 'z': None}):
    """Update a coordinate bounds with new values.
    
    Args:
        bounds ::: dict of (min, max) bounds for each coordinate ('x', 'y' and 'z')
        update ::: dict of new bounds for any coordinates
    
    Returns:
        updated ::: bounds, with relevent values replaced by those in update
    """
    updated = dict(bounds) # copy bounds dict
    for coord, c_bounds in update.iteritems(): # update bounds
        if c_bounds: # skip any None's
            updated[coord] = c_bounds
    return updated

def generate_pointcloud_from_coord_tuples(coord_tuples):
    """Convert any set of coordinate tuples to a pointcloud object.
    Args:
        coord_tuples ::: a list or tuple of coordinate tuples
                         e.g. [(x_1,y_1,z_1),(x_2,y_2,z_2), ...]
                         Typically ABCDEFGH coordinates
    Returns:
        PC ::: A PointCloud object with coordinates in x, y and z attributes
    """
    # Empty lists to store x, y and z arrays
    x_array = []
    y_array = []
    z_array = []

    # Seperate coordinate elements from set of (x,y,z) tuples
    for (x, y, z) in coord_tuples:
        x_array.append(x)
        y_array.append(y)
        z_array.append(z)

    # Generate new PointCloud object    
    return PointCloud(np.array([x_array, y_array, z_array]))

def _check_frac_bounds(frac_bounds):
    """Ensure that fractional bounds dict is full and valid.
    
    Args:
        frac_bounds ::: dict of fractional (lower_bound, upper_bound) for any coordinates.
    Returns:
        frac_bounds ::: updated to:
                        - fill any missing coordinates with (0., 1.), i.e no selection
                        - trim any values outside of 0 -- 1 to within fractional range, with warning
    """
    for coord in ['x', 'y', 'z']:
        # Fill in missing coordinates
        if coord not in frac_bounds.keys():
            frac_bounds[coord] = (0., 1.)
        else:
            # Ensure bounds are fractional
            l_frac, u_frac = frac_bounds[coord]
            if l_frac < 0 or u_frac > 1:
                warnings.warn("Selection outside fractional bounds and will be be trimmed in %s"%coord)
                # Trim values to within fractional bounds, and ensure float
            frac_bounds[coord] = (max(l_frac, 0.), min(u_frac, 1.))
            
    return frac_bounds

# copied from construct_slice_dict(), only name and varnames changed
def make_frac_bounds(xs_centre = 0.5, ys_centre = 0.5, zs_centre = 0.5, xs_width = 1e-2, ys_width = 1e-2, zs_width = 0.5):
    """ Create a dict which is used for slicing cuboids from pointcloud arrays centred on fractional coordinates.
    
    Args:
        xs_centre, ys_centre, zs_centre ::: fractional central coordinates of desired slice 
        xs_width, ys_width, zs_width ::: desired widths of slice
        (defaults to a full-z column on centre of array with 1e-6 fractional xy area)
    Returns:
        frac_bounds ::: dict of fractional (lower_bound, upper_bound) for each coordinate 
    """
    # Check centres are fractions
    assert not any([centre < 0 or centre > 1 for centre in (xs_centre, ys_centre)]), "Centre values must be fractional"
    
    # Ensure all inputs are float
    xs_centre, ys_centre, zs_centre, xs_width, ys_width, zs_width = map(float,
                                    [xs_centre, ys_centre, zs_centre, xs_width, ys_width, zs_width])
    # Construct dict
    frac_bounds = {"x" : (xs_centre-xs_width, xs_centre+xs_width),
                   "y" : (ys_centre-ys_width, ys_centre+ys_width),
                   "z" : (zs_centre-zs_width, zs_centre+zs_width)}
    
    # Check values are valid
    frac_bounds = _check_frac_bounds(frac_bounds)

    return frac_bounds

def convert_f_bounds(c_bounds, f_bounds):
    """Apply fractional bounds to determine upper and lower bounds of a 1D array.
    
    Args:
        c_bounds ::: tuple (min, max) values of the array of  1 dimensions of coords in (e.g x, or y, or z)
        f_bounds ::: tuple (lower, upper) fractional bounds to be applied to range of 1D coord array 
    Returns:
        s_bounds ::: tuple (lower, upper) bound values of slice (from f_bounds) of coord array (from c_bounds) 
    """
    # Unpack tuples
    l_bound, u_bound = f_bounds
    c_min, c_max = c_bounds
    # Determine bounds
    c_range = c_max - c_min # size of value range
    s_1 = c_min + c_range * l_bound # lower bound
    s_2 = c_min + c_range * u_bound # upper bound
    # Pack to tuple
    s_bounds = (s_1, s_2)
    return s_bounds

def make_slice_bounds_from_fracs(bounds, frac_bounds):
    """Apply fractional bounds to determine coord bounds of 3D slice of a pointcloud.
    
    Args:
        bounds ::: dict of (min, max) values for each coord in pointcloud
        frac_bounds ::: dict of fractional (lower_bound, upper_bound) for x, y and z coordinates
    Returns:
        bounds ::: dict of (lower, upper) bound values of pointcloud slice in each dimension (x, y and z)
    """
    # Construct dict of s_bounds for each coord from f_bounds and c_bounds
    bounds = {coord: convert_f_bounds(bounds[coord], frac_bounds[coord]) for coord in ['x', 'y' , 'z']}
    
    return bounds


def select_coord_slice(c_array, s_bounds):
    """Select array values within specified slice bounds.

    Args:
        c_array ::: array of x, y or z component of pointcloud coordinates 
        s_bounds ::: tuple (lower, upper) bound values of slice to create from c_array  
    Returns:
        c_mask ::: bool array indicating indices of array values within slice
    """
    
    # Unpack s_bounds for lower and upper slice bounds
    s_1, s_2 = s_bounds
    # Determine position of values within slice bounds
    c_mask = np.logical_and(np.greater_equal(c_array, s_1), np.less_equal(c_array, s_2))
    
    return c_mask

def make_slice_mask(PC, bounds, previous_selection = None):
    """Select indices of points within specified 3D slice of pointcloud.
    
    Args:
        PC ::: PointCloud object with x, y and z attributes
        bounds ::: dict of (lower, upper) bound values of pointcloud slice in each dimension (x, y and z) 
        previous (opt) ::: bool array from prior filtering to build on
                           shape must broadcast to self.{x,y,z}.shape
    Returns:
        slice_mask ::: bool array indicating indices of pointcloud values within 3D slice
    """
    
    # Initial mask
    if previous_selection:
        slice_mask = previous_selection # build on previous base
    else: 
        slice_mask = True # build on blank if no previous
        
    # Iterate through dimensions to build 3D slice mask
    for coord, s_bounds in bounds.iteritems():
        slice_mask = np.logical_and(slice_mask, # retain only elements within new slice
                                    select_coord_slice(getattr(PC, coord), s_bounds))
        
    return slice_mask

def _apply_selection(PC, selection):
    """Return a new PC object by applying selection.
    
    Args:
        PC ::: A PointCloud object with coordinates in x, y and z attributes
        selection ::: a 1D bool array indicating indices of pointcloud points to return
    Returns:
        PC2 ::: A PointCloud object containing only selected coordinates of PC
    """   
        
    PC2 = PointCloud(np.array( # array from generator object of
            [getattr(PC, coord)[selection] # selected indices of array
             for coord in ['x', 'y', 'z']])) # for each coord
    return PC2

def slice_pointcloud(PC, bounds):
    """Slice PointCloud within bounds.
    
    Args:
        PC ::: A PointCloud object with coordinates in x, y and z attributes
        bounds ::: dict of (lower, upper) bound values of pointcloud slice in each dimension (x, y and z) 
    Returns:
        PC_slice ::: new PointCloud object with sliced pointcloud
    """
    
    # Convert slice bounds to selection
    slice_mask = make_slice_mask(PC, bounds)

    # Generate sliced PointCloud
    PC_slice = _apply_selection(PC, slice_mask)
    
    return PC_slice

### Tiles ###
def extract_tile_bounds(fpaths):
    """ Determine the bounds for each tile (a .las file) in fpaths.
    
    Args:
        fpaths ::: list of str filepaths of .las files, or
                :: str path to directory containing .las files
    Returns:
        tiles_bounds ::: dict of bounds dicts for each file
                        i.e. `{fpath: bounds}` where `bounds = {'x': (min, max), ...}`.
    """
    
    # Find all .las files in directory
    if isinstance(fpaths, str):
        fpaths = glob(fpaths + '*.las')

    tiles_bounds = {} # to store bounds for each tile
    # Iterate over all .las files, finding bounds (takes a min)
    for fname in fpaths:
        with laspy.file.File(fname) as lp:
            bounds = dict(zip(('x', 'y', 'z'),
                          zip(lp.header.min, lp.header.max)))
            tiles_bounds[fname] = bounds
    
    return tiles_bounds

def select_tiles(tiles_bounds, area_bounds):
    """ Return the subset of tiles which contain data in specified area."""
    # is any part of (min, max) tuple c1 in range of c2?
    c_overlap = lambda c1, c2: any([c >= c2[0] and c <= c2[1] for c in c1])
    # is there overlap of b1 with all of the dimensions in b2?
    bounds_overlap = lambda b1, b2: all([c_overlap(b1[coord], b2[coord]) for coord in b2.keys()])
    
    # Filter dict to fpaths with overlapping area
    subset = {fpath: bounds for fpath, bounds in tiles_bounds.iteritems()
              if bounds_overlap(bounds, area_bounds)}
    return subset

### Non-PC utilities ###

def find_array_range(*arrays):
    """ Determine the integer range of values for a set of arrays.
    Args:
        *arrays ::: Any number of numpy arrays
    Returns:
        low, high ::: the floor and ceil of the min and max values in *arrays
        ! None ::: if all arrays are empty
    """
    # Filter out zero length arrays
    arrays = [array for array in arrays if len(array)]
    try:
        # Find ranges, coerced to interger
        low = np.floor(np.min([np.min(array) for array in arrays])) # floor to integer (neatness)
        high = np.ceil(np.max([np.max(array) for array in arrays])) # ceil to integer
    
    except ValueError: # Where all arrays are empty
        return None
    
    return low, high

def histogram(*arrays, **kwargs):
    """ Generate aligned histograms of supplied arrays.
    
    Args:
        *arrays ::: 1D arrays
        **kwargs ::: Histogram options;
                 : density : bool option for probability density function (default=False)
                             (see numpy.histogram() documentation for explanation)
                 : bin_width : numeric size of bins
    Returns:
        bin_edges ::: array of bin edge values
        hists ::: list of arrays of bin values (densities or counts) for each PC
    """

    # Unpack kwargs
    bin_width = kwargs.pop('bin_width', 1.)
    density = kwargs.pop('density', False)
    if kwargs:
        raise TypeError, 'Unused kwargs: %s'%kwargs
    
    try: # Get minimum and maximum values
        low, high = find_array_range(*arrays)
    except TypeError: # If there are no values
        return None, len(arrays)*[None]

    # Generate bins
    bin_edges = np.linspace(low, high, (high-low)/bin_width+1) # cover entire range in even bins
    
    # Retreive bin counts
    hists = [np.histogram(array, bins=bin_edges, density=density)[0] for array in arrays]
    
    return bin_edges, hists

def pmf(hist):
    """ Return the probability mass function for each element in an array of counts, such that sum(pmf(hist)) == 1. """
    return hist/float(np.sum(hist)) 

def rmse(o, e):
    """ Determine root mean square error of fit between observed and expected arrays"""
    return np.sqrt(np.mean(((o-e)**2)))

"""Box comparison functions """

def _compare_n_points(PC1, PC2):
    """Determine the ratio of number of points in two pointclouds."""
    n1 = PC1.n_points
    n2 = PC2.n_points
    return 1.0*n2 / n1

def _compare_point_density(PC1, PC2):
    """Determine the ratio of point density between two pointclouds."""
    d1 = PC1._get_point_density_2D()
    d2 = PC2._get_point_density_2D()
    return d2 / d1

def _compare_coord_ranges(PC1, PC2):
    """Determine the ratio of  ranges in each coordinate between two pointcloud boxes."""
    r1 = PC1._get_box_ranges()
    r2 = PC2._get_box_ranges()
    return {coord: r2[coord]/coord_range for coord, coord_range in r1.iteritems()}

def _compare_box_areas(PC1, PC2):
    """Determine the ratio of three face areas between two pointcloud boxes."""
    a1 = PC1._get_box_areas()
    a2 = PC2._get_box_areas()
    return {area_name : a2[area_name] / area for area_name, area in a1.iteritems()}

def _compare_box_volumes(PC1, PC2):
    """Determine the ratio of volumes between two pointcloud boxes."""
    v1 = PC1._get_box_volume()
    v2 = PC2._get_box_volume()
    return v2 / v1

def _evaluate_bounds_inferiority(bounds_a, bounds_b):
    """Assess inferiority of one set of bounds to another, element-wise.
    
    Args:
        bounds_a, bounds_b ::: bounds dicts to compare to one another
    Returns:
        a_lt_b ::: dict of 2 2-tuples for each coord x, y and z in bounds a and b
                   e.g 'x': ((x^a_1 < x^b_1, x^a_1 < x^b_2), (x^a_2 < x^b_1, x^a_2 < x^b_2))
    """
    
    a_lt_b = {}
    # for each dim, compare d^a_1 (then d^a_2) to d^b_1 and d^b_2, with true if less than
    for coord, (a1, a2) in bounds_a.iteritems():
        # unpack min and max of that coord for b
        b1, b2 = bounds_b[coord]
        # test for inferiority and assign to dict
        a_lt_b[coord] = (a1 < b1, a1 < b2), (a2 < b1, a2 < b2)
    return a_lt_b

def _make_infer_cube(a_lt_b):
    """Convert dict of bounds inferiority to 3D array.
    
    Args:
        a_lt_b ::: dict of 2 2-tuples for each coord x, y and z in bounds a and b
                   usually the result of evaluate_bounds_inferiority()
    Returns:
        infer_cube ::: 3D array of bools:
                       [i,:,:] = coords [x, y, z]
                       [i,j,:] = bounds_a [a_1, a_2]
                       [i,j,k] = bounds_b [b_1, b_2]
    """
    
    # Ball up into a 3D array [x, y, z [a1, a2 [b1, b2]]]
    infer_cube = np.array([a_lt_b[coord] for coord in ['x', 'y', 'z']])
    return infer_cube

def _sum_infer_cube(infer_cube):
    """Return sum of row sums col sums for each coord in a cube of bools.
    
    Args:
        infer_cube ::: 3-layer (x, y, z) 3D array of bools of inferiority
    Returns:
        infer_sums ::: list of sum of inferiority bools for x, y and z
    """
    # Sum each layer
    infer_sums = [np.sum(c_infers) for c_infers in infer_cube]
    #infer_sums = [np.sum([c_infers.sum(axis=0), c_infers.sum(axis=1)]) for c_infers in infer_cube]
    return infer_sums

def _determine_within_type(cba, cbb):
    """Return a modifier after determining how a and b relate.
    Args:
        cba, cbb ::: tuple (min, max) bounds for a single coordinate of two boxes
    Returns:
        type_modifier ::: int value to be added to the 'within' position signature
    """
    # Unpack bounds
    a1, a2 = cba
    b1, b2 = cbb
    
    # Check which way round bounds are
    if a1 == a2 and b1 == b2: # identical bounds
        type_modifier = 0
    elif b1 <= a1 and a2 <= b2: # a within b
        type_modifier = 3
    elif a1 <= b1 and b2 <= a2: # b within a
        type_modifier = 4
        
    return type_modifier

def _compare_box_positions(PC1, PC2):
    """Determine the 'signature' of the relative spatial positions of two boxes.
    
    Args:
        bounds_a, bounds_b ::: bounds dicts of two pointcloud boxes to be compared
    Returns:
        position_sig ::: dict of bounds comparison signature, values indicating
                         the relative positions of boxes in each of x, y and z
                           
    Explanation:
        In each coord, bounds of two boxes can either make 'no contact', 'overlap', 
        or one can be 'within' the other. The table below illustrates the meaning
        of values in comparison_sig for boxes {a} and [b].
        'o-wise' means 'is towards the coordinate origin (i.e. 0) of'
            
            value |    pictogram    | description
            ------|-----------------|--------------
            0     | [ ] {}          | no contact (b o-wise a)
            1     | [ {] }          | overlap (b o-wise a)
            2     | ( )             | identical
            3     | { [} ]          | overlap (a o-wise b)
            4     | { } [ ]         | no contact (a o-wise b)
            5     | [{ }]           | a within b
            6     | {[ ]}           | b within a
    """
    # Get bounds of boxes
    bounds_a = PC1.bounds
    bounds_b = PC2.bounds
    
    # Make comparisons of inferiority
    a_lt_b = _evaluate_bounds_inferiority(bounds_a, bounds_b)
    # Shape into a 3D array
    infer_cube = _make_infer_cube(a_lt_b)
    # Sum the rows and columns
    infer_sums = _sum_infer_cube(infer_cube)
    # Wrap into dict
    position_sig = dict(zip(['x', 'y' ,'z'], infer_sums))
    
    # Modify position signature if 'within' to determine type
    for coord, sig in position_sig.iteritems():
        if sig == 2:
            within_modifier = _determine_within_type(bounds_a[coord], bounds_b[coord])
            position_sig[coord] += within_modifier # update sig value
    
    return position_sig

def _describe_comparison(comparison):
    """Translate the comparison dict to human readable format."""
    
    # Translate position signatures to human readable format
    sig_dict = {0: 'no contact (B o--> A)', 1: 'overlap (B o--> A)', 2: 'A and B identical',
            3: 'overlap (A o--> B)', 4: 'no contact (A o--> B)', 5: 'A within B', 6: 'B within A'}
    translated_sigs = {coord: sig_dict[sig] for coord, sig in comparison['position'].iteritems()}

    # Flatten dicts to sorted k:v lists for printing
    ranges = tuple([item for pair in sorted(comparison['ranges'].items()) for item in pair])
    areas = tuple([item for pair in sorted(comparison['areas'].items()) for item in pair])
    sigs = tuple([item for pair in sorted(translated_sigs.items()) for item in pair])

    # Set up description strings
    p_n = "Points: \n\t %s \n" % comparison['n_points']
    p_d = "Point density: \n\t %s \n" % comparison['point_density']
    p_r = "Coordinate Ranges:\n\t %s: %s \n\t %s: %s \n\t %s: %s \n" % ranges
    p_v = "Box Volume: \n\t %s \n" % comparison['volume']
    p_a = "Box Face Areas: \n\t %s: %s \n\t %s: %s \n\t %s: %s \n" % areas
    p_p = "Positions: \n\t %s: %s \n\t %s: %s \n\t %s: %s \n" % sigs 
    # Print 
    print "POINTCLOUD 1 : POINTCLOUD 2\n\n", p_n, p_d, p_r, p_a, p_v, p_p

def compare_boxes(PC1, PC2, human = False):
    """Compare the relative sizes and positions of two pointcloud boxes.
    
    Args:
        PC1, PC2 ::: two pointcloud boxes to compare to one another
        human ::: optionally describe in human readable format
    Returns:
        comparison ::: dict of ratios of PC1: PC2 in number of points, coordinate ranges,
                       face areas and volume, and signature of relative box position 
    """
       
    points_ratio = _compare_n_points(PC1, PC2)
    density_ratio = _compare_point_density(PC1, PC2)
    ranges_ratios = _compare_coord_ranges(PC1, PC2)
    areas_ratios = _compare_box_areas(PC1, PC2)
    volume_ratio = _compare_box_volumes(PC1, PC2)
    position_sig = _compare_box_positions(PC1, PC2)
    
    comparison = {'n_points': points_ratio, 'point_density': density_ratio, 'ranges': ranges_ratios, 
                'areas': areas_ratios, 'volume': volume_ratio, 'position' : position_sig}
    
    if human == True:
            _describe_comparison(comparison)
    else: return comparison

""" Plotting """
""" Plotting utilities """

# Module level global plotting parameters
default_pars = {
'limit': 20000, # limit for subsampling,
'colours': ['red', 'blue', 'orange', 'purple', 'pink', 'green'], # plotting point colours
'x_axis': 'x', 'y_axis': 'y', 'z_axis': 'z', # plotting axes,
'hist_axis': 'z', # histogram axis
'bin_width': 0.1, # histogram width
'density': False, # histogram density (or counts)
'figsize': (6, 6)} # figure size

def _get_pars(*PCs, **kwargs):
    """ Return plotting parameters updated by PointCloud and user arguments.

        Args:
            *PCs ::: PointCloud objects
            **kwargs ::: plotting parameters to update defaults
        Returns:
            pars ::: dict of plotting parameters determined from PCs
    """
    
    # Initialise parameters from module defaults
    pars = default_pars.copy()
    
    # Update with parameters determined from PointCloud objects
    auto_pars = _auto_pars(*PCs)
    pars.update(auto_pars)
    
    # Update with user-defined parameters
    pars.update(kwargs)
    
    return pars

def _auto_pars(*PCs):
    """ Determine plotting parameters from PointClouds.

        Args:
            *PCs ::: PointCloud objects
        Returns:
            pars ::: dict of plotting parameters extracted from PCs
    """

    # Check all objects are PointClouds (not necessary, learn to quack)
    # assert all([isinstance(PC, PointCloud) for PC in PCs]), 'Need PointCloud objects'
    
    # Copy defaults
    def_colours = list(default_pars['colours'])
    def_colours.reverse()

    # Lists to accumulate parameters
    labels = []
    colours = []

    # Retrieve parameters from PCs
    for i, PC in enumerate(PCs):
        # Use fname until a proper labelling system employed
        labels.append(getattr(PC, 'label', getattr(PC, 'fname', str(i))))
        colours.append(getattr(PC, 'colour', def_colours.pop())) 
    
    # Pack parameters dictionary
    pars = {'labels': labels, 'colours': colours}    
    
    return pars

def _slice_and_subsample(bounds=None, limit=None, *PCs):
    """ Optionally slice and subsample the input PointClouds

        Args:
            bounds ::: dict of coordinate ('x', 'y' and/or 'z') bounds to slice PCs to
            limit ::: Maximum number of points to return for each PC by subsampling
            *PCs ::: PointCloud objects to operate on
        Returns:
            PCs ::: PointCloud objects, slice and/or subsampled according to arguments
    """

    # Slice PointClouds to bounds
    if bounds:
        PCs = tuple((PC.slice(bounds) for PC in PCs))

    # Subsample PointClouds to size limit
    if limit:
        PCs = tuple((PC.subsample(limit) for PC in PCs))
    
    return PCs

def _retrieve_pars(requested_pars, *PCs, **kwargs):
    """ Retrieve requested pars from PointClouds and defaults.
    
        Args:
            requested_pars ::: list of string paramater names to return
            *PCs ::: PointCloud objects from which certain parameters are determined
            **kwargs ::: User-defined parameters
        Returns:
            pars ::: list of parameter values, in same order as requested_pars 
    """

    # Update default pars with kw arguments
    all_pars = _get_pars(*PCs, **kwargs)

    # Create list of requested pars (None as default value)   
    pars = [all_pars.get(par, None) for par in requested_pars]
    
    return pars

""" Plotting functions """

def plot_3D(*PCs,**kwargs):
    """Plot pointcloud in 3 dimensions.
    
    Args:
        *PCs ::: PointCloud objects
        **kwargs ::: Plotting parameters
            e.g : x_axis='y', y_axis='x' : To map xy axes in reverse 
    Note:
        z is not fixed in proportion to x and y, and it is difficult, and probably unecessary to do so.
    """

    # Get plotting parameters
    colours, labels, limit, x_axis, y_axis, z_axis, bounds, figsize, title = _retrieve_pars(
        ['colours', 'labels', 'limit', 'x_axis', 'y_axis', 'z_axis', 'bounds','figsize', 'title'],
        *PCs, **kwargs)

    # Slice and subsample
    PCs = _slice_and_subsample(bounds, limit, *PCs)
    
    # Initialise 3D plot
    fig = plt.figure(figsize=figsize)
    ax3D = fig.add_subplot(111, projection='3d')
    ax3D.set_aspect("equal") # scale x and y to each other
    
    # Plot each pointcloud sequentially
    for i, PC in enumerate(PCs):
        # Get coordinate arrays in each dimension
        xs =  getattr(PC, 'x')
        ys =  getattr(PC, 'y')
        zs =  getattr(PC, 'z')
        
        ax3D.scatter(xs, ys, zs,
                     s=2, c=colours[i], edgecolors='none', label=labels[i])
    # Set legend and axes labels
    ax3D.legend(loc = 0)
    ax3D.set_xlabel(x_axis)
    ax3D.set_ylabel(y_axis)

    if title:
        ax3d.set_title(title)
    
    plt.show()

def plot_2D(*PCs, **kwargs):
    """Plot PointClouds in two dimensions.
    
    Args:
        *PCs ::: PointCloud objects to plot
        **kwargs ::: Custom plotting parameters
                     e.g. : x_axis='y', y_axis='z' : to plot y vs z
    Plots:
        A single 2D scatter plot of pointcloud
        Multiple pointclouds uniquely coloured, plotted atop one another
    """

    # Get plotting parameters
    colours, labels, limit, x_axis, y_axis, bounds, figsize, title = _retrieve_pars(
        ['colours', 'labels', 'limit', 'x_axis', 'y_axis', 'bounds', 'figsize', 'title'],
        *PCs, **kwargs)
    
    # Slice and subsample
    PCs = _slice_and_subsample(bounds, limit, *PCs)

    fig = plt.figure(figsize=figsize)
    ax = fig.add_subplot(111, projection=None)
    ax.set_aspect("equal") # fix x and y scales in proportion
    
    for i, PC in enumerate(PCs):

        # Get coordinate arrays in each dimension
        xs =  getattr(PC, x_axis)
        ys =  getattr(PC, y_axis)

        ax.scatter(xs, ys,
               s=2, c=colours[i], edgecolors='none', label=labels[i])

    # Set axes labels
    ax.legend()
    ax.set_xlabel(x_axis)
    ax.set_ylabel(y_axis)

    if title:
        ax.set_title(title)

def highlight_bounds(bounds, *PCs, **kwargs):
    """Highlight xy area of provided bounds. Should be extended to all dimensions.
    
    Args:
        bounds ::: region highlight , dict of (min, max) values for each coord x, y and z 
        *PCs ::: arbitrary number of PointCloud objects to plot
        **kwargs ::: custom plotting parameters (e.g. limit, colours, labels)
    Plots:
        Highlight of bounds in entire PointCloud space in xy
    """
    
    # Get plotting parameters
    colours, labels, limit, figsize = _retrieve_pars(
        ['colours', 'labels', 'limit', 'figsize'],
        *PCs, **kwargs)

    # Subsample
    PCs = _slice_and_subsample(None, limit, *PCs)
 
   # Set up plot
    fig = plt.figure(figsize=figsize)
    ax = fig.add_subplot(111)
    ax.set_aspect("equal")
    
    # Plot pointclouds
    for i, PC in enumerate(PCs):
        xs, ys, zs = PC.return_xyz()
        ax.scatter(xs, ys,
        c=colours[i], s=2, edgecolors='none', label=labels[i])
    
    xs, ys = _make_bounds_rectangle(bounds)
    ax.plot(xs, ys, color='yellow')
    
    ax.set_xlabel('x')
    ax.set_ylabel('y')
    
    #plt.show()

def plot_alldims(*PCs, **kwargs):
    """ Plot pointcloud in 3D and all 2D views.
    Args:
        *PCs ::: arbitrary number of PointCloud objects to plot
        **kwargs ::: custom plotting parameters (e.g. limit, colours, labels)
    Plots:
        Supplied pointclouds overlaid in 3D and 2D (xy, xz and xz)
    """

    # Get plotting parameters
    colours, labels, limit, bounds, title = _retrieve_pars(
        ['colours', 'labels', 'limit', 'bounds', 'title'],
        *PCs, **kwargs)

    # Slice and subsample
    PCs = _slice_and_subsample(bounds, limit, *PCs)
    
    # Initialise figure
    fig = plt.figure(figsize = (10, 10))
    
    # Plot 3D
    ax3D = fig.add_subplot(221, projection='3d')
    ax3D.set_aspect("equal") # scale x and y to each other
    
    # Plot each pointcloud sequentially
    for i, PC in enumerate(PCs):
        # Get coordinate arrays in each dimension
        xs =  getattr(PC, 'x')
        ys =  getattr(PC, 'y')
        zs =  getattr(PC, 'z')
        
        ax3D.scatter(xs, ys, zs,
                     s=2, c=colours[i], edgecolors='none', label=labels[i])
    # Set legend and axes labels
    ax3D.legend(loc = 0)
    ax3D.set_xlabel('x')
    ax3D.set_ylabel('y')
    ax3D.set_zlabel('z')
    
    # Plot 2D
    for i, (x_axis, y_axis, view) in enumerate([('x', 'y', 'z'), ('x', 'z', 'y'), ('y', 'z', 'x')]):

        # Determine which axis is missing (superseded, but nice function)
        #flattened = [coord for coord in ['x', 'y', 'z'] if coord not in (x_axis, y_axis)]

        ax = fig.add_subplot(222+i)
        ax.set_aspect("equal") # scale x and y to each other
        # Plot points
        for i, PC in enumerate(PCs):
            # Get data to plot
            xs = getattr(PC, x_axis)
            ys = getattr(PC, y_axis)
            
            ax.scatter(xs, ys,
                       s=2, c=colours[i], edgecolors='none')
            
        # Set axes labels
        ax.set_title("Along %s" % view)
        ax.set_xlabel(x_axis)
        ax.set_ylabel(y_axis)

    if title:
        fig.suptitle(title)
    
    plt.show()

def explore_pointclouds(bounds, *PCs, **kwargs):
    """ Plot position and all-dimension view of PointClouds.
    
    Args:
        bounds ::: region to plot, dict of (min, max) values for each coord x, y and z 
        *PCs ::: arbitrary number of PointCloud objects to plot
        **kwargs ::: custom plotting parameters (e.g. limit, colours, labels)
    Plots:
        Highlight of bounds in entire pointcloud space in xy
        Supplied pointclouds overlaid in xyz, xy, xz and xz
    """
    
    highlight_bounds(bounds, *PCs, **kwargs)
    plot_alldims(bounds=bounds, *PCs, **kwargs)

""" Box Plotting Functions """

def _make_bounds_rectangle(bounds):
    """Return coordinate array of rectangle corners of bounds."""
    A, B, C, D = [(bounds['x'][x], bounds['y'][y]) # build coordinate tuple (x,y)
        for x in (0,1) for y in (0,1)] # from A(11) to D (2,2)
    rectangle = np.array((A, B, D, C, A)).T
    return rectangle


def trace_box_edges(PC): # keep this func global as multiple funcs will use it
    """Generate ... , for plotting boxes."""
    
    A,B,C,D,E,F,G,H = vertices_from_bounds(PC.bounds) # unpack to individual vertices
    # Create continuous line tracing box edges
    box_edges = np.array([A, B, D, C, A, B, F, E, A, C, G, E, F, H, G, C, D, H]).T
    return box_edges

def plot_box(*PCs, **kwargs):
    """Plot pointcloud boxes in 3D.

    Args:
        PCs ::: PointCloud objects
    Plots:
        A 3D plot of the box boundaries of all passed PointCloud objects
    """
    
    # Get plotting parameters
    colours, labels, bounds = _retrieve_pars(
        ['colours', 'labels', 'bounds'],
        *PCs, **kwargs)
    
    # Slice and subsample
    PCs = _slice_and_subsample(bounds, None, *PCs)
    
    # Set up plot
    fig = plt.figure()
    ax3D = fig.add_subplot(111, projection='3d')
    ax3D.set_aspect("equal") # scale x and y to each other
    
    # Plot each PC object
    for i, PC in enumerate(PCs):
        xs, ys, zs = trace_box_edges(PC) # get coordinates of trace
        ax3D.plot(xs, ys, zs, marker = "x", label = i)
    
    ax3D.legend(loc = 0)
    ax3D.set_xlabel('x')
    ax3D.set_ylabel('y')
    ax3D.set_zlabel('z')

def plot_box_2D(*PCs, **kwargs):
    """ Plot pointcloud boxes in 2D.
    
    Args:
        *PCs ::: PointCloud objects
        **kwargs ::: Custom plotting parameters
            e.g. : x_axis='y', y_axis='z' : to plot x vs z
    Plots:
        The given boxes as seen along 'missing' plane.
    """
    
    # Get plotting parameters
    colours, labels, bounds, x_axis, y_axis = _retrieve_pars(
        ['colours', 'labels', 'bounds', 'x_axis', 'y_axis'],
        *PCs, **kwargs)
    
    # Slice and subsample
    PCs = _slice_and_subsample(bounds, None, *PCs)

    # Create references of coord name to array position
    coord_pos = {'x': 0, 'y': 1, 'z': 2}
    
    # Determine which axis is missing
    view = [coord for coord in coord_pos.keys()
            if coord not in (x_axis, y_axis)]
    
    # Set up plot
    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.set_aspect("equal") # scale x and y to each other

    # Plot each PC object
    for i, PC in enumerate(PCs):
        boxtrace = trace_box_edges(PC) # get coordinates of trace
        xs = boxtrace[coord_pos[x_axis]]
        ys = boxtrace[coord_pos[y_axis]]
        ax.plot(xs, ys, marker = "x", label=labels[i])
        
    ax.legend()
    ax.margins(x=0.1, y=0.1) # small buffer around edges of lines
    ax.set_title("Along %s" % view[0])
    ax.set_xlabel(x_axis)
    ax.set_ylabel(y_axis)

def plot_box_alldims(*PCs, **kwargs):
    """Visualise an arbitrary number of pointcloud boxes in all dimensions.
    
    Args:
        *PCs ::: PointCloud objects
        **kwargs ::: Custom plotting parameters
    Plots:
        A 4 panel plot of 1: PC boxes in 3D;
                        2-4: PC boxes in 2D, flattened in z, y and x respectively
    """
    
    # Get plotting parameters
    colours, labels, bounds = _retrieve_pars(
        ['colours', 'labels', 'bounds'],
        *PCs, **kwargs)
    
    # Slice and subsample
    PCs = _slice_and_subsample(bounds, None, *PCs)

    # Position of coordinates in array
    coord_pos = {'x': 0, 'y': 1, 'z': 2}

    # Initialise figure
    fig = plt.figure(figsize = (10, 10))
    
    # Plot 3D
    ax3D = fig.add_subplot(221, projection='3d')
    ax3D.set_aspect("equal") # scale x and y to each other
    for i, PC in enumerate(PCs): # plot each PC object
        xs, ys, zs = trace_box_edges(PC) # get coordinates of trace
        ax3D.plot(xs, ys, zs,
                c=colours[i], marker="x", label=labels[i])

    ax3D.legend(loc=0)
    ax3D.set_xlabel('x')
    ax3D.set_ylabel('y')
    ax3D.set_zlabel('z')

    # Create 2D plot axes objects
    for i, (x_axis, y_axis, view) in enumerate([('x', 'y', 'z'), ('x', 'z', 'y'), ('y', 'z', 'x')]):

        ax = fig.add_subplot(222+i)
        ax.set_aspect("equal") # scale x and y to each other
        for i, PC in enumerate(PCs):
            # Get data to plot
            boxtrace = trace_box_edges(PC) # get coordinates of trace
            xs = boxtrace[coord_pos[x_axis]]
            ys = boxtrace[coord_pos[y_axis]]

            ax.plot(xs, ys,
                    c=colours[i], marker="x", label=labels[i])
            ax.margins(x=0.1, y=0.1) # small buffer around edges of lines

        ax.set_title("Along %s" % view)
        ax.set_xlabel(x_axis)
        ax.set_ylabel(y_axis)

def plot_raster(PC, **kwargs):
    """Plot PointCloud as a raster image of density in 2D.

    Args:
        PC ::: A PointCloud object
        **kwargs ::: Plotting parameters
            e.g. : x_axis='x', y_axis='y' : to plot x,y
    Plots:
        2D image plot of pointcloud density (metre squared resolution)      
    """

    # Get plotting parameters
    x_axis, y_axis, title = _retrieve_pars(
        ['x_axis', 'y_axis', 'title'],
        PC, **kwargs)

    # Get data
    img, (x_offset, y_offset) = PC.rasterise(**kwargs)

    # Plot data
    fig, ax = plt.subplots()
    # Image plot with log-scaled blue-red colorbar
    cax = ax.imshow(img, cmap=plt.cm.coolwarm, norm=LogNorm())
    ax.set_aspect("equal")

    # Update tick labels to represent real space
    xticks = ax.get_xticks()
    yticks = ax.get_yticks()[::-1] # reverse up/down (i.e. matrix to cartesian) 
    # Check no rounding occurs
    if np.any(xticks-xticks.astype(int)) | np.any(yticks-yticks.astype(int)):
        warnings.warn('The Ticks Are Not What They Seem')
    ax.set_xticklabels(xticks.astype(int) + x_offset) # un-offset
    ax.set_yticklabels(yticks.astype(int) + y_offset)
    ax.set_xlabel(x_axis)
    ax.set_ylabel(y_axis)
    
    if title:
        ax.set_title(title)
    
    cbar = fig.colorbar(cax)
    cbar.set_label('Point density ($points\cdot m^{-2}$)')
                        
    plt.show()

def plot_heightmap(PC, **kwargs):
    """Plot PointCloud in 2D with 3rd dimension in colour

    Args:
        PC ::: A PointCloud object
        **kwargs ::: Plotting parameters
            e.g. : x_axis='x', y_axis='y', z_axis='z' : plot xy, using z as colours
                 : cmap='summer' : use the 'summer' pyplot colormap 
   """

    # Get plotting parameters
    limit, x_axis, y_axis, z_axis, bounds, cmap, title = _retrieve_pars(
        ['limit', 'x_axis', 'y_axis', 'z_axis', 'bounds', 'cmap', 'title'],
        PC, **kwargs)

    if cmap is None:
        cmap = 'summer'

    # Slice and subsample
    PC, = _slice_and_subsample(bounds, limit, PC)

    xs = getattr(PC, x_axis)
    ys = getattr(PC, y_axis)
    zs = getattr(PC, z_axis)
    
    # Plot data
    fig, ax = plt.subplots()

    sc = ax.scatter(xs, ys, c=zs,
               cmap=cmap, s=3, linewidths=0)
    
    cb = plt.colorbar(sc)
    cb.set_label('%s (m)' % (z_axis)) 
    
    ax.set_xlabel(x_axis)
    ax.set_ylabel(y_axis)
    
    if title:
        ax.set_title(title)
    
    plt.show()

def plot_histogram(*PCs, **kwargs):
    """ Plot a vertical density histogram of PointCloud values in specified coordinate.

        Args:
            *PCs ::: PointCloud objects
            **kwargs ::: Histogram options and plotting parameters 
                e.g. : z_axis : str name of coordinate to plot. 'z' by default, but can be
                                any attribute containing an array
                     : bin_width : numeric sizes of bins
    """
    
    colours, labels, hist_axis, bounds, bin_width, density, log = _retrieve_pars(
        ['colours', 'labels', 'hist_axis', 'bounds', 'bin_width', 'density', 'log'],
        *PCs, **kwargs)
 
    # Process pointclouds
    PCs = _slice_and_subsample(bounds, None, *PCs)
    
    # Generate histograms
    arrays = [getattr(PC, hist_axis) for PC in PCs]
    bin_edges, hists = histogram(*arrays, bin_width=bin_width, density=density) 
    
    # Find central values of bins
    centre = (bin_edges[:-1] + bin_edges[1:]) / 2

    # Switch for ylabel
    x_label = {True: 'Probability density', False: 'Points'}[density]

    # Plot histogram
    fig, ax = plt.subplots()
    
    for i, hist in enumerate(hists):
        ax.barh(centre, hist,
                height=bin_width, linewidth=0, color=colours[i], alpha=0.5, label=labels[i], log=log)
    
    ax.legend()
    ax.set_xlabel(x_label)
    ax.set_ylabel('%s (m)' % (hist_axis))

