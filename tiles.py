"""tiles
Used for dealing with tiled datasets
"""

import os
from glob import glob
import laspy

class Tiles(object):
    """ A container for tiled .las filepaths"""
    def __init__(self, files):
        """ Initialise from a directory or list of filepaths.
        
        Args:
            files ::: str directory path or list of str filepaths        
        """
        
        if isinstance(files, list):
            fpaths = files
        elif os.path.isdir(files):
            fpaths = glob(files + '*.las')
        else:
            raise TypeError, "`files` has to be str directory path or list of file paths"
        
        self.tiles = self._extract_tiles_bounds(fpaths)
        
    def _extract_tiles_bounds(self, fpaths):
        """ Store the bounds for each tile (a .las file) in fpaths in a dictionary.

        Args:
            fpaths ::: list of str filepaths of .las files, or
                    :: str path to directory containing .las files
        Returns:
            tiles_bounds ::: dict of bounds dicts for each file
                            i.e. `{fpath: bounds}` where `bounds = {'x': (min, max), ...}`.
        """

        tiles_bounds = {} # to store bounds for each tile
        # Iterate over all .las files, finding bounds (takes a min)
        for fname in fpaths:
            with laspy.file.File(fname) as lasfile:
                bounds = dict(zip(('x', 'y', 'z'),
                              zip(lasfile.header.min, lasfile.header.max)))
                tiles_bounds[fname] = bounds

        return tiles_bounds

    def select_tiles(self, selection_bounds):
        """ Return the subset of tiles which contain data in specified area.
        
        Args:
            area_bounds ::: dict of (min, max) for 'x' and 'y' specifying area to select
        Returns:
            tiles ::: dict of {filepaths: bounds} of .las files whose area overlaps with that of area_bounds  
        """
        
        # is any part of (min, max) tuple c1 in range of c2?
        c_overlap = lambda c1, c2: any([c >= c2[0] and c <= c2[1] for c in c1])
        # is there overlap of b1 with all of the dimensions in b2?
        bounds_overlap = lambda b1, b2: all([c_overlap(b1[coord], b2[coord]) for coord in b2.keys()])

        # Filter dict to fpaths with overlapping area
        tiles = {fpath: bounds for fpath, bounds in self.tiles.iteritems()
                  if bounds_overlap(bounds, selection_bounds)}
        return tiles