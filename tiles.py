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
        
        if isinstance(files, list): # from list of files
            fpaths = files
            tiles = self._extract_tiles_bounds(fpaths)
        elif isinstance(files, dict): # from slicing
            tiles = files
        elif os.path.isdir(files): # from directory
            fpaths = glob(files + '*.las')
            tiles = self._extract_tiles_bounds(fpaths)
        else:
            raise TypeError, "`files` has to be str directory path or list of file paths"
        
        self.tiles = tiles
        self._set_bounds()
        ## Find bounds of tiles
    
    def __getitem__(self, selection):
        """ Slice by bounds dict to return new object with only matching tiles."""
        
        tiles = {fpath: bounds for fpath, bounds in self.tiles.iteritems()
                  if self._bounds_overlap(bounds, selection)}
        if len(tiles):
            return Tiles(tiles)
    
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
    
    def _set_bounds(self):
        """ Determine and store the bounds of combined area covered by tiles."""
        
        allvals = {'x': [], 'y': [], 'z': []} # to store all bounds
        # Extract each coordinate from bounds of all tiles
        for tile in self.tiles.itervalues():
            for c, val in tile.iteritems():
                allvals[c].extend(val)
        
        # Overall minimum and maximum
        self.bounds = {c: (min(vals), max(vals)) for c, vals in allvals.iteritems()}
    
    def select_tiles(self, selection_bounds):
        """ Return list of tiles which contain data in specified area.
        
        Args:
            area_bounds ::: dict of (min, max) for 'x' and 'y' specifying area to select
        Returns:
            fpaths ::: list of .las filepaths whose pointcloud area overlaps with that of area_bounds  
        """
        
        # Filter dict to fpaths with overlapping area
        fpaths = [fpath for fpath, bounds in self.tiles.iteritems()
                  if self._bounds_overlap(bounds, selection_bounds)]
        return fpaths
    
    def _bounds_overlap(self, b1, b2):
        """ Determine whether bounds dicts b1 and b2 overlap in all dimensions."""
        return all([self._c_overlap(b1[coord], b2[coord]) for coord in b2.keys()])
    
    def _c_overlap(self, c1, c2):
        """ Determine whether (min, max) coordinate bounds c1 and c2 overlap."""
        return any([not (c1[1] < c2[0]) ^ (c1[0] > c2[1]) for c in c1])
        # return any([c >= c2[0] and c <= c2[1] for c in c1])