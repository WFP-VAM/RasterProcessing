# RasterProcessing

Allows fast windowing of 3D raster stack in all number of dimensions using memory striding technique. Memory requirement is O(N) vs O(NW) where W is the number of windows. 
Main function is 'sliding_window' which takes 3 arguments: (1) A NUMPY array (2) window size (3) step size for windowing

