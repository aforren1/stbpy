#cython: c_string_type=unicode, c_string_encoding=ascii
#cython: boundscheck=False
#cython: nonecheck=False
#cython: wraparound=False
#cython: infertypes=True
#cython: initializedcheck=False
#cython: cdivision=True
cimport cython
from cpython.mem cimport PyMem_Malloc, PyMem_Free
import os.path as op

cdef extern from *:
    """
    #define STB_RECT_PACK_IMPLEMENTATION
    #define STB_IMAGE_IMPLEMENTATION
    """

# get stb_image for atlas class
cdef extern from '../_stb/stb_image.h':
    int stbi_info(const char* filename, int *x, int *y, int *comp)
    unsigned char* stbi_load(const char* filename, int *x, int *y, int *channels_in_file, int desired_channels)
    const char* stbi_failure_reason()

cdef extern from "../_stb/stb_rect_pack.h":
    struct stbrp_context:
        pass
        
    struct stbrp_node:
        pass

    struct stbrp_rect:
        int id
        int w, h
        int x, y
        int was_packed
    
    int stbrp_pack_rects(stbrp_context *context, stbrp_rect *rects, int num_rects)
    void stbrp_init_target(stbrp_context *context, int width, int height, stbrp_node *nodes, int num_nodes)
    void stbrp_setup_heuristic(stbrp_context *context, int heuristic)

cpdef enum Heuristic:
    DEFAULT = 0
    BL_SORTHEIGHT = DEFAULT
    BF_SORTHEIGHT

# 
@cython.no_gc_clear
cdef class AtlasPacker:
    cdef stbrp_context* context
    cdef int width
    cdef int height
    cdef int num_nodes
    cdef int heuristic
    # 
    cdef int num_keys
    cdef stbrp_node* nodes

    def __init__(self, width, height, heuristic=Heuristic.DEFAULT):
        self.width = width
        self.height = height
        self.num_nodes = width
        self.heuristic = heuristic
        self.keys = []
        self.num_keys = 0
    
    def pack(self, images):
        # take list of image paths
        # return nothing for now (or just warning/err)-- on request, give memoryview & dict
        if self.context == NULL:
            self.nodes = <unsigned char*> PyMem_Malloc(self.num_nodes * sizeof(stbrp_node))
            stbrp_init_target(self.context, self.width, self.height, self.nodes, self.num_nodes)
            stbrp_setup_heuristic(self.context, self.heuristic)
        
        # step 1: read image attributes
        potential_keys = []
        cdef int counter = 0
        cdef stbrp_rect* rects
        cdef int x, y, channels_in_file
        cdef stbrp_rect temp_rect
        try:
            rects = <stbrp_rect*> PyMem_Malloc(len(images) * sizeof(stbrp_rect))
            for im in images:
                if not stbi_info(im, &x, &y, &channels_in_file):
                    raise RuntimeError('Image property query failed. %s' % stbi_failure_reason())
                potential_keys.append(op.splitext(op.basename(im))[0])
                temp_rect = &rects[counter]
                temp_rect.id = counter
                temp_rect.w = x
                temp_rect.h = y
                counter += 1
            
        # all done, free rects
        finally:
            PyMem_Free(rects)

    
    def __dealloc__(self):
        PyMem_Free(self.nodes)
        


# thinking a little about what the python API would look like
# returning the packed atlas would be nice, plus coordinates of each image?
# passing in images by reference?
# R(G)(B)(A)?

# returns RGBA texture + dict
# cpdef (unsigned char[:, :, :], dict) pack(list filenames, int width, int height):



# cdef class RectPacker:

#     cdef stbrp_context* context
#     cdef int width
#     cdef int height
#     cdef int num_nodes # malloc max number of temporary nodes

#     def __init__(self, width, height, heuristic=Heuristic.DEFAULT):
#         # if width < num_nodes:
#         #     pass # raise warning or something
#         self.width = width
#         self.height = height
#         self.num_nodes = width # for simplicity for now
    
#     # take a list of dicts containing {'id': a, 'w': b, 'h': c}
#     # return list of dicts containing {'id': a, 'x': d, 'y': e}
#     # TODO: leaving incomplete for now, b/c I'm not sure whether I'd even use this class yet
#     def pack_rects(self):
#         pass


