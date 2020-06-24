# cython: c_string_type=unicode, c_string_encoding=ascii
import numpy as np
cimport numpy as np

cdef extern from "defs.h":
    pass

cdef extern from "../_stb/stb_image.h":
    unsigned char* stbi_load(const char* filename, int *x, int *y, int *channels_in_file, int desired_channels)
    unsigned char* stbi_load_from_memory(const unsigned char* buffer, int length, 
                                         int *x, int *y, int *channels_in_file, int desired_channels)
    const char* stbi_failure_reason()

cdef extern from "../_stb/stb_image_resize.h":
    int stbir_resize_uint8(const unsigned char* input_pixels, int input_w , int input_h , int input_stride_in_bytes,
                                           unsigned char* output_pixels, int output_w, int output_h, 
                                           int output_stride_in_bytes, int num_channels)

cdef extern from "numpy/arrayobject.h":
    void PyArray_ENABLEFLAGS(np.ndarray arr, int flags)

cpdef load(const char* filename):
    cdef int x, y, channels_in_file, size
    cdef unsigned char* data = stbi_load(filename, &x, &y, &channels_in_file, 0)
    if data is NULL:
        raise RuntimeError('File failed to load. %s' % stbi_failure_reason())
    cdef np.ndarray[np.uint8_t, ndim=3] arr
    cdef np.npy_intp *dims = [x, y, channels_in_file]
    arr = np.PyArray_SimpleNewFromData(3, dims, np.NPY_UINT8, data)
    PyArray_ENABLEFLAGS(arr, np.NPY_ARRAY_OWNDATA)
    return arr

cpdef load_from_memory(const unsigned char[:] buffer):
    cdef int x, y, channels_in_mem, size
    cdef unsigned char* data = stbi_load_from_memory(&buffer[0], buffer.shape[0], &x, &y, &channels_in_mem, 0)
    if data is NULL:
        raise RuntimeError('Memory failed to load. %s' % stbi_failure_reason())
    cdef np.ndarray[np.uint8_t, ndim=3] arr
    cdef np.npy_intp *dims = [x, y, channels_in_mem]
    arr = np.PyArray_SimpleNewFromData(3, dims, np.NPY_UINT8, data)
    PyArray_ENABLEFLAGS(arr, np.NPY_ARRAY_OWNDATA)
    return arr

cpdef resize(const unsigned char[:, :, :] image, int width, int height):
    cdef unsigned char* data
    cdef int result = stbir_resize_uint8(&image[0, 0, 0], image.shape[0], image.shape[1], 0,
                                         data, width, height, 0, image.shape[2])
    if result == 0:
        raise RuntimeError('Error resizing the image.')
    cdef np.ndarray[np.uint8_t, ndim=3] arr
    cdef np.npy_intp *dims = [width, height, image.shape[2]]
    arr = np.PyArray_SimpleNewFromData(3, dims, np.NPY_UINT8, data)
    PyArray_ENABLEFLAGS(arr, np.NPY_ARRAY_OWNDATA)
    return arr
