# cython: c_string_type=unicode, c_string_encoding=ascii
cimport numpy as np
import numpy as np
from libc.stdlib cimport malloc, free
import os.path as op

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

cdef extern from '../_stb/stb_image_write.h':
     int stbi_write_png(const char* filename, int w, int h, int comp, const void* data, int stride_in_bytes)
     int stbi_write_bmp(const char* filename, int w, int h, int comp, const void* data)
     int stbi_write_tga(const char* filename, int w, int h, int comp, const void* data)
     int stbi_write_jpg(const char* filename, int w, int h, int comp, const void* data, int quality)
     # int stbi_write_hdr(const char* filename, int w, int h, int comp, const float* data)
     unsigned char* stbi_write_png_to_mem(const unsigned char* pixels, int stride_bytes, int x, int y, int n, int* out_len)

    
cdef extern from "numpy/arrayobject.h":
    void PyArray_ENABLEFLAGS(np.ndarray arr, int flags)

cpdef np.ndarray[np.uint8_t, ndim=3] load(const char* filename):
    cdef int x, y, channels_in_file, size
    cdef unsigned char* data = stbi_load(filename, &x, &y, &channels_in_file, 0)
    if data is NULL:
        raise RuntimeError('File failed to load. %s' % stbi_failure_reason())
    cdef np.ndarray[np.uint8_t, ndim=3] arr
    cdef np.npy_intp *dims = [x, y, channels_in_file]
    arr = np.PyArray_SimpleNewFromData(3, dims, np.NPY_UINT8, data)
    PyArray_ENABLEFLAGS(arr, np.NPY_ARRAY_OWNDATA)
    return arr

cpdef np.ndarray[np.uint8_t, ndim=3] load_from_memory(const unsigned char[:] buffer):
    cdef int x, y, channels_in_mem, size
    cdef unsigned char* data = stbi_load_from_memory(&buffer[0], buffer.shape[0], &x, &y, &channels_in_mem, 0)
    if data is NULL:
        raise RuntimeError('Memory failed to load. %s' % stbi_failure_reason())
    cdef np.ndarray[np.uint8_t, ndim=3] arr
    cdef np.npy_intp *dims = [x, y, channels_in_mem]
    arr = np.PyArray_SimpleNewFromData(3, dims, np.NPY_UINT8, data)
    PyArray_ENABLEFLAGS(arr, np.NPY_ARRAY_OWNDATA)
    return arr

cpdef np.ndarray[np.uint8_t, ndim=3] resize(const unsigned char[:, :, :] image, int width, int height):
    cdef unsigned char* data = <unsigned char*> malloc(width * height * image.shape[2])
    if not data:
        raise MemoryError()
    cdef int result = stbir_resize_uint8(&image[0, 0, 0], image.shape[0], image.shape[1], 0,
                                         data, width, height, 0, image.shape[2])
    if result == 0:
        free(data)
        raise RuntimeError('Error resizing the image.')
    cdef np.ndarray[np.uint8_t, ndim=3] arr
    cdef np.npy_intp *dims = [width, height, image.shape[2]]
    arr = np.PyArray_SimpleNewFromData(3, dims, np.NPY_UINT8, data)
    PyArray_ENABLEFLAGS(arr, np.NPY_ARRAY_OWNDATA)
    return arr

cpdef void write(const char* filename, const unsigned char[:, :, :] image, int quality = 100):
    ext = op.splitext(filename)[1]
    cdef int res
    if ext == '.png':
        res = stbi_write_png(filename, image.shape[0], image.shape[1], image.shape[2], &image[0, 0, 0], 0)
    elif ext == '.jpg':
        res = stbi_write_jpg(filename, image.shape[0], image.shape[1], image.shape[2], &image[0, 0, 0], quality)
    elif ext == '.bmp':
        res = stbi_write_bmp(filename, image.shape[0], image.shape[1], image.shape[2], &image[0, 0, 0])
    elif ext == '.tga':
        res = stbi_write_tga(filename, image.shape[0], image.shape[1], image.shape[2], &image[0, 0, 0])
    else:
        raise ValueError('Unknown file extension.')
    if res == 0:
        raise RuntimeError('Write failed.')


cpdef np.ndarray[np.uint8_t, ndim=1] write_png_to_memory(const unsigned char[:, :, :] image):
    cdef int out_len
    cdef unsigned char* data = stbi_write_png_to_mem(&image[0, 0, 0], 0, image.shape[0], image.shape[1], image.shape[2], &out_len)
    if data is NULL:
        raise RuntimeError('Image failed to write.')
    cdef np.ndarray[np.uint8_t, ndim=1] arr
    cdef np.npy_intp *dims = [out_len]
    arr = np.PyArray_SimpleNewFromData(1, dims, np.NPY_UINT8, data)
    PyArray_ENABLEFLAGS(arr, np.NPY_ARRAY_OWNDATA)
    return arr
