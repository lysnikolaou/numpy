/* stringlib: find/index implementation */

#ifndef STRINGLIB_FASTSEARCH_H
#error must include "stringlib/fastsearch.h" before including this module
#endif

#ifndef STRINGLIB_LENGTH_H
#error must include "stringlib/length.h" before including this module
#endif

/* helper macro to fixup start/end slice values */
#define ADJUST_INDICES(start, end, len)         \
    if (end > len)                              \
        end = len;                              \
    else if (end < 0) {                         \
        end += len;                             \
        if (end < 0)                            \
            end = 0;                            \
    }                                           \
    if (start < 0) {                            \
        start += len;                           \
        if (start < 0)                          \
            start = 0;                          \
    }

static inline int
STRINGLIB(find)(PyArrayIterObject **in_iters, int numiters, PyArrayIterObject *out_iter)
{
    const STRINGLIB_CHAR *buf1, *buf2;
    Py_ssize_t pos, len1, len2, start, end;

    buf1 = (STRINGLIB_CHAR *) in_iters[0]->dataptr;
    buf2 = (STRINGLIB_CHAR *) in_iters[1]->dataptr;
    if (numiters > 2) {
        start = *((Py_ssize_t *) in_iters[2]->dataptr);
    } else {
        start = 0;
    }
    if (numiters > 3) {
        end = *((Py_ssize_t *) in_iters[3]->dataptr);
    } else {
        end = PY_SSIZE_T_MAX;
    }

    len1 = STRINGLIB(get_length)(in_iters[0]);
    len2 = STRINGLIB(get_length)(in_iters[1]);
    ADJUST_INDICES(start, end, len1);

    if (end - start < len2) {
        pos = -1;
        memcpy(out_iter->dataptr, &pos, sizeof(npy_int64));
        return 1;
    }

    if (len2 == 0) {
        memcpy(out_iter->dataptr, &start, sizeof(npy_int64));
        return 1;
    }

    if (len2 == 1) {
        pos = STRINGLIB(find_char)(buf1 + start, end - start, *buf2);
        if (pos >= 0) {
            pos += start;
        }
        memcpy(out_iter->dataptr, &pos, sizeof(npy_int64));
        return 1;
    }

    pos = STRINGLIB(fastsearch)(buf1, end - start, buf2, len2, -1, FAST_SEARCH);
    if (pos >= 0) {
        pos += start;
    }
    memcpy(out_iter->dataptr, &pos, sizeof(npy_int64));
    return 1;
}

static inline int
STRINGLIB(rfind)(PyArrayIterObject **in_iters, int numiters, PyArrayIterObject *out_iter)
{
    const STRINGLIB_CHAR *buf1, *buf2;
    Py_ssize_t pos, len1, len2, start, end;

    buf1 = (STRINGLIB_CHAR *) in_iters[0]->dataptr;
    buf2 = (STRINGLIB_CHAR *) in_iters[1]->dataptr;
    if (numiters > 2) {
        start = *((Py_ssize_t *) in_iters[2]->dataptr);
    } else {
        start = 0;
    }
    if (numiters > 3) {
        end = *((Py_ssize_t *) in_iters[3]->dataptr);
    } else {
        end = PY_SSIZE_T_MAX;
    }

    len1 = STRINGLIB(get_length)(in_iters[0]);
    len2 = STRINGLIB(get_length)(in_iters[1]);
    ADJUST_INDICES(start, end, len1);

    if (end - start < len2) {
        pos = -1;
        memcpy(out_iter->dataptr, &pos, sizeof(npy_int64));
        return 1;
    }

    if (len2 == 0) {
        memcpy(out_iter->dataptr, &end, sizeof(npy_int64));
        return 1;
    }

    if (len2 == 1) {
        pos = STRINGLIB(rfind_char)(buf1 + start, end - start, *buf2);
        if (pos >= 0) {
            pos += start;
        }
        memcpy(out_iter->dataptr, &pos, sizeof(npy_int64));
        return 1;
    }

    pos = STRINGLIB(fastsearch)(buf1, end - start, buf2, len2, -1, FAST_RSEARCH);
    if (pos >= 0) {
        pos += start;
    }
    memcpy(out_iter->dataptr, &pos, sizeof(npy_int64));
    return 1;
}
