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
STRINGLIB(find)(PyArrayIterObject **in_iters, int numiters, PyArrayIterObject *out_iter)/*const STRINGLIB_CHAR* str, Py_ssize_t str_len,
                                                                            const STRINGLIB_CHAR* sub, Py_ssize_t sub_len,
                                                                            Py_ssize_t offset) */
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

/*static inline Py_ssize_t
STRINGLIB(rfind)(const STRINGLIB_CHAR* str, Py_ssize_t str_len,
                const STRINGLIB_CHAR* sub, Py_ssize_t sub_len,
                Py_ssize_t offset)
{
    Py_ssize_t pos;

    assert(str_len >= 0);
    if (sub_len == 0)
        return str_len + offset;

    pos = STRINGLIB(fastsearch)(str, str_len, sub, sub_len, -1, FAST_RSEARCH);

    if (pos >= 0)
        pos += offset;

    return pos;
}*/
