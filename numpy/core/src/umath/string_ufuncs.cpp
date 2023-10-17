#include <Python.h>
#include <string.h>

#define NPY_NO_DEPRECATED_API NPY_API_VERSION
#define _MULTIARRAYMODULE
#define _UMATHMODULE

#include "numpy/ndarraytypes.h"
#include "numpy/npy_math.h"
#include "numpy/ufuncobject.h"

#include "numpyos.h"
#include "dispatching.h"
#include "dtypemeta.h"
#include "common_dtype.h"
#include "convert_datatype.h"

#include "string_ufuncs.h"
#include "string_fastsearch.h"


template <typename character>
static inline int
character_cmp(character a, character b)
{
    if (a == b) {
        return 0;
    }
    else if (a < b) {
        return -1;
    }
    else {
        return 1;
    }
}

template<typename character>
static inline int
string_rstrip(const character *str, int elsize)
{
    /*
     * Ignore/"trim" trailing whitespace (and 0s).  Note that this function
     * does not support unicode whitespace (and never has).
     */
    while (elsize > 0) {
        character c = str[elsize-1];
        if (c != (character)0 && !NumPyOS_ascii_isspace(c)) {
            break;
        }
        elsize--;
    }
    return elsize;
}

template<typename character>
static inline int
get_length(const character *str, int elsize)
{
    const character *d = str + elsize - 1;
    while (d >= str && *d == '\0') {
        d--;
    }
    return d - str + 1;
}

template <typename character>
static inline Py_ssize_t
findslice(const character* str, Py_ssize_t str_len,
           const character* sub, Py_ssize_t sub_len,
           Py_ssize_t offset)
{
    Py_ssize_t pos;

    if (sub_len == 0) {
        return offset;
    }

    pos = fastsearch<character>(str, str_len, sub, sub_len, -1, FAST_SEARCH);

    if (pos >= 0) {
        pos += offset;
    }

    return pos;
}

template <typename character>
static inline Py_ssize_t
rfindslice(const character* str, Py_ssize_t str_len,
           const character* sub, Py_ssize_t sub_len,
           Py_ssize_t offset)
{
    Py_ssize_t pos;

    if (sub_len == 0) {
        return str_len + offset;
    }

    pos = fastsearch<character>(str, str_len, sub, sub_len, -1, FAST_RSEARCH);

    if (pos >= 0) {
        pos += offset;
    }

    return pos;
}

template <typename character>
static inline Py_ssize_t
countslice(const character* str, Py_ssize_t str_len,
           const character* sub, Py_ssize_t sub_len,
           Py_ssize_t maxcount)
{
    Py_ssize_t count;

    if (str_len < 0) {
        return 0; /* start > len(str) */
    }
    if (sub_len == 0) {
        return (str_len < maxcount) ? str_len + 1 : maxcount;
    }

    count = fastsearch<character>(str, str_len, sub, sub_len, maxcount, FAST_COUNT);

    if (count < 0) {
        return 0; /* no match */
    }

    return count;
}

/*
 * Compare two strings of different length.  Note that either string may be
 * zero padded (trailing zeros are ignored in other words, the shorter word
 * is always padded with zeros).
 */
template <bool rstrip, typename character>
static inline int
string_cmp(const character *str1, int elsize1, const character *str2, int elsize2)
{
    int len1 = elsize1, len2 = elsize2;
    if (rstrip) {
        len1 = string_rstrip(str1, elsize1);
        len2 = string_rstrip(str2, elsize2);
    }

    int n = PyArray_MIN(len1, len2);

    if (sizeof(character) == 1) {
        /*
         * TODO: `memcmp` makes things 2x faster for longer words that match
         *       exactly, but at least 2x slower for short or mismatching ones.
         */
        int cmp = memcmp(str1, str2, n);
        if (cmp != 0) {
            return cmp;
        }
        str1 += n;
        str2 += n;
    }
    else {
        for (int i = 0; i < n; i++) {
            int cmp = character_cmp(*str1, *str2);
            if (cmp != 0) {
                return cmp;
            }
            str1++;
            str2++;
        }
    }
    if (len1 > len2) {
        for (int i = n; i < len1; i++) {
            int cmp = character_cmp(*str1, (character)0);
            if (cmp != 0) {
                return cmp;
            }
            str1++;
        }
    }
    else if (len2 > len1) {
        for (int i = n; i < len2; i++) {
            int cmp = character_cmp((character)0, *str2);
            if (cmp != 0) {
                return cmp;
            }
            str2++;
        }
    }
    return 0;
}

template <typename character>
static inline npy_bool
string_isalpha(const character *str, int elsize)
{
    int len = get_length(str, elsize);

    if (len == 0) {
        return (npy_bool) 0;
    }

    for (int i = 0; i < len; i++) {
        npy_bool isalpha = (npy_bool) NumPyOS_ascii_isalpha(*str);
        if (!isalpha) {
            return isalpha;
        }
        str++;
    }
    return (npy_bool) 1;
}

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

template <typename character>
static inline npy_long
string_find(character *str1, int elsize1, character *str2, int elsize2, npy_long start, npy_long end)
{
    int len1, len2;
    npy_long result;

    len1 = get_length<character>(str1, elsize1);
    len2 = get_length<character>(str2, elsize2);

    ADJUST_INDICES(start, end, len1);
    if (end - start < len2) {
        return (npy_long) -1;
    }

    if (len2 == 1) {
        character ch = *str2;
        result = findchar<character>(str1 + start, end - start, ch);
        if (result == -1) {
            return -1;
        } else {
            return start + result;
        }
    }

    return findslice<character>(str1 + start, end - start, str2, len2, start);
}

template <typename character>
static inline npy_long
string_rfind(character *str1, int elsize1, character *str2, int elsize2, npy_long start, npy_long end)
{
    int len1, len2;
    npy_long result;

    len1 = get_length<character>(str1, elsize1);
    len2 = get_length<character>(str2, elsize2);

    ADJUST_INDICES(start, end, len1);
    if (end - start < len2) {
        return (npy_long) -1;
    }

    if (len2 == 1) {
        character ch = *str2;
        result = rfindchar<character>(str1 + start, end - start, ch);
        if (result == -1) {
            return -1;
        } else {
            return start + result;
        }
    }

    return rfindslice<character>(str1 + start, end - start, str2, len2, start);
}

template <typename character>
static inline npy_long
string_count(character *str1, int elsize1, character *str2, int elsize2, npy_long start, npy_long end)
{
    int len1, len2;

    len1 = get_length<character>(str1, elsize1);
    len2 = get_length<character>(str2, elsize2);

    ADJUST_INDICES(start, end, len1);
    if (end - start < len2) {
        return (npy_long) 0;
    }

    return countslice<character>(str1 + start, end - start, str2, len2, PY_SSIZE_T_MAX);
}

template <typename character>
static inline void
string_replace(character *str1, int elsize1, character *str2, int elsize2, character *str3, int elsize3,
               npy_long count, character *out, int outsize)
{
    memset(out, 'h', outsize * sizeof(character));
}


/*
 * Helper for templating, avoids warnings about uncovered switch paths.
 */
enum class COMP {
    EQ, NE, LT, LE, GT, GE,
};

static char const *
comp_name(COMP comp) {
    switch(comp) {
        case COMP::EQ: return "equal";
        case COMP::NE: return "not_equal";
        case COMP::LT: return "less";
        case COMP::LE: return "less_equal";
        case COMP::GT: return "greater";
        case COMP::GE: return "greater_equal";
        default:
            assert(0);
            return nullptr;
    }
}


template <bool rstrip, COMP comp, typename character>
static int
string_comparison_loop(PyArrayMethod_Context *context,
        char *const data[], npy_intp const dimensions[],
        npy_intp const strides[], NpyAuxData *NPY_UNUSED(auxdata))
{
    /*
     * Note, fetching `elsize` from the descriptor is OK even without the GIL,
     * however it may be that this should be moved into `auxdata` eventually,
     * which may also be slightly faster/cleaner (but more involved).
     */
    int elsize1 = context->descriptors[0]->elsize / sizeof(character);
    int elsize2 = context->descriptors[1]->elsize / sizeof(character);

    char *in1 = data[0];
    char *in2 = data[1];
    char *out = data[2];

    npy_intp N = dimensions[0];

    while (N--) {
        int cmp = string_cmp<rstrip>(
                (character *)in1, elsize1, (character *)in2, elsize2);
        npy_bool res;
        switch (comp) {
            case COMP::EQ:
                res = cmp == 0;
                break;
            case COMP::NE:
                res = cmp != 0;
                break;
            case COMP::LT:
                res = cmp < 0;
                break;
            case COMP::LE:
                res = cmp <= 0;
                break;
            case COMP::GT:
                res = cmp > 0;
                break;
            case COMP::GE:
                res = cmp >= 0;
                break;
        }
        *(npy_bool *)out = res;

        in1 += strides[0];
        in2 += strides[1];
        out += strides[2];
    }
    return 0;
}

template <typename character>
static int
string_str_len_loop(PyArrayMethod_Context *context,
        char *const data[], npy_intp const dimensions[],
        npy_intp const strides[], NpyAuxData *NPY_UNUSED(auxdata))
{
    int elsize = context->descriptors[0]->elsize / sizeof(character);

    char *in = data[0];
    char *out = data[1];

    npy_intp N = dimensions[0];

    while (N--) {
        npy_bool res = get_length<character>((character *) in, elsize);
        *(npy_long *)out = res;

        in += strides[0];
        out += strides[1];
    }

    return 0;
}

template <typename character>
static int
string_isalpha_loop(PyArrayMethod_Context *context,
        char *const data[], npy_intp const dimensions[],
        npy_intp const strides[], NpyAuxData *NPY_UNUSED(auxdata))
{
    /*
     * Note, fetching `elsize` from the descriptor is OK even without the GIL,
     * however it may be that this should be moved into `auxdata` eventually,
     * which may also be slightly faster/cleaner (but more involved).
     */
    int elsize = context->descriptors[0]->elsize / sizeof(character);

    char *in = data[0];
    char *out = data[1];

    npy_intp N = dimensions[0];

    while (N--) {
        npy_bool res = string_isalpha<character>((character *) in, elsize);
        *(npy_bool *)out = res;

        in += strides[0];
        out += strides[1];
    }

    return 0;
}


template<typename character>
static int
string_find_loop(PyArrayMethod_Context *context,
        char *const data[], npy_intp const dimensions[],
        npy_intp const strides[], NpyAuxData *NPY_UNUSED(auxdata))
{
    int elsize1 = context->descriptors[0]->elsize / sizeof(character);
    int elsize2 = context->descriptors[1]->elsize / sizeof(character);

    char *in1 = data[0];
    char *in2 = data[1];
    char *in3 = data[2];
    char *in4 = data[3];
    char *out = data[4];

    npy_intp N = dimensions[0];

    while (N--) {
        npy_long idx = string_find<character>((character *) in1, elsize1, (character *) in2, elsize2,
                                              *(npy_long *)in3, *(npy_long *)in4);
        *(npy_long *)out = idx;

        in1 += strides[0];
        in2 += strides[1];
        in3 += strides[2];
        in4 += strides[3];
        out += strides[4];
    }
    return 0;
}

template<typename character>
static int
string_rfind_loop(PyArrayMethod_Context *context,
        char *const data[], npy_intp const dimensions[],
        npy_intp const strides[], NpyAuxData *NPY_UNUSED(auxdata))
{
    int elsize1 = context->descriptors[0]->elsize / sizeof(character);
    int elsize2 = context->descriptors[1]->elsize / sizeof(character);

    char *in1 = data[0];
    char *in2 = data[1];
    char *in3 = data[2];
    char *in4 = data[3];
    char *out = data[4];

    npy_intp N = dimensions[0];

    while (N--) {
        npy_long idx = string_rfind<character>((character *) in1, elsize1, (character *) in2, elsize2,
                                              *(npy_long *)in3, *(npy_long *)in4);
        *(npy_long *)out = idx;

        in1 += strides[0];
        in2 += strides[1];
        in3 += strides[2];
        in4 += strides[3];
        out += strides[4];
    }
    return 0;
}

template<typename character>
static int
string_count_loop(PyArrayMethod_Context *context,
        char *const data[], npy_intp const dimensions[],
        npy_intp const strides[], NpyAuxData *NPY_UNUSED(auxdata))
{
    int elsize1 = context->descriptors[0]->elsize / sizeof(character);
    int elsize2 = context->descriptors[1]->elsize / sizeof(character);

    char *in1 = data[0];
    char *in2 = data[1];
    char *in3 = data[2];
    char *in4 = data[3];
    char *out = data[4];

    npy_intp N = dimensions[0];

    while (N--) {
        npy_long idx = string_count<character>((character *) in1, elsize1, (character *) in2, elsize2,
                                              *(npy_long *)in3, *(npy_long *)in4);
        *(npy_long *)out = idx;

        in1 += strides[0];
        in2 += strides[1];
        in3 += strides[2];
        in4 += strides[3];
        out += strides[4];
    }
    return 0;
}

template<typename character>
static int
string_replace_loop(PyArrayMethod_Context *context,
        char *const data[], npy_intp const dimensions[],
        npy_intp const strides[], NpyAuxData *NPY_UNUSED(auxdata))
{
    int elsize1 = context->descriptors[0]->elsize / sizeof(character);
    int elsize2 = context->descriptors[1]->elsize / sizeof(character);
    int elsize3 = context->descriptors[2]->elsize / sizeof(character);
    int outsize = context->descriptors[4]->elsize / sizeof(character);

    char *in1 = data[0];
    char *in2 = data[1];
    char *in3 = data[2];
    char *in4 = data[3];
    char *out = data[4];

    npy_intp N = dimensions[0];

    while (N--) {
        string_replace<character>((character *) in1, elsize1, (character *) in2, elsize2,
                                  (character *) in3, elsize3, *(npy_long *)in4,
                                  (character *) out, outsize);

        in1 += strides[0];
        in2 += strides[1];
        in3 += strides[2];
        in4 += strides[3];
        out += strides[4];
    }
    return 0;
}

/*
 * Machinery to add the string loops to the existing ufuncs.
 */

/*
 * This function replaces the strided loop with the passed in one,
 * and registers it with the given ufunc.
 */
static int
add_loop(PyObject *umath, const char *ufunc_name,
         PyArrayMethod_Spec *spec, PyArrayMethod_StridedLoop *loop)
{
    PyObject *name = PyUnicode_FromString(ufunc_name);
    if (name == nullptr) {
        return -1;
    }
    PyObject *ufunc = PyObject_GetItem(umath, name);
    Py_DECREF(name);
    if (ufunc == nullptr) {
        return -1;
    }
    spec->slots[0].pfunc = (void *)loop;

    int res = PyUFunc_AddLoopFromSpec(ufunc, spec);
    Py_DECREF(ufunc);
    return res;
}


template<bool rstrip, typename character, COMP...>
struct add_loops;

template<bool rstrip, typename character>
struct add_loops<rstrip, character> {
    int operator()(PyObject*, PyArrayMethod_Spec*) {
        return 0;
    }
};

template<bool rstrip, typename character, COMP comp, COMP... comps>
struct add_loops<rstrip, character, comp, comps...> {
    int operator()(PyObject* umath, PyArrayMethod_Spec* spec) {
        PyArrayMethod_StridedLoop* loop = string_comparison_loop<rstrip, comp, character>;

        if (add_loop(umath, comp_name(comp), spec, loop) < 0) {
            return -1;
        }
        else {
            return add_loops<rstrip, character, comps...>()(umath, spec);
        }
    }
};


static int
init_comparison(PyObject *umath)
{
    int res = -1;
    /* NOTE: This should receive global symbols? */
    PyArray_DTypeMeta *String = PyArray_DTypeFromTypeNum(NPY_STRING);
    PyArray_DTypeMeta *Unicode = PyArray_DTypeFromTypeNum(NPY_UNICODE);
    PyArray_DTypeMeta *Bool = PyArray_DTypeFromTypeNum(NPY_BOOL);

    /* We start with the string loops: */
    PyArray_DTypeMeta *dtypes[] = {String, String, Bool};
    /*
     * We only have one loop right now, the strided one.  The default type
     * resolver ensures native byte order/canonical representation.
     */
    PyType_Slot slots[] = {
        {NPY_METH_strided_loop, nullptr},
        {0, nullptr}
    };

    PyArrayMethod_Spec spec = {};
    spec.name = "templated_string_comparison";
    spec.nin = 2;
    spec.nout = 1;
    spec.dtypes = dtypes;
    spec.slots = slots;
    spec.flags = NPY_METH_NO_FLOATINGPOINT_ERRORS;

    /* All String loops */
    using string_looper = add_loops<false, npy_byte, COMP::EQ, COMP::NE, COMP::LT, COMP::LE, COMP::GT, COMP::GE>;
    if (string_looper()(umath, &spec) < 0) {
        goto finish;
    }

    /* All Unicode loops */
    using ucs_looper = add_loops<false, npy_ucs4, COMP::EQ, COMP::NE, COMP::LT, COMP::LE, COMP::GT, COMP::GE>;
    dtypes[0] = Unicode;
    dtypes[1] = Unicode;
    if (ucs_looper()(umath, &spec) < 0) {
        goto finish;
    }

    res = 0;
  finish:
    Py_DECREF(String);
    Py_DECREF(Unicode);
    Py_DECREF(Bool);
    return res;
}

static int
init_str_len(PyObject *umath)
{
    {
    int res = -1;
    /* NOTE: This should receive global symbols? */
    PyArray_DTypeMeta *String = PyArray_DTypeFromTypeNum(NPY_STRING);
    PyArray_DTypeMeta *Unicode = PyArray_DTypeFromTypeNum(NPY_UNICODE);
    PyArray_DTypeMeta *Long = PyArray_DTypeFromTypeNum(NPY_LONG);

    /* We start with the string loops: */
    PyArray_DTypeMeta *dtypes[] = {String, Long};
    /*
     * We only have one loop right now, the strided one.  The default type
     * resolver ensures native byte order/canonical representation.
     */
    PyType_Slot slots[] = {
        {NPY_METH_strided_loop, nullptr},
        {0, nullptr}
    };

    PyArrayMethod_Spec spec = {};
    spec.name = "templated_string_str_len";
    spec.nin = 1;
    spec.nout = 1;
    spec.dtypes = dtypes;
    spec.slots = slots;
    spec.flags = NPY_METH_NO_FLOATINGPOINT_ERRORS;

    /* All String loops */
    if (add_loop(umath, "str_len", &spec, string_str_len_loop<npy_byte>) < 0) {
        goto finish;
    }

    /* All Unicode loops */
    dtypes[0] = Unicode;
    if (add_loop(umath, "str_len", &spec, string_str_len_loop<npy_ucs4>) < 0) {
        goto finish;
    }

    res = 0;
  finish:
    Py_DECREF(String);
    Py_DECREF(Unicode);
    Py_DECREF(Long);
    return res;
}
}


static int
init_isalpha(PyObject *umath)
{
    int res = -1;
    /* NOTE: This should receive global symbols? */
    PyArray_DTypeMeta *String = PyArray_DTypeFromTypeNum(NPY_STRING);
    PyArray_DTypeMeta *Unicode = PyArray_DTypeFromTypeNum(NPY_UNICODE);
    PyArray_DTypeMeta *Bool = PyArray_DTypeFromTypeNum(NPY_BOOL);

    /* We start with the string loops: */
    PyArray_DTypeMeta *dtypes[] = {String, Bool};
    /*
     * We only have one loop right now, the strided one.  The default type
     * resolver ensures native byte order/canonical representation.
     */
    PyType_Slot slots[] = {
        {NPY_METH_strided_loop, nullptr},
        {0, nullptr}
    };

    PyArrayMethod_Spec spec = {};
    spec.name = "templated_string_isalpha";
    spec.nin = 1;
    spec.nout = 1;
    spec.dtypes = dtypes;
    spec.slots = slots;
    spec.flags = NPY_METH_NO_FLOATINGPOINT_ERRORS;

    /* All String loops */
    if (add_loop(umath, "isalpha", &spec, string_isalpha_loop<npy_byte>) < 0) {
        goto finish;
    }

    /* All Unicode loops */
    dtypes[0] = Unicode;
    if (add_loop(umath, "isalpha", &spec, string_isalpha_loop<npy_ucs4>) < 0) {
        goto finish;
    }

    res = 0;
  finish:
    Py_DECREF(String);
    Py_DECREF(Unicode);
    Py_DECREF(Bool);
    return res;
}


static int
init_find(PyObject *umath)
{
    int res = -1;
    /* NOTE: This should receive global symbols? */
    PyArray_DTypeMeta *String = PyArray_DTypeFromTypeNum(NPY_STRING);
    PyArray_DTypeMeta *Unicode = PyArray_DTypeFromTypeNum(NPY_UNICODE);
    PyArray_DTypeMeta *Long = PyArray_DTypeFromTypeNum(NPY_LONG);

    /* We start with the string loops: */
    PyArray_DTypeMeta *dtypes[] = {String, String, Long, Long, Long};
    /*
     * We only have one loop right now, the strided one.  The default type
     * resolver ensures native byte order/canonical representation.
     */
    PyType_Slot slots[] = {
        {NPY_METH_strided_loop, nullptr},
        {0, nullptr}
    };

    PyArrayMethod_Spec spec = {};
    spec.name = "templated_string_find";
    spec.nin = 4;
    spec.nout = 1;
    spec.dtypes = dtypes;
    spec.slots = slots;
    spec.flags = NPY_METH_NO_FLOATINGPOINT_ERRORS;

    /* All String loops */
    if (add_loop(umath, "find", &spec, string_find_loop<npy_byte>) < 0) {
        goto finish;
    }

    /* All Unicode loops */
    dtypes[0] = Unicode;
    dtypes[1] = Unicode;
    if (add_loop(umath, "find", &spec, string_find_loop<npy_ucs4>) < 0) {
        goto finish;
    }

    res = 0;
  finish:
    Py_DECREF(String);
    Py_DECREF(Unicode);
    Py_DECREF(Long);
    return res;
}


static int
init_rfind(PyObject *umath)
{
    int res = -1;
    /* NOTE: This should receive global symbols? */
    PyArray_DTypeMeta *String = PyArray_DTypeFromTypeNum(NPY_STRING);
    PyArray_DTypeMeta *Unicode = PyArray_DTypeFromTypeNum(NPY_UNICODE);
    PyArray_DTypeMeta *Long = PyArray_DTypeFromTypeNum(NPY_LONG);

    /* We start with the string loops: */
    PyArray_DTypeMeta *dtypes[] = {String, String, Long, Long, Long};
    /*
     * We only have one loop right now, the strided one.  The default type
     * resolver ensures native byte order/canonical representation.
     */
    PyType_Slot slots[] = {
        {NPY_METH_strided_loop, nullptr},
        {0, nullptr}
    };

    PyArrayMethod_Spec spec = {};
    spec.name = "templated_string_rfind";
    spec.nin = 4;
    spec.nout = 1;
    spec.dtypes = dtypes;
    spec.slots = slots;
    spec.flags = NPY_METH_NO_FLOATINGPOINT_ERRORS;

    /* All String loops */
    if (add_loop(umath, "rfind", &spec, string_rfind_loop<npy_byte>) < 0) {
        goto finish;
    }

    /* All Unicode loops */
    dtypes[0] = Unicode;
    dtypes[1] = Unicode;
    if (add_loop(umath, "rfind", &spec, string_rfind_loop<npy_ucs4>) < 0) {
        goto finish;
    }

    res = 0;
  finish:
    Py_DECREF(String);
    Py_DECREF(Unicode);
    Py_DECREF(Long);
    return res;
}


static int
init_count(PyObject *umath)
{
    int res = -1;
    /* NOTE: This should receive global symbols? */
    PyArray_DTypeMeta *String = PyArray_DTypeFromTypeNum(NPY_STRING);
    PyArray_DTypeMeta *Unicode = PyArray_DTypeFromTypeNum(NPY_UNICODE);
    PyArray_DTypeMeta *Long = PyArray_DTypeFromTypeNum(NPY_LONG);

    /* We start with the string loops: */
    PyArray_DTypeMeta *dtypes[] = {String, String, Long, Long, Long};
    /*
     * We only have one loop right now, the strided one.  The default type
     * resolver ensures native byte order/canonical representation.
     */
    PyType_Slot slots[] = {
        {NPY_METH_strided_loop, nullptr},
        {0, nullptr}
    };

    PyArrayMethod_Spec spec = {};
    spec.name = "templated_string_count";
    spec.nin = 4;
    spec.nout = 1;
    spec.dtypes = dtypes;
    spec.slots = slots;
    spec.flags = NPY_METH_NO_FLOATINGPOINT_ERRORS;

    /* All String loops */
    if (add_loop(umath, "count", &spec, string_count_loop<npy_byte>) < 0) {
        goto finish;
    }

    /* All Unicode loops */
    dtypes[0] = Unicode;
    dtypes[1] = Unicode;
    if (add_loop(umath, "count", &spec, string_count_loop<npy_ucs4>) < 0) {
        goto finish;
    }

    res = 0;
  finish:
    Py_DECREF(String);
    Py_DECREF(Unicode);
    Py_DECREF(Long);
    return res;
}


static NPY_CASTING
string_replace_resolve_descriptors(
        PyArrayMethodObject *NPY_UNUSED(self),
        PyArray_DTypeMeta *NPY_UNUSED(dtypes[3]),
        PyArray_Descr *given_descrs[5],
        PyArray_Descr *loop_descrs[5],
        npy_intp *NPY_UNUSED(view_offset))
{
    if (given_descrs[4] == NULL) {
        PyErr_SetString(PyExc_ValueError, "out kwarg should be given");
        return _NPY_ERROR_OCCURRED_IN_CAST;
    }
    loop_descrs[0] = NPY_DT_CALL_ensure_canonical(given_descrs[0]);
    loop_descrs[1] = NPY_DT_CALL_ensure_canonical(given_descrs[1]);
    loop_descrs[2] = NPY_DT_CALL_ensure_canonical(given_descrs[2]);
    loop_descrs[3] = NPY_DT_CALL_ensure_canonical(given_descrs[3]);

    Py_INCREF(given_descrs[4]);
    loop_descrs[4] = given_descrs[4];
    return NPY_NO_CASTING;
}


static int
init_replace(PyObject *umath)
{
    int res = -1;
    /* NOTE: This should receive global symbols? */
    PyArray_DTypeMeta *String = PyArray_DTypeFromTypeNum(NPY_STRING);
    PyArray_DTypeMeta *Unicode = PyArray_DTypeFromTypeNum(NPY_UNICODE);
    PyArray_DTypeMeta *Long = PyArray_DTypeFromTypeNum(NPY_LONG);

    /* We start with the string loops: */
    PyArray_DTypeMeta *dtypes[] = {String, String, String, Long, String};
    /*
     * We only have one loop right now, the strided one.  The default type
     * resolver ensures native byte order/canonical representation.
     */
    PyType_Slot slots[] = {
        {NPY_METH_strided_loop, nullptr},
        {NPY_METH_resolve_descriptors, (void *) &string_replace_resolve_descriptors},
        {0, nullptr}
    };

    PyArrayMethod_Spec spec = {};
    spec.name = "templated_string_replace";
    spec.nin = 4;
    spec.nout = 1;
    spec.dtypes = dtypes;
    spec.slots = slots;
    spec.flags = NPY_METH_NO_FLOATINGPOINT_ERRORS;

    /* All String loops */
    if (add_loop(umath, "__replace_impl", &spec, string_replace_loop<npy_byte>) < 0) {
        goto finish;
    }

    /* All Unicode loops */
    dtypes[0] = Unicode;
    dtypes[1] = Unicode;
    dtypes[2] = Unicode;
    dtypes[4] = Unicode;
    if (add_loop(umath, "__replace_impl", &spec, string_replace_loop<npy_ucs4>) < 0) {
        goto finish;
    }

    res = 0;
  finish:
    Py_DECREF(String);
    Py_DECREF(Unicode);
    Py_DECREF(Long);
    return res;
}


NPY_NO_EXPORT int
init_string_ufuncs(PyObject *umath)
{
    if (init_comparison(umath) < 0) {
        return -1;
    }

    if (init_str_len(umath) < 0) {
        return -1;
    }

    if (init_isalpha(umath) < 0) {
        return -1;
    }

    if (init_find(umath) < 0) {
        return -1;
    }

    if (init_rfind(umath) < 0) {
        return -1;
    }

    if (init_count(umath) < 0) {
        return -1;
    }

    if (init_replace(umath) < 0) {
        return -1;
    }

    return 0;
}


template <bool rstrip, typename character>
static PyArrayMethod_StridedLoop *
get_strided_loop(int comp)
{
    switch (comp) {
        case Py_EQ:
            return string_comparison_loop<rstrip, COMP::EQ, character>;
        case Py_NE:
            return string_comparison_loop<rstrip, COMP::NE, character>;
        case Py_LT:
            return string_comparison_loop<rstrip, COMP::LT, character>;
        case Py_LE:
            return string_comparison_loop<rstrip, COMP::LE, character>;
        case Py_GT:
            return string_comparison_loop<rstrip, COMP::GT, character>;
        case Py_GE:
            return string_comparison_loop<rstrip, COMP::GE, character>;
        default:
            assert(false);  /* caller ensures this */
    }
    return nullptr;
}


/*
 * This function is used for `compare_chararrays` and currently also void
 * comparisons (unstructured voids).  The first could probably be deprecated
 * and removed but is used by `np.char.chararray` the latter should also be
 * moved to the ufunc probably (removing the need for manual looping).
 *
 * The `rstrip` mechanism is presumably for some fortran compat, but the
 * question is whether it would not be better to have/use `rstrip` on such
 * an array first...
 *
 * NOTE: This function is also used for unstructured voids, this works because
 *       `npy_byte` is correct.
 */
NPY_NO_EXPORT PyObject *
_umath_strings_richcompare(
        PyArrayObject *self, PyArrayObject *other, int cmp_op, int rstrip)
{
    NpyIter *iter = nullptr;
    PyObject *result = nullptr;

    char **dataptr = nullptr;
    npy_intp *strides = nullptr;
    npy_intp *countptr = nullptr;
    npy_intp size = 0;

    PyArrayMethod_Context context = {};
    NpyIter_IterNextFunc *iternext = nullptr;

    npy_uint32 it_flags = (
            NPY_ITER_EXTERNAL_LOOP | NPY_ITER_ZEROSIZE_OK |
            NPY_ITER_BUFFERED | NPY_ITER_GROWINNER);
    npy_uint32 op_flags[3] = {
            NPY_ITER_READONLY | NPY_ITER_ALIGNED,
            NPY_ITER_READONLY | NPY_ITER_ALIGNED,
            NPY_ITER_WRITEONLY | NPY_ITER_ALLOCATE | NPY_ITER_ALIGNED};

    PyArrayMethod_StridedLoop *strided_loop = nullptr;
    NPY_BEGIN_THREADS_DEF;

    if (PyArray_TYPE(self) != PyArray_TYPE(other)) {
        /*
         * Comparison between Bytes and Unicode is not defined in Py3K;
         * we follow.
         * TODO: This makes no sense at all for `compare_chararrays`, kept
         *       only under the assumption that we are more likely to deprecate
         *       than fix it to begin with.
         */
        Py_INCREF(Py_NotImplemented);
        return Py_NotImplemented;
    }

    PyArrayObject *ops[3] = {self, other, nullptr};
    PyArray_Descr *descrs[3] = {nullptr, nullptr, PyArray_DescrFromType(NPY_BOOL)};
    /* TODO: ensuring native byte order is not really necessary for == and != */
    descrs[0] = NPY_DT_CALL_ensure_canonical(PyArray_DESCR(self));
    if (descrs[0] == nullptr) {
        goto finish;
    }
    descrs[1] = NPY_DT_CALL_ensure_canonical(PyArray_DESCR(other));
    if (descrs[1] == nullptr) {
        goto finish;
    }

    /*
     * Create the iterator:
     */
    iter = NpyIter_AdvancedNew(
            3, ops, it_flags, NPY_KEEPORDER, NPY_SAFE_CASTING, op_flags, descrs,
            -1, nullptr, nullptr, 0);
    if (iter == nullptr) {
        goto finish;
    }

    size = NpyIter_GetIterSize(iter);
    if (size == 0) {
        result = (PyObject *)NpyIter_GetOperandArray(iter)[2];
        Py_INCREF(result);
        goto finish;
    }

    iternext = NpyIter_GetIterNext(iter, nullptr);
    if (iternext == nullptr) {
        goto finish;
    }

    /*
     * Prepare the inner-loop and execute it (we only need descriptors to be
     * passed in).
     */
    context.descriptors = descrs;

    dataptr = NpyIter_GetDataPtrArray(iter);
    strides = NpyIter_GetInnerStrideArray(iter);
    countptr = NpyIter_GetInnerLoopSizePtr(iter);

    if (rstrip == 0) {
        /* NOTE: Also used for VOID, so can be STRING, UNICODE, or VOID: */
        if (descrs[0]->type_num != NPY_UNICODE) {
            strided_loop = get_strided_loop<false, npy_byte>(cmp_op);
        }
        else {
            strided_loop = get_strided_loop<false, npy_ucs4>(cmp_op);
        }
    }
    else {
        if (descrs[0]->type_num != NPY_UNICODE) {
            strided_loop = get_strided_loop<true, npy_byte>(cmp_op);
        }
        else {
            strided_loop = get_strided_loop<true, npy_ucs4>(cmp_op);
        }
    }

    NPY_BEGIN_THREADS_THRESHOLDED(size);

    do {
         /* We know the loop cannot fail */
         strided_loop(&context, dataptr, countptr, strides, nullptr);
    } while (iternext(iter) != 0);

    NPY_END_THREADS;

    result = (PyObject *)NpyIter_GetOperandArray(iter)[2];
    Py_INCREF(result);

 finish:
    if (NpyIter_Deallocate(iter) < 0) {
        Py_CLEAR(result);
    }
    Py_XDECREF(descrs[0]);
    Py_XDECREF(descrs[1]);
    Py_XDECREF(descrs[2]);
    return result;
}
