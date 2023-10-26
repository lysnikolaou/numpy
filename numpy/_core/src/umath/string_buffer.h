#ifndef _NPY_CORE_SRC_UMATH_STRING_BUFFER_H_
#define _NPY_CORE_SRC_UMATH_STRING_BUFFER_H_

#include <Python.h>
#include <assert.h>

#define NPY_NO_DEPRECATED_API NPY_API_VERSION
#define _MULTIARRAYMODULE
#define _UMATHMODULE

#include "numpy/ndarraytypes.h"

#define CHECK_OVERFLOW(index) if ((index) >= elsize) return 0
#define MSB(val) ((val) >> 7 & 1)


enum class ENCODING {
    ASCII, UTF8, UTF32
};


template <ENCODING enc>
inline npy_ucs4
getchar(const unsigned char *buf, int *bytes);


template <>
inline npy_ucs4
getchar<ENCODING::ASCII>(const unsigned char *buf, int *bytes)
{
    *bytes = 1;
    return (npy_ucs4) *buf;
}


template <>
inline npy_ucs4
getchar<ENCODING::UTF32>(const unsigned char *buf, int *bytes)
{
    *bytes = 4;
    return *(npy_ucs4 *)buf;
}


template <>
inline npy_ucs4
getchar<ENCODING::UTF8>(const unsigned char *buf, int *bytes)
{
    if (buf[0] <= 0x7F) {
        // 0zzzzzzz -> 0zzzzzzz
        *bytes = 1;
        return (npy_ucs4)(buf[0]);
    }
    else if (buf[0] <= 0xDF) {
        // 110yyyyy 10zzzzzz -> 00000yyy yyzzzzzz
        *bytes = 2;
        return (npy_ucs4)(((buf[0] << 6) + buf[1]) - ((0xC0 << 6) + 0x80));
    }
    else if (buf[0] <= 0xEF) {
        // 1110xxxx 10yyyyyy 10zzzzzz -> xxxxyyyy yyzzzzzz
        *bytes = 3;
        return (npy_ucs4)(((buf[0] << 12) + (buf[1] << 6) + buf[2]) -
                          ((0xE0 << 12) + (0x80 << 6) + 0x80));
    }
    else {
        // 11110www 10xxxxxx 10yyyyyy 10zzzzzz -> 000wwwxx xxxxyyyy yyzzzzzz
        *bytes = 4;
        return (npy_ucs4)(((buf[0] << 18) + (buf[1] << 12) + (buf[2] << 6) + buf[3]) -
                          ((0xF0 << 18) + (0x80 << 12) + (0x80 << 6) + 0x80));
    }
}


template <ENCODING enc>
struct Buffer {
    char *buf;
    int elsize;

    inline Buffer<enc>() {
        buf = NULL;
        elsize = 0;
    }

    inline Buffer<enc>(char *buf_, int elsize_) :
        buf(buf_),
        elsize(elsize_) { }

    inline int length() {
        Buffer tmp(buf + elsize, 0);
        tmp--;
        while (tmp >= *this && *tmp == '\0') {
            tmp--;
        }
        return tmp - *this + 1;
    }

    inline Buffer<enc>& operator+=(npy_int64 rhs) {
        switch (enc) {
        case ENCODING::ASCII:
            buf += rhs;
            break;
        case ENCODING::UTF8:
            int bytes;
            getchar<enc>((const unsigned char *) buf, &bytes);
            buf += rhs * bytes;
            break;
        case ENCODING::UTF32:
            buf += rhs * sizeof(npy_ucs4);
            break;
        }
        return *this;
    }

    inline Buffer<enc>& operator-=(npy_int64 rhs) {
        switch (enc) {
        case ENCODING::ASCII:
            buf -= rhs;
            break;
        case ENCODING::UTF8: {
            for (int i = 0; i < rhs; i++) {
                for (int j = 0; j < 4; j++) {
                    buf--;
                    if (MSB(*buf) == 0) {
                        break;
                    }
                }
            }
        }
        case ENCODING::UTF32:
            buf -= rhs * sizeof(npy_ucs4);
            break;
        }
        return *this;
    }

    inline Buffer<enc>& operator++() {
        *this += 1;
        return *this;
    }

    inline Buffer<enc> operator++(int) {
        Buffer<enc> old = *this;
        operator++();
        return old; 
    }

    inline Buffer<enc>& operator--() {
        *this -= 1;
        return *this;
    }

    inline Buffer<enc> operator--(int) {
        Buffer<enc> old = *this;
        operator--();
        return old; 
    }

    inline npy_ucs4 operator*() {
        int bytes;
        return getchar<enc>((unsigned char *) buf, &bytes);
    }

    inline npy_ucs4 operator[](size_t index) {
        int bytes;
        switch (enc) {
        case ENCODING::ASCII:
            CHECK_OVERFLOW(index);
            return getchar<enc>((unsigned char *) (buf + index), &bytes);
        case ENCODING::UTF8: {
            int i, j;
            for (i = 0, j = 0; i < index; i++) {
                CHECK_OVERFLOW(j);
                getchar<enc>((unsigned char *) (buf + j), &bytes);
                j += bytes;
            }
            CHECK_OVERFLOW(j);
            return getchar<enc>((unsigned char *) (buf + j), &bytes);
        }
        case ENCODING::UTF32:
            CHECK_OVERFLOW(index * sizeof(npy_ucs4));
            return getchar<enc>((unsigned char *) (buf + index * sizeof(npy_ucs4)), &bytes);
        }
    }

    inline Buffer<enc> fast_memchr(npy_ucs4 ch, int len) {
        switch (enc) {
        case ENCODING::ASCII:
            buf = (char *) memchr(buf, ch, len);
            return *this;
        case ENCODING::UTF8:
        case ENCODING::UTF32:
            buf = (char *) wmemchr((wchar_t *) buf, ch, len);
            return *this;
        }
    }

    inline int fast_memcmp(Buffer<enc> other, int len) {
        return memcmp(buf, other.buf, len);
    }
};


template <ENCODING enc>
inline Buffer<enc> operator+(Buffer<enc> lhs, npy_int64 rhs) {
    lhs += rhs;
    return lhs;
}


template <ENCODING enc>
inline size_t operator-(Buffer<enc> lhs, Buffer<enc> rhs) {
    switch (enc) {
    case ENCODING::ASCII:
        return lhs.buf - rhs.buf;
    case ENCODING::UTF8: {
        int bytes, j = 0, res = 0;
        while (lhs.buf + j < rhs.buf) {
            getchar<enc>((unsigned char *) (lhs.buf + j), &bytes);
            j += bytes;
            res++;
        }
        assert(lhs.buf + j == rhs.buf);
        return res;
    }
    case ENCODING::UTF32:
        return (lhs.buf - rhs.buf) / sizeof(npy_ucs4);
    }
}


template <ENCODING enc>
inline Buffer<enc> operator-(Buffer<enc> lhs, npy_int64 rhs) {
    lhs -= rhs;
    return lhs;
}


template <ENCODING enc>
inline bool operator==(Buffer<enc> lhs, Buffer<enc> rhs) {
    return lhs.buf == rhs.buf;
}


template <ENCODING enc>
inline bool operator!=(Buffer<enc> lhs, Buffer<enc> rhs) {
    return !(rhs == lhs);
}


template <ENCODING enc>
inline bool operator<(Buffer<enc> lhs, Buffer<enc> rhs) {
    return lhs.buf < rhs.buf;
}


template <ENCODING enc>
inline bool operator>(Buffer<enc> lhs, Buffer<enc> rhs) {
    return rhs < lhs;
}


template <ENCODING enc>
inline bool operator<=(Buffer<enc> lhs, Buffer<enc> rhs) {
    return !(lhs > rhs);
}


template <ENCODING enc>
inline bool operator>=(Buffer<enc> lhs, Buffer<enc> rhs) {
    return !(lhs < rhs);
}


#endif /* _NPY_CORE_SRC_UMATH_STRING_BUFFER_H_ */
