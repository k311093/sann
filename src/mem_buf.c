#include "mem_buf.h"

#include "lj_arch.h"
#include "lj_def.h"

MemBuf *mem_buf_read_new(const void *data, uint32_t len)
{
    MemBuf *buf = malloc(sizeof(MemBuf));

    buf->rdata = (const uint8_t *)data;
    buf->wdata = NULL;
    buf->pos = 0;
    buf->len = len;

    return buf;
}

MemBuf *mem_buf_write_new(void)
{
    MemBuf *buf = malloc(sizeof(MemBuf));

    buf->rdata = NULL;
    buf->wdata = malloc(sizeof(uint8_t) * kMemBufUnitSize);
    buf->pos = 0;
    buf->len = kMemBufUnitSize;

    return buf;
}

void mem_buf_free(MemBuf *buf)
{
    if (buf->wdata != NULL) {
        free(buf->wdata);
    }

    free(buf);
}

uint32_t mem_buf_read_uint32(MemBuf *buf)
{
    uint32_t result = 0;

    if (buf->rdata != NULL
        && !(buf->len < buf->pos + sizeof(uint32_t))) {
        result = *((const uint32_t *)(buf->rdata + buf->pos));
        buf->pos += sizeof(uint32_t);
    }

    return LJ_ENDIAN_SELECT(result, lj_bswap(result));
}

double mem_buf_read_double(MemBuf *buf)
{
    MemDouble md;
    uint64_t data = 0;

    if (buf->rdata != NULL
        && !(buf->len < buf->pos + sizeof(double))) {
        data = *((const uint64_t *)(buf->rdata + buf->pos));
        md.u = LJ_ENDIAN_SELECT(data, lj_bswap64(data));
        buf->pos += sizeof(double);
    }

    return md.d;
}

uint32_t mem_buf_read_data(MemBuf *buf, void *dst, uint32_t size)
{
    uint32_t len = 0;

    if (buf->rdata != NULL) {
        if (!(buf->len < buf->pos + size)) {
            len = size;
        }
        else {
            len = buf->len - buf->pos;
        }

        memcpy(dst, buf->rdata + buf->pos, len);
        buf->pos += len;
    }

    return len;
}

void mem_buf_write_uint32(MemBuf *buf, uint32_t val)
{
    if (buf->wdata != NULL) {
        if (buf->len < buf->pos + sizeof(uint32_t)) {
            buf->wdata = realloc(buf->wdata,
                sizeof(uint8_t) * (buf->len + kMemBufUnitSize));
            buf->len += kMemBufUnitSize;
        }

        *((uint32_t *)(buf->wdata + buf->pos)) =
            LJ_ENDIAN_SELECT(val, lj_bswap(val));
        buf->pos += sizeof(uint32_t);
    }
}

void mem_buf_write_double(MemBuf *buf, double val)
{
    MemDouble md;

    if (buf->wdata != NULL) {
        if (buf->len < buf->pos + sizeof(double)) {
            buf->wdata = realloc(buf->wdata,
                sizeof(uint8_t) * (buf->len + kMemBufUnitSize));
            buf->len += kMemBufUnitSize;
        }

        md.d = val;

        *((uint64_t *)(buf->wdata + buf->pos)) =
            LJ_ENDIAN_SELECT(md.u, lj_bswap64(md.u));
        buf->pos += sizeof(double);
    }
}

void mem_buf_write_data(MemBuf *buf, const void *src, uint32_t size)
{
    if (buf->wdata != NULL) {
        if (buf->len < buf->pos + size) {
            buf->wdata = realloc(buf->wdata,
                sizeof(uint8_t) * (buf->len + size + kMemBufUnitSize));
            buf->len += (size + kMemBufUnitSize);
        }

        memcpy(buf->wdata + buf->pos, src, size);

        buf->pos += size;
    }
}

void mem_buf_toluastring(lua_State *L, MemBuf *buf)
{
    if (buf->wdata != NULL) {
        lua_pushlstring(L, (const char *)buf->wdata, buf->pos);
    }
    else {
        lua_pushnil(L);
    }
}

