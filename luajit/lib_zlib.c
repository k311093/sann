#include "lua.h"
#include "lauxlib.h"
#include "lualib.h"

#include "mem_buf.h"
#include "zlib/zlib.h"

#define CHUNK 16384

static int zlib_deflate(lua_State *L)
{
    const char *data = NULL;
    size_t dataLen = 0;
    MemBuf *inBuf = NULL;
    MemBuf *outBuf = NULL;
    int ret = 0, flush = 0;
    unsigned have;
    z_stream strm;
    unsigned char in[CHUNK];
    unsigned char out[CHUNK];

    data = luaL_checklstring(L, -1, &dataLen);

    inBuf = mem_buf_read_new(data, (uint32_t)dataLen);
    outBuf = mem_buf_write_new();

    strm.zalloc = Z_NULL;
    strm.zfree = Z_NULL;
    strm.opaque = Z_NULL;
    ret = deflateInit(&strm, Z_DEFAULT_COMPRESSION);

    if (ret != Z_OK) {
        luaL_error(L, "deflateInit error");
    }

    do {
        strm.avail_in = mem_buf_read_data(inBuf, in, CHUNK);
        strm.next_in = in;
        flush = (strm.avail_in == CHUNK) ? Z_NO_FLUSH : Z_FINISH;

        do {
            strm.avail_out = CHUNK;
            strm.next_out = out;
            ret = deflate(&strm, flush);

            if (ret == Z_STREAM_ERROR) {
                deflateEnd(&strm);
                luaL_error(L, "deflate error");
            }

            have = CHUNK - strm.avail_out;

            mem_buf_write_data(outBuf, out, have);
        } while (strm.avail_out == 0);

    } while (flush != Z_FINISH);

    deflateEnd(&strm);

    mem_buf_toluastring(L, outBuf);
    mem_buf_free(inBuf);
    mem_buf_free(outBuf);

    return 1;
}

static int zlib_inflate(lua_State *L)
{
    const char *data = NULL;
    size_t dataLen = 0;
    MemBuf *inBuf = NULL;
    MemBuf *outBuf = NULL;
    int ret = 0;
    unsigned have;
    z_stream strm;
    unsigned char in[CHUNK];
    unsigned char out[CHUNK];

    data = luaL_checklstring(L, -1, &dataLen);

    inBuf = mem_buf_read_new(data, (uint32_t)dataLen);
    outBuf = mem_buf_write_new();

    strm.zalloc = Z_NULL;
    strm.zfree = Z_NULL;
    strm.opaque = Z_NULL;
    strm.avail_in = 0;
    strm.next_in = Z_NULL;
    ret = inflateInit(&strm);

    if (ret != Z_OK) {
        luaL_error(L, "inflateInit error");
    }

    do {
        strm.avail_in = mem_buf_read_data(inBuf, in, CHUNK);

        if (strm.avail_in == 0)
            break;
        strm.next_in = in;

        do {
            strm.avail_out = CHUNK;
            strm.next_out = out;
            ret = inflate(&strm, Z_NO_FLUSH);

            if (ret == Z_STREAM_ERROR) {
                inflateEnd(&strm);
                luaL_error(L, "inflate error");
            }

            switch (ret) {
                case Z_NEED_DICT:
                    ret = Z_DATA_ERROR;
                case Z_DATA_ERROR:
                case Z_MEM_ERROR:
                    inflateEnd(&strm);
                    luaL_error(L, "inflate data error");
            }

            have = CHUNK - strm.avail_out;

            mem_buf_write_data(outBuf, out, have);
        } while (strm.avail_out == 0);

    } while (ret != Z_STREAM_END);

    inflateEnd(&strm);

    mem_buf_toluastring(L, outBuf);
    mem_buf_free(inBuf);
    mem_buf_free(outBuf);

    return 1;
}

static luaL_reg zlibFuncs[] = {
    { "deflate", zlib_deflate },
    { "inflate", zlib_inflate },
    { NULL, NULL },
};

int luaopen_zlib(lua_State *L)
{
    luaL_register(L, "zlib", zlibFuncs);

    return 1;
}
