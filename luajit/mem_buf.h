#ifndef _MEM_BUF_H_
#define _MEM_BUF_H_

#include "lua.h"
#include "lauxlib.h"
#include "lualib.h"

#include <stdint.h>
#include <string.h>
#include <stdlib.h>

#define kMemBufUnitSize       4096

typedef union _MemDouble {
    uint64_t u;
    double d;
} MemDouble;

typedef struct _MemBuf {
    const uint8_t *rdata;
    uint8_t *wdata;
    uint32_t len;
    uint32_t pos;
} MemBuf;

MemBuf *mem_buf_read_new(const void *data, uint32_t len);
MemBuf *mem_buf_write_new(void);
void mem_buf_free(MemBuf *buf);
uint32_t mem_buf_read_uint32(MemBuf *buf);
double mem_buf_read_double(MemBuf *buf);
uint32_t mem_buf_read_data(MemBuf *buf, void *dst, uint32_t size);
void mem_buf_write_uint32(MemBuf *buf, uint32_t val);
void mem_buf_write_double(MemBuf *buf, double val);
void mem_buf_write_data(MemBuf *buf, const void *src, uint32_t size);
void mem_buf_toluastring(lua_State *L, MemBuf *buf);

#endif /* _MEM_BUF_H_ */

