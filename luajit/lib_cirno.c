#include "lua.h"
#include "lauxlib.h"
#include "lualib.h"

#include "mem_buf.h"

#include <stdint.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <time.h>
#include <float.h>

/* ------------------------------------------------------------------------ */
/* MACROS */
/* ------------------------------------------------------------------------ */

#ifndef UNREFERENCED_PARAMETER
#define UNREFERENCED_PARAMETER(X)   ((void) X)
#endif /* UNREFERENCED_PARAMETER */

/* ------------------------------------------------------------------------ */
/* CONSTANTS */
/* ------------------------------------------------------------------------ */

#define TRUE    1
#define FALSE   0

#define kMtConstW ((uint32_t)32)
#define kMtConstN ((uint32_t)624)
#define kMtConstM ((uint32_t)397)
#define kMtConstR ((uint32_t)31)
#define kMtConstA ((uint32_t)0x9908B0DF)
#define kMtConstU ((uint32_t)11)
#define kMtConstD ((uint32_t)0xFFFFFFFF)
#define kMtConstS ((uint32_t)7)
#define kMtConstB ((uint32_t)0x9D2C5680)
#define kMtConstT ((uint32_t)15)
#define kMtConstC ((uint32_t)0xEFC60000)
#define kMtConstL ((uint32_t)18)
#define kMtConstF ((uint32_t)0x6C078965)

#define kMtConstUpperMask ((uint32_t)0x80000000)
#define kMtConstLowerMask ((uint32_t)0x7FFFFFFF)

#define kMtConstDefaultSeed ((uint32_t)5489)

#define kLevaConstS  ((double)0.449871)
#define kLevaConstT  ((double)0.386595)
#define kLevaConstA  ((double)0.19600)
#define kLevaConstB  ((double)0.25472)
#define kLevaConstR1 ((double)0.27597)
#define kLevaConstR2 ((double)0.27846)

#define kCirnoLayerNull         0
#define kCirnoLayerConv         1
#define kCirnoLayerDropout      2
#define kCirnoLayerFullyConn    3
#define kCirnoLayerInput        4
#define kCirnoLayerPad          5
#define kCirnoLayerPool         6
#define kCirnoLayerRegression   7
#define kCirnoLayerRelu         8
#define kCirnoLayerSigmoid      9
#define kCirnoLayerSoftmax      10
#define kCirnoLayerStride       11
#define kCirnoLayerSvm          12
#define kCirnoLayerTanh         13
#define kCirnoLayerLstm         14
#define kCirnoLayerMdpAgent     15
#define kCirnoLayerRecurrent    16
#define kCirnoLayerNoise        17
#define kCirnoLayerEnd          18

#define kCirnoActivationNull    0
#define kCirnoActivationRelu    1
#define kCirnoActivationSigmoid 2
#define kCirnoActivationTanh    3
#define kCirnoActivationEnd     4

#define kCirnoNoiseNull         0
#define kCirnoNoiseUniform      1
#define kCirnoNoiseRUniform     2
#define kCirnoNoiseGaussian     3
#define kCirnoNoiseRGaussian    4
#define kCirnoNoiseEnd          5

#define kCirnoTrainerNull       0
#define kCirnoTrainerAdaDelta   1
#define kCirnoTrainerAdaGrad    2
#define kCirnoTrainerAdam       3
#define kCirnoTrainerNesterov   4
#define kCirnoTrainerSgd        5
#define kCirnoTrainerWindowGrad 6
#define kCirnoTrainerEnd        7

/* ------------------------------------------------------------------------ */
/* MT19937 */
/* ------------------------------------------------------------------------ */

static uint32_t g_mtState[kMtConstN] = {0,};
static uint32_t g_mtIndex = kMtConstN + 1;

static void mt19937_seed(uint32_t seed)
{
    uint32_t i = 0;

    g_mtIndex = kMtConstN;
    g_mtState[0] = seed;

    for (i = 1; i < kMtConstN; ++i) {
        g_mtState[i] = kMtConstF * (g_mtState[i - 1] ^
                       (g_mtState[i - 1] >> (kMtConstW - 2))) + i;
    }
}

static void mt19937_twist(void)
{
    uint32_t i = 0;
    uint32_t y = 0;

    for (i = 0; i < kMtConstN; ++i) {
        y = (g_mtState[i] & kMtConstUpperMask) +
            (g_mtState[(i + 1) % kMtConstN] & kMtConstLowerMask);

        g_mtState[i] = g_mtState[(i + kMtConstM) % kMtConstN] ^ (y >> 1);

        if (y % 2) {
            g_mtState[i] ^= kMtConstA;
        }

    }

    g_mtIndex = 0;
}

static uint32_t mt19937_rand(void)
{
    uint32_t result = 0;

    if (g_mtIndex == kMtConstN) {
        mt19937_twist();
    }

    result = g_mtState[g_mtIndex++];

    result ^= (result >> kMtConstU);
    result ^= ((result << kMtConstS) & kMtConstB);
    result ^= ((result << kMtConstT) & kMtConstC);
    result ^= (result >> kMtConstL);

    return result;
}

/* ------------------------------------------------------------------------ */
/* RANDOM */
/* ------------------------------------------------------------------------ */

static double runif(void)
{
    return (((double)mt19937_rand()) * (1.0 / 4294967296.0));
}

static double rnorm(void)
{
    double u = 0.0;
    double v = 0.0;
    double x = 0.0;
    double y = 0.0;
    double q = 0.0;

    do {
        u = 1.0 - runif();
        v = 1.7156 * (runif() - 0.5);
        x = u - kLevaConstS;
        y = fabs(v) + kLevaConstT;
        q = (x * x) + y * (kLevaConstA * y - kLevaConstB * x);
    } while (q > kLevaConstR1 &&
        ((q > kLevaConstR2) || ((v * v) > -4.0 * log(u) * (u * u))));

    return (v / u);
}

/* ------------------------------------------------------------------------ */
/* DATATYPE */
/* ------------------------------------------------------------------------ */

typedef struct _CirnoVol {
    uint32_t sx;
    uint32_t sy;
    uint32_t sz;
    uint32_t depth;
    uint32_t n;
    uint32_t useGxsum;
    uint32_t nSnapshot;
    uint32_t curSnapshot;
    uint32_t k;

    double *wBase;
    double *dwBase;

    double *w;
    double *dw;

    double *gsum;
    double *xsum;
} CirnoVol;

typedef struct _CirnoLayerInfoConv {
    double l1DecayMul;
    double l2DecayMul;

    uint32_t sx;
    uint32_t sy;
    uint32_t sz;
    uint32_t depth;

    CirnoVol *filter;
    CirnoVol *bias;
} CirnoLayerInfoConv;

typedef struct _CirnoLayerInfoDopout {
    uint32_t nSnapshot;
    uint32_t curSnapshot;
    double drop;
    double present;

    uint8_t *isDroppedBase;
    uint8_t *isDropped;
} CirnoLayerInfoDropout;

typedef struct _CirnoLayerInfoFullyConn {
    double l1DecayMul;
    double l2DecayMul;

    CirnoVol *weight;
    CirnoVol *bias;
} CirnoLayerInfoFullyConn;

typedef struct _CirnoLayerInfoPad {
    uint32_t padx;
    uint32_t pady;
    uint32_t padz;
} CirnoLayerInfoPad;

typedef struct _CirnoLayerInfoPool {
    uint32_t nSnapshot;
    uint32_t curSnapshot;
    uint32_t sx;
    uint32_t sy;
    uint32_t sz;

    uint32_t *switchxBase;
    uint32_t *switchyBase;
    uint32_t *switchzBase;

    uint32_t *switchx;
    uint32_t *switchy;
    uint32_t *switchz;
} CirnoLayerInfoPool;

typedef struct _CirnoLayerInfoStride {
    uint32_t stride;
} CirnoLayerInfoStride;

typedef struct _CirnoLayerInfoLstm {
    double l1DecayMul;
    double l2DecayMul;

    uint32_t hsize;

    CirnoVol *rin;
    CirnoVol *rout;
    CirnoVol *weight;
    CirnoVol *bias;

    CirnoVol *figo;
    CirnoVol *afigo;
} CirnoLayerInfoLstm;

typedef struct _CirnoLayerInfoMdpAgent {
    uint32_t nSnapshot;
    uint32_t curSnapshot;

    double gamma;
    double eps;

    double *rewards;
    uint32_t *actions;
} CirnoLayerInfoMdpAgent;

typedef struct _CirnoLayerInfoRecurrent {
    double l1DecayMul;
    double l2DecayMul;

    CirnoVol *rin;
    CirnoVol *rout;
    CirnoVol *weight;
    CirnoVol *bias;
} CirnoLayerInfoRecurrent;

typedef struct _CirnoLayerInfoNoise {
    uint32_t noiseType;

    double nw;
    double nsd;
} CirnoLayerInfoNoise;

typedef struct _CirnoTrainerOption {
    uint32_t trainMethod;
    double lr;
    double l1Decay;
    double l2Decay;
    uint32_t batch;
    uint32_t manual;
    double momentum;
    double ro;
    double eps;
    double beta1;
    double beta2;
    double maxGrad;
    uint64_t k;
    uint32_t bpttStep;
} CirnoTrainerOption;

typedef struct _CirnoLayerOption {
    uint32_t type;
    uint32_t noise;
    uint32_t act;
    double drop;
    double l1DecayMul;
    double l2DecayMul;
    uint32_t out;
    uint32_t hidden;
    uint32_t sx;
    uint32_t sy;
    uint32_t sz;
    uint32_t depth;
    uint32_t filters;
    uint32_t stride;
    uint32_t padx;
    uint32_t pady;
    uint32_t padz;
    double gamma;
    double eps;
    double nw;
    double nsd;
} CirnoLayerOption;

typedef struct _CirnoLayer {
    uint32_t type;

    union _LayerInfo {
        CirnoLayerInfoConv *conv;
        CirnoLayerInfoDropout *dropout;
        CirnoLayerInfoFullyConn *fullyConn;
        CirnoLayerInfoPad *pad;
        CirnoLayerInfoPool *pool;
        CirnoLayerInfoStride *stride;
        CirnoLayerInfoLstm *lstm;
        CirnoLayerInfoMdpAgent *mdpAgent;
        CirnoLayerInfoRecurrent *recurrent;
        CirnoLayerInfoNoise *noise;
    } info;

    CirnoVol *inVol;
    CirnoVol *outVol;
} CirnoLayer;

typedef void (*cirno_layer_free)(CirnoLayer *layer);
typedef void (*cirno_layer_forward)(CirnoLayer *layer, uint8_t isTraining);
typedef void (*cirno_layer_backward)(CirnoLayer *layer, uint32_t t, uint8_t resetHidden);
typedef void (*cirno_layer_clear_rin)(CirnoLayer *layer);
typedef double (*cirno_layer_apply_grad)(CirnoLayer *layer, CirnoTrainerOption *option);
typedef void (*cirno_layer_persist)(MemBuf *buf, CirnoLayer *layer);
typedef CirnoVol *(*cirno_layer_unpersist)(lua_State *L, int index, CirnoVol *prevVol,
    MemBuf *buf, int newIndex);
typedef void (*cirno_layer_persist_grad)(MemBuf *buf, CirnoLayer *layer);
typedef uint32_t (*cirno_layer_unpersist_grad)(MemBuf *buf, CirnoLayer *layer);

/* ------------------------------------------------------------------------ */
/* LAYER_FUNCTIONS */
/* ------------------------------------------------------------------------ */

static cirno_layer_free cirnoFreeFuncs[kCirnoLayerEnd] = {0,};
static cirno_layer_forward cirnoForwardFuncs[kCirnoLayerEnd] = {0,};
static cirno_layer_backward cirnoBackwardFuncs[kCirnoLayerEnd] = {0,};
static cirno_layer_clear_rin cirnoClearRinFuncs[kCirnoLayerEnd] = {0,};
static cirno_layer_apply_grad cirnoApplyGradFuncs[kCirnoLayerEnd] = {0,};
static cirno_layer_persist cirnoPersistFuncs[kCirnoLayerEnd] = {0,};
static cirno_layer_unpersist cirnoUnpersistFuncs[kCirnoLayerEnd] = {0,};
static cirno_layer_persist_grad cirnoPersistGradFuncs[kCirnoLayerEnd] = {0,};
static cirno_layer_unpersist_grad cirnoUnpersistGradFuncs[kCirnoLayerEnd] = {0,};

/* ------------------------------------------------------------------------ */
/* FUNCTION_DECLARATIONS */
/* ------------------------------------------------------------------------ */

static CirnoVol *cirno_vol_new(uint32_t sx, uint32_t sy, uint32_t sz,
    uint32_t depth, uint8_t needGxsum, uint32_t nSnapshot);
static CirnoVol *cirno_random_vol_new(uint32_t sx, uint32_t sy, uint32_t sz,
    uint32_t depth, uint32_t scaleFactor, uint8_t needGxsum, uint32_t nSnapshot);
static void cirno_vol_free(CirnoVol *vol);
static void cirno_vol_persist(MemBuf *buf, CirnoVol *vol, uint8_t persistw);
static CirnoVol *cirno_vol_unpersist(MemBuf *buf, uint8_t unpersistw);
static void cirno_vol_persist_grad(MemBuf *buf, CirnoVol *vol);
static uint32_t cirno_vol_unpersist_grad(MemBuf *buf, CirnoVol *vol);
static void cirno_vol_next_snapshot(CirnoVol *vol);
static double *cirno_vol_next_w(CirnoVol *vol, uint32_t t);
static double *cirno_vol_prev_w(CirnoVol *vol, uint32_t t);
static double *cirno_vol_prev_dw(CirnoVol *vol, uint32_t t);

static CirnoVol *cirno_input_new(lua_State *L, int index, CirnoVol *prevVol,
    CirnoLayerOption *option, int newIndex, uint32_t nSnapshot);
static void cirno_input_free(CirnoLayer *layer);
static void cirno_input_forward(CirnoLayer *layer, uint8_t isTraining);
static void cirno_input_backward(CirnoLayer *layer, uint32_t t, uint8_t resetHidden);
static void cirno_input_clear_rin(CirnoLayer *layer);
static double cirno_input_apply_grad(CirnoLayer *layer, CirnoTrainerOption *option);
static void cirno_input_persist(MemBuf *buf, CirnoLayer *layer);
static CirnoVol *cirno_input_unpersist(lua_State *L, int index, CirnoVol *prevVol,
    MemBuf *buf, int newIndex);
static void cirno_input_persist_grad(MemBuf *buf, CirnoLayer *layer);
static uint32_t cirno_input_unpersist_grad(MemBuf *buf, CirnoLayer *layer);

static CirnoVol *cirno_regression_new(lua_State *L, int index, CirnoVol *prevVol,
    CirnoLayerOption *option, int newIndex, uint32_t nSnapshot);
static void cirno_regression_free(CirnoLayer *layer);
static void cirno_regression_forward(CirnoLayer *layer, uint8_t isTraining);
static void cirno_regression_backward(CirnoLayer *layer, uint32_t t, uint8_t resetHidden);
static void cirno_regression_clear_rin(CirnoLayer *layer);
static double cirno_regression_apply_grad(CirnoLayer *layer, CirnoTrainerOption *option);
static void cirno_regression_persist(MemBuf *buf, CirnoLayer *layer);
static CirnoVol *cirno_regression_unpersist(lua_State *L, int index, CirnoVol *prevVol,
    MemBuf *buf, int newIndex);
static void cirno_regression_persist_grad(MemBuf *buf, CirnoLayer *layer);
static uint32_t cirno_regression_unpersist_grad(MemBuf *buf, CirnoLayer *layer);

static CirnoVol *cirno_softmax_new(lua_State *L, int index, CirnoVol *prevVol,
    CirnoLayerOption *option, int newIndex, uint32_t nSnapshot);
static void cirno_softmax_free(CirnoLayer *layer);
static void cirno_softmax_forward(CirnoLayer *layer, uint8_t isTraining);
static void cirno_softmax_backward(CirnoLayer *layer, uint32_t t, uint8_t resetHidden);
static void cirno_softmax_clear_rin(CirnoLayer *layer);
static double cirno_softmax_apply_grad(CirnoLayer *layer, CirnoTrainerOption *option);
static void cirno_softmax_persist(MemBuf *buf, CirnoLayer *layer);
static CirnoVol *cirno_softmax_unpersist(lua_State *L, int index, CirnoVol *prevVol,
    MemBuf *buf, int newIndex);
static void cirno_softmax_persist_grad(MemBuf *buf, CirnoLayer *layer);
static uint32_t cirno_softmax_unpersist_grad(MemBuf *buf, CirnoLayer *layer);

static CirnoVol *cirno_svm_new(lua_State *L, int index, CirnoVol *prevVol,
    CirnoLayerOption *option, int newIndex, uint32_t nSnapshot);
static void cirno_svm_free(CirnoLayer *layer);
static void cirno_svm_forward(CirnoLayer *layer, uint8_t isTraining);
static void cirno_svm_backward(CirnoLayer *layer, uint32_t t, uint8_t resetHidden);
static void cirno_svm_clear_rin(CirnoLayer *layer);
static double cirno_svm_apply_grad(CirnoLayer *layer, CirnoTrainerOption *option);
static void cirno_svm_persist(MemBuf *buf, CirnoLayer *layer);
static CirnoVol *cirno_svm_unpersist(lua_State *L, int index, CirnoVol *prevVol,
    MemBuf *buf, int newIndex);
static void cirno_svm_persist_grad(MemBuf *buf, CirnoLayer *layer);
static uint32_t cirno_svm_unpersist_grad(MemBuf *buf, CirnoLayer *layer);

static CirnoVol *cirno_pad_new(lua_State *L, int index, CirnoVol *prevVol,
    CirnoLayerOption *option, int newIndex, uint32_t nSnapshot);
static void cirno_pad_free(CirnoLayer *layer);
static void cirno_pad_forward(CirnoLayer *layer, uint8_t isTraining);
static void cirno_pad_backward(CirnoLayer *layer, uint32_t t, uint8_t resetHidden);
static void cirno_pad_clear_rin(CirnoLayer *layer);
static double cirno_pad_apply_grad(CirnoLayer *layer, CirnoTrainerOption *option);
static void cirno_pad_persist(MemBuf *buf, CirnoLayer *layer);
static CirnoVol *cirno_pad_unpersist(lua_State *L, int index, CirnoVol *prevVol,
    MemBuf *buf, int newIndex);
static void cirno_pad_persist_grad(MemBuf *buf, CirnoLayer *layer);
static uint32_t cirno_pad_unpersist_grad(MemBuf *buf, CirnoLayer *layer);

static CirnoVol *cirno_pool_new(lua_State *L, int index, CirnoVol *prevVol,
    CirnoLayerOption *option, int newIndex, uint32_t nSnapshot);
static void cirno_pool_free(CirnoLayer *layer);
static void cirno_pool_forward(CirnoLayer *layer, uint8_t isTraining);
static void cirno_pool_backward(CirnoLayer *layer, uint32_t t, uint8_t resetHidden);
static void cirno_pool_clear_rin(CirnoLayer *layer);
static double cirno_pool_apply_grad(CirnoLayer *layer, CirnoTrainerOption *option);
static void cirno_pool_persist(MemBuf *buf, CirnoLayer *layer);
static CirnoVol *cirno_pool_unpersist(lua_State *L, int index, CirnoVol *prevVol,
    MemBuf *buf, int newIndex);
static void cirno_pool_persist_grad(MemBuf *buf, CirnoLayer *layer);
static uint32_t cirno_pool_unpersist_grad(MemBuf *buf, CirnoLayer *layer);

static CirnoVol *cirno_stride_new(lua_State *L, int index, CirnoVol *prevVol,
    CirnoLayerOption *option, int newIndex, uint32_t nSnapshot);
static void cirno_stride_free(CirnoLayer *layer);
static void cirno_stride_forward(CirnoLayer *layer, uint8_t isTraining);
static void cirno_stride_backward(CirnoLayer *layer, uint32_t t, uint8_t resetHidden);
static void cirno_stride_clear_rin(CirnoLayer *layer);
static double cirno_stride_apply_grad(CirnoLayer *layer, CirnoTrainerOption *option);
static void cirno_stride_persist(MemBuf *buf, CirnoLayer *layer);
static CirnoVol *cirno_stride_unpersist(lua_State *L, int index, CirnoVol *prevVol,
    MemBuf *buf, int newIndex);
static void cirno_stride_persist_grad(MemBuf *buf, CirnoLayer *layer);
static uint32_t cirno_stride_unpersist_grad(MemBuf *buf, CirnoLayer *layer);

static CirnoVol *cirno_conv_new(lua_State *L, int index, CirnoVol *prevVol,
    CirnoLayerOption *option, int newIndex, uint32_t nSnapshot);
static void cirno_conv_free(CirnoLayer *layer);
static void cirno_conv_forward(CirnoLayer *layer, uint8_t isTraining);
static void cirno_conv_backward(CirnoLayer *layer, uint32_t t, uint8_t resetHidden);
static void cirno_conv_clear_rin(CirnoLayer *layer);
static double cirno_conv_apply_grad(CirnoLayer *layer, CirnoTrainerOption *option);
static void cirno_conv_persist(MemBuf *buf, CirnoLayer *layer);
static CirnoVol *cirno_conv_unpersist(lua_State *L, int index, CirnoVol *prevVol,
    MemBuf *buf, int newIndex);
static void cirno_conv_persist_grad(MemBuf *buf, CirnoLayer *layer);
static uint32_t cirno_conv_unpersist_grad(MemBuf *buf, CirnoLayer *layer);

static CirnoVol *cirno_fullyconn_new(lua_State *L, int index, CirnoVol *prevVol,
    CirnoLayerOption *option, int newIndex, uint32_t nSnapshot);
static void cirno_fullyconn_free(CirnoLayer *layer);
static void cirno_fullyconn_forward(CirnoLayer *layer, uint8_t isTraining);
static void cirno_fullyconn_backward(CirnoLayer *layer, uint32_t t, uint8_t resetHidden);
static void cirno_fullyconn_clear_rin(CirnoLayer *layer);
static double cirno_fullyconn_apply_grad(CirnoLayer *layer, CirnoTrainerOption *option);
static void cirno_fullyconn_persist(MemBuf *buf, CirnoLayer *layer);
static CirnoVol *cirno_fullyconn_unpersist(lua_State *L, int index, CirnoVol *prevVol,
    MemBuf *buf, int newIndex);
static void cirno_fullyconn_persist_grad(MemBuf *buf, CirnoLayer *layer);
static uint32_t cirno_fullyconn_unpersist_grad(MemBuf *buf, CirnoLayer *layer);

static CirnoVol *cirno_dropout_new(lua_State *L, int index, CirnoVol *prevVol,
    CirnoLayerOption *option, int newIndex, uint32_t nSnapshot);
static void cirno_dropout_free(CirnoLayer *layer);
static void cirno_dropout_forward(CirnoLayer *layer, uint8_t isTraining);
static void cirno_dropout_backward(CirnoLayer *layer, uint32_t t, uint8_t resetHidden);
static void cirno_dropout_clear_rin(CirnoLayer *layer);
static double cirno_dropout_apply_grad(CirnoLayer *layer, CirnoTrainerOption *option);
static void cirno_dropout_persist(MemBuf *buf, CirnoLayer *layer);
static CirnoVol *cirno_dropout_unpersist(lua_State *L, int index, CirnoVol *prevVol,
    MemBuf *buf, int newIndex);
static void cirno_dropout_persist_grad(MemBuf *buf, CirnoLayer *layer);
static uint32_t cirno_dropout_unpersist_grad(MemBuf *buf, CirnoLayer *layer);

static CirnoVol *cirno_tanh_new(lua_State *L, int index, CirnoVol *prevVol,
    CirnoLayerOption *option, int newIndex, uint32_t nSnapshot);
static void cirno_tanh_free(CirnoLayer *layer);
static void cirno_tanh_forward(CirnoLayer *layer, uint8_t isTraining);
static void cirno_tanh_backward(CirnoLayer *layer, uint32_t t, uint8_t resetHidden);
static void cirno_tanh_clear_rin(CirnoLayer *layer);
static double cirno_tanh_apply_grad(CirnoLayer *layer, CirnoTrainerOption *option);
static void cirno_tanh_persist(MemBuf *buf, CirnoLayer *layer);
static CirnoVol *cirno_tanh_unpersist(lua_State *L, int index, CirnoVol *prevVol,
    MemBuf *buf, int newIndex);
static void cirno_tanh_persist_grad(MemBuf *buf, CirnoLayer *layer);
static uint32_t cirno_tanh_unpersist_grad(MemBuf *buf, CirnoLayer *layer);

static CirnoVol *cirno_sigmoid_new(lua_State *L, int index, CirnoVol *prevVol,
    CirnoLayerOption *option, int newIndex, uint32_t nSnapshot);
static void cirno_sigmoid_free(CirnoLayer *layer);
static void cirno_sigmoid_forward(CirnoLayer *layer, uint8_t isTraining);
static void cirno_sigmoid_backward(CirnoLayer *layer, uint32_t t, uint8_t resetHidden);
static void cirno_sigmoid_clear_rin(CirnoLayer *layer);
static double cirno_sigmoid_apply_grad(CirnoLayer *layer, CirnoTrainerOption *option);
static void cirno_sigmoid_persist(MemBuf *buf, CirnoLayer *layer);
static CirnoVol *cirno_sigmoid_unpersist(lua_State *L, int index, CirnoVol *prevVol,
    MemBuf *buf, int newIndex);
static void cirno_sigmoid_persist_grad(MemBuf *buf, CirnoLayer *layer);
static uint32_t cirno_sigmoid_unpersist_grad(MemBuf *buf, CirnoLayer *layer);

static CirnoVol *cirno_relu_new(lua_State *L, int index, CirnoVol *prevVol,
    CirnoLayerOption *option, int newIndex, uint32_t nSnapshot);
static void cirno_relu_free(CirnoLayer *layer);
static void cirno_relu_forward(CirnoLayer *layer, uint8_t isTraining);
static void cirno_relu_backward(CirnoLayer *layer, uint32_t t, uint8_t resetHidden);
static void cirno_relu_clear_rin(CirnoLayer *layer);
static double cirno_relu_apply_grad(CirnoLayer *layer, CirnoTrainerOption *option);
static void cirno_relu_persist(MemBuf *buf, CirnoLayer *layer);
static CirnoVol *cirno_relu_unpersist(lua_State *L, int index, CirnoVol *prevVol,
    MemBuf *buf, int newIndex);
static void cirno_relu_persist_grad(MemBuf *buf, CirnoLayer *layer);
static uint32_t cirno_relu_unpersist_grad(MemBuf *buf, CirnoLayer *layer);

static CirnoVol *cirno_lstm_new(lua_State *L, int index, CirnoVol *prevVol,
    CirnoLayerOption *option, int newIndex, uint32_t nSnapshot);
static void cirno_lstm_free(CirnoLayer *layer);
static void cirno_lstm_forward(CirnoLayer *layer, uint8_t isTraining);
static void cirno_lstm_backward(CirnoLayer *layer, uint32_t t, uint8_t resetHidden);
static void cirno_lstm_clear_rin(CirnoLayer *layer);
static double cirno_lstm_apply_grad(CirnoLayer *layer, CirnoTrainerOption *option);
static void cirno_lstm_persist(MemBuf *buf, CirnoLayer *layer);
static CirnoVol *cirno_lstm_unpersist(lua_State *L, int index, CirnoVol *prevVol,
    MemBuf *buf, int newIndex);
static void cirno_lstm_persist_grad(MemBuf *buf, CirnoLayer *layer);
static uint32_t cirno_lstm_unpersist_grad(MemBuf *buf, CirnoLayer *layer);

static CirnoVol *cirno_mdpagent_new(lua_State *L, int index, CirnoVol *prevVol,
    CirnoLayerOption *option, int newIndex, uint32_t nSnapshot);
static void cirno_mdpagent_free(CirnoLayer *layer);
static void cirno_mdpagent_forward(CirnoLayer *layer, uint8_t isTraining);
static void cirno_mdpagent_backward(CirnoLayer *layer, uint32_t t, uint8_t resetHidden);
static void cirno_mdpagent_clear_rin(CirnoLayer *layer);
static double cirno_mdpagent_apply_grad(CirnoLayer *layer, CirnoTrainerOption *option);
static void cirno_mdpagent_persist(MemBuf *buf, CirnoLayer *layer);
static CirnoVol *cirno_mdpagent_unpersist(lua_State *L, int index, CirnoVol *prevVol,
    MemBuf *buf, int newIndex);
static void cirno_mdpagent_persist_grad(MemBuf *buf, CirnoLayer *layer);
static uint32_t cirno_mdpagent_unpersist_grad(MemBuf *buf, CirnoLayer *layer);

static CirnoVol *cirno_recurrent_new(lua_State *L, int index, CirnoVol *prevVol,
    CirnoLayerOption *option, int newIndex, uint32_t nSnapshot);
static void cirno_recurrent_free(CirnoLayer *layer);
static void cirno_recurrent_forward(CirnoLayer *layer, uint8_t isTraining);
static void cirno_recurrent_backward(CirnoLayer *layer, uint32_t t, uint8_t resetHidden);
static void cirno_recurrent_clear_rin(CirnoLayer *layer);
static double cirno_recurrent_apply_grad(CirnoLayer *layer, CirnoTrainerOption *option);
static void cirno_recurrent_persist(MemBuf *buf, CirnoLayer *layer);
static CirnoVol *cirno_recurrent_unpersist(lua_State *L, int index, CirnoVol *prevVol,
    MemBuf *buf, int newIndex);
static void cirno_recurrent_persist_grad(MemBuf *buf, CirnoLayer *layer);
static uint32_t cirno_recurrent_unpersist_grad(MemBuf *buf, CirnoLayer *layer);

static CirnoVol *cirno_noise_new(lua_State *L, int index, CirnoVol *prevVol,
    CirnoLayerOption *option, int newIndex, uint32_t nSnapshot);
static void cirno_noise_free(CirnoLayer *layer);
static void cirno_noise_forward(CirnoLayer *layer, uint8_t isTraining);
static void cirno_noise_backward(CirnoLayer *layer, uint32_t t, uint8_t resetHidden);
static void cirno_noise_clear_rin(CirnoLayer *layer);
static double cirno_noise_apply_grad(CirnoLayer *layer, CirnoTrainerOption *option);
static void cirno_noise_persist(MemBuf *buf, CirnoLayer *layer);
static CirnoVol *cirno_noise_unpersist(lua_State *L, int index, CirnoVol *prevVol,
    MemBuf *buf, int newIndex);
static void cirno_noise_persist_grad(MemBuf *buf, CirnoLayer *layer);
static uint32_t cirno_noise_unpersist_grad(MemBuf *buf, CirnoLayer *layer);

static int cirno_is_layer_option(lua_State *L, int index);
static int cirno_layer_option_new(lua_State *L);
static int cirno_is_trainer_option(lua_State *L, int index);
static int cirno_trainer_option_new(lua_State *L);
static void cirno_trainer_option_persist(MemBuf *buf, CirnoTrainerOption *option);
static int cirno_trainer_option_unpersist(lua_State *L, MemBuf *buf);
static double cirno_trainer_apply(CirnoTrainerOption *option, CirnoVol *vol,
    double l1DecayMul, double l2DecayMul);

static int cirno_layer_gc(lua_State *L);
static int cirno_is_network(lua_State *L, int index);
static int cirno_network_new(lua_State *L);
static CirnoVol *cirno_network_forward(lua_State *L,
    int networkIndex, int inputIndex, uint8_t isTraining,
    uint32_t *lastLayerType, uint32_t *action);
static double cirno_network_backward_with_desired(lua_State *L,
    CirnoTrainerOption *option, int networkIndex, int desiredIndex);
static double cirno_network_backward_with_class(lua_State *L,
    CirnoTrainerOption *option, int networkIndex, uint32_t classNo);
static double cirno_network_backward_with_reward(lua_State *L,
    CirnoTrainerOption *option, int networkIndex, double reward);
static double cirno_network_backward_with_index_value(lua_State *L,
    CirnoTrainerOption *option, int networkIndex, uint32_t index, double value);
static double cirno_network_apply_grad_inner(lua_State *L,
    CirnoTrainerOption *option, int networkIndex);
static int cirno_network_predict(lua_State *L);
static int cirno_network_train(lua_State *L);
static int cirno_network_clear_rin(lua_State *L);
static int cirno_network_persist(lua_State *L);
static int cirno_network_unpersist(lua_State *L);
static int cirno_network_persist_grad(lua_State *L);
static int cirno_network_unpersist_grad(lua_State *L);
static int cirno_network_apply_grad(lua_State *L);

/* ------------------------------------------------------------------------ */
/* VOL */
/* ------------------------------------------------------------------------ */

static CirnoVol *cirno_vol_new(uint32_t sx, uint32_t sy, uint32_t sz,
    uint32_t depth, uint8_t needGxsum, uint32_t nSnapshot)
{
    CirnoVol *vol = malloc(sizeof(CirnoVol));

    uint32_t n = sx * sy * sz * depth;

    vol->sx = sx;
    vol->sy = sy;
    vol->sz = sz;
    vol->depth = depth;
    vol->n = n;
    vol->nSnapshot = nSnapshot;
    vol->wBase = malloc(sizeof(double) * n * nSnapshot);
    vol->dwBase = malloc(sizeof(double) * n * nSnapshot);
    vol->w = vol->wBase;
    vol->dw = vol->dwBase;
    vol->curSnapshot = 0;
    vol->k = 0;

    if (needGxsum) {
        vol->useGxsum = TRUE;
        vol->gsum = malloc(sizeof(double) * n);
        vol->xsum = malloc(sizeof(double) * n);

        memset(vol->gsum, 0, sizeof(double) * n);
        memset(vol->xsum, 0, sizeof(double) * n);
    }
    else {
        vol->useGxsum = FALSE;
        vol->gsum = NULL;
        vol->xsum = NULL;
    }

    memset(vol->wBase, 0, sizeof(double) * (n * nSnapshot));
    memset(vol->dwBase, 0, sizeof(double) * (n * nSnapshot));

    return vol;
}

static CirnoVol *cirno_random_vol_new(uint32_t sx, uint32_t sy, uint32_t sz,
    uint32_t depth, uint32_t scaleFactor, uint8_t needGxsum, uint32_t nSnapshot)
{
    CirnoVol *vol = malloc(sizeof(CirnoVol));

    uint32_t n = sx * sy * sz * depth;
    uint32_t i = 0;
    double scale = sqrt(1.0 / scaleFactor);

    vol->sx = sx;
    vol->sy = sy;
    vol->sz = sz;
    vol->depth = depth;
    vol->n = n;
    vol->nSnapshot = nSnapshot;
    vol->wBase = malloc(sizeof(double) * n * nSnapshot);
    vol->dwBase = malloc(sizeof(double) * n * nSnapshot);
    vol->w = vol->wBase;
    vol->dw = vol->dwBase;
    vol->curSnapshot = 0;
    vol->k = 0;

    if (needGxsum) {
        vol->useGxsum = TRUE;
        vol->gsum = malloc(sizeof(double) * n);
        vol->xsum = malloc(sizeof(double) * n);

        memset(vol->gsum, 0, sizeof(double) * n);
        memset(vol->xsum, 0, sizeof(double) * n);
    }
    else {
        vol->useGxsum = FALSE;
        vol->gsum = NULL;
        vol->xsum = NULL;
    }

    for (i = 0; i < n; ++i) {
        vol->w[i] = rnorm() * scale;
    }

    memset(vol->dw, 0, sizeof(double) * (n * nSnapshot));

    return vol;
}

static void cirno_vol_free(CirnoVol *vol)
{
    if (vol->wBase != NULL) {
        free(vol->wBase);
        vol->wBase = NULL;
    }

    if (vol->dwBase != NULL) {
        free(vol->dwBase);
        vol->dwBase = NULL;
    }

    if (vol->gsum != NULL) {
        free(vol->gsum);
        vol->gsum = NULL;
    }

    if (vol->xsum != NULL) {
        free(vol->xsum);
        vol->xsum = NULL;
    }

    free(vol);
}

static void cirno_vol_persist(MemBuf *buf, CirnoVol *vol, uint8_t persistw)
{
    uint32_t i = 0;
    uint32_t n = vol->n;
    double *w = vol->w;

    mem_buf_write_uint32(buf, vol->sx);
    mem_buf_write_uint32(buf, vol->sy);
    mem_buf_write_uint32(buf, vol->sz);
    mem_buf_write_uint32(buf, vol->depth);
    mem_buf_write_uint32(buf, vol->useGxsum);
    mem_buf_write_uint32(buf, vol->nSnapshot);

    if (persistw) {
        for (i = 0; i < n; ++i) {
            mem_buf_write_double(buf, *(w++));
        }
    }
}

static CirnoVol *cirno_vol_unpersist(MemBuf *buf, uint8_t unpersistw)
{
    uint32_t i = 0;
    uint32_t n = 0;
    double *w = NULL;
    CirnoVol *vol = malloc(sizeof(CirnoVol));

    vol->sx = mem_buf_read_uint32(buf);
    vol->sy = mem_buf_read_uint32(buf);
    vol->sz = mem_buf_read_uint32(buf);
    vol->depth = mem_buf_read_uint32(buf);
    vol->useGxsum = mem_buf_read_uint32(buf);
    vol->nSnapshot = mem_buf_read_uint32(buf);

    n = vol->sx * vol->sy * vol->sz * vol->depth;

    vol->n = n;

    vol->wBase = malloc(sizeof(double) * n * vol->nSnapshot);
    vol->dwBase = malloc(sizeof(double) * n * vol->nSnapshot);

    vol->w = vol->wBase;
    vol->dw = vol->dwBase;

    w = vol->w;

    if (vol->useGxsum) {
        vol->gsum = malloc(sizeof(double) * n);
        vol->xsum = malloc(sizeof(double) * n);

        memset(vol->gsum, 0, sizeof(double) * n);
        memset(vol->xsum, 0, sizeof(double) * n);
    }
    else {
        vol->gsum = NULL;
        vol->xsum = NULL;
    }

    if (unpersistw) {
        for (i = 0; i < n; ++i) {
            *(w++) = mem_buf_read_double(buf);
        }
    }
    else {
        memset(vol->wBase, 0, sizeof(double) * (n * vol->nSnapshot));
    }

    memset(vol->dwBase, 0, sizeof(double) * (n * vol->nSnapshot));

    vol->curSnapshot = 0;
    vol->k = 0;

    return vol;
}

static void cirno_vol_persist_grad(MemBuf *buf, CirnoVol *vol)
{
    uint32_t i = 0;
    uint32_t n = vol->n;
    double *dw = vol->dw;

    mem_buf_write_uint32(buf, n);
    mem_buf_write_uint32(buf, vol->k);

    for (i = 0; i < n; ++i) {
        mem_buf_write_double(buf, *(dw++));
    }

    vol->k = 0;
    memset(vol->dw, 0, sizeof(double) * n);
}

static uint32_t cirno_vol_unpersist_grad(MemBuf *buf, CirnoVol *vol)
{
    uint32_t i = 0;
    uint32_t bufn = 0;
    uint32_t result = FALSE;
    uint32_t n = vol->n;
    double *dw = vol->dw;

    bufn = mem_buf_read_uint32(buf);
    vol->k += mem_buf_read_uint32(buf);

    if (n == bufn) {
        for (i = 0; i < n; ++i) {
            *(dw++) += mem_buf_read_double(buf);
        }

        result = TRUE;
    }

    return result;
}

static void cirno_vol_next_snapshot(CirnoVol *vol)
{
    uint32_t curSnapshot = vol->curSnapshot;
    uint32_t n = vol->n;

    curSnapshot = (curSnapshot + 1) % vol->nSnapshot;

    vol->curSnapshot = curSnapshot;

    vol->w = vol->wBase + (n * curSnapshot);
    vol->dw = vol->dwBase + (n * curSnapshot);
}

static double *cirno_vol_next_w(CirnoVol *vol, uint32_t t)
{
    return vol->wBase + (vol->n *
        ((vol->curSnapshot + t) % vol->nSnapshot));
}

static double *cirno_vol_prev_w(CirnoVol *vol, uint32_t t)
{
    return vol->wBase + (vol->n *
        ((vol->curSnapshot - t + vol->nSnapshot) % vol->nSnapshot));
}

static double *cirno_vol_prev_dw(CirnoVol *vol, uint32_t t)
{
    return vol->dwBase + (vol->n *
        ((vol->curSnapshot - t + vol->nSnapshot) % vol->nSnapshot));
}

/* ------------------------------------------------------------------------ */
/* INPUT_LAYER */
/* ------------------------------------------------------------------------ */

static CirnoVol *cirno_input_new(lua_State *L, int index, CirnoVol *prevVol,
    CirnoLayerOption *option, int newIndex, uint32_t nSnapshot)
{
    UNREFERENCED_PARAMETER(prevVol);

    CirnoLayer *curLayer = lua_newuserdata(L, sizeof(CirnoLayer));
    luaL_getmetatable(L, "cirno.layer_instance");
    lua_setmetatable(L, -2);

    curLayer->type = kCirnoLayerInput;
    curLayer->inVol = NULL;
    curLayer->outVol = cirno_vol_new(option->sx,
        option->sy, option->sz,
        option->depth, FALSE, nSnapshot);

    lua_rawseti(L, index - 1, newIndex);

    return curLayer->outVol;
}

static void cirno_input_free(CirnoLayer *layer)
{
    cirno_vol_free(layer->outVol);
}

static void cirno_input_forward(CirnoLayer *layer, uint8_t isTraining)
{
    UNREFERENCED_PARAMETER(layer);
    UNREFERENCED_PARAMETER(isTraining);

    return;
}

static void cirno_input_backward(CirnoLayer *layer, uint32_t t, uint8_t resetHidden)
{
    UNREFERENCED_PARAMETER(layer);
    UNREFERENCED_PARAMETER(t);
    UNREFERENCED_PARAMETER(resetHidden);

    return;
}

static void cirno_input_clear_rin(CirnoLayer *layer)
{
    UNREFERENCED_PARAMETER(layer);

    return;
}

static double cirno_input_apply_grad(CirnoLayer *layer, CirnoTrainerOption *option)
{
    UNREFERENCED_PARAMETER(layer);
    UNREFERENCED_PARAMETER(option);

    return 0.0;
}

static void cirno_input_persist(MemBuf *buf, CirnoLayer *layer)
{
    mem_buf_write_uint32(buf, layer->type);
    cirno_vol_persist(buf, layer->outVol, FALSE);
}

static CirnoVol *cirno_input_unpersist(lua_State *L, int index, CirnoVol *prevVol,
    MemBuf *buf, int newIndex)
{
    UNREFERENCED_PARAMETER(prevVol);

    CirnoLayer *curLayer = lua_newuserdata(L, sizeof(CirnoLayer));
    luaL_getmetatable(L, "cirno.layer_instance");
    lua_setmetatable(L, -2);

    curLayer->type = kCirnoLayerInput;
    curLayer->inVol = NULL;
    curLayer->outVol = cirno_vol_unpersist(buf, FALSE);

    lua_rawseti(L, index - 1, newIndex);

    return curLayer->outVol;
}

static void cirno_input_persist_grad(MemBuf *buf, CirnoLayer *layer)
{
    UNREFERENCED_PARAMETER(buf);
    UNREFERENCED_PARAMETER(layer);

    return;
}

static uint32_t cirno_input_unpersist_grad(MemBuf *buf, CirnoLayer *layer)
{
    UNREFERENCED_PARAMETER(buf);
    UNREFERENCED_PARAMETER(layer);

    return TRUE;
}

/* ------------------------------------------------------------------------ */
/* REGRESSION_LAYER */
/* ------------------------------------------------------------------------ */

static CirnoVol *cirno_regression_new(lua_State *L, int index, CirnoVol *prevVol,
    CirnoLayerOption *option, int newIndex, uint32_t nSnapshot)
{
    UNREFERENCED_PARAMETER(option);
    UNREFERENCED_PARAMETER(nSnapshot);

    CirnoLayer *curLayer = lua_newuserdata(L, sizeof(CirnoLayer));
    luaL_getmetatable(L, "cirno.layer_instance");
    lua_setmetatable(L, -2);

    curLayer->type = kCirnoLayerRegression;
    curLayer->inVol = prevVol;
    curLayer->outVol = NULL;

    lua_rawseti(L, index - 1, newIndex);

    return curLayer->outVol;
}

static void cirno_regression_free(CirnoLayer *layer)
{
    UNREFERENCED_PARAMETER(layer);

    return;
}

static void cirno_regression_forward(CirnoLayer *layer, uint8_t isTraining)
{
    UNREFERENCED_PARAMETER(layer);
    UNREFERENCED_PARAMETER(isTraining);

    return;
}

static void cirno_regression_backward(CirnoLayer *layer, uint32_t t, uint8_t resetHidden)
{
    UNREFERENCED_PARAMETER(layer);
    UNREFERENCED_PARAMETER(t);
    UNREFERENCED_PARAMETER(resetHidden);

    return;
}

static void cirno_regression_clear_rin(CirnoLayer *layer)
{
    UNREFERENCED_PARAMETER(layer);

    return;
}

static double cirno_regression_apply_grad(CirnoLayer *layer, CirnoTrainerOption *option)
{
    UNREFERENCED_PARAMETER(layer);
    UNREFERENCED_PARAMETER(option);

    return 0.0;
}

static void cirno_regression_persist(MemBuf *buf, CirnoLayer *layer)
{
    mem_buf_write_uint32(buf, layer->type);
}

static CirnoVol *cirno_regression_unpersist(lua_State *L, int index, CirnoVol *prevVol,
    MemBuf *buf, int newIndex)
{
    UNREFERENCED_PARAMETER(buf);

    CirnoLayer *curLayer = lua_newuserdata(L, sizeof(CirnoLayer));
    luaL_getmetatable(L, "cirno.layer_instance");
    lua_setmetatable(L, -2);

    curLayer->type = kCirnoLayerRegression;
    curLayer->inVol = prevVol;
    curLayer->outVol = NULL;

    lua_rawseti(L, index - 1, newIndex);

    return curLayer->outVol;
}

static void cirno_regression_persist_grad(MemBuf *buf, CirnoLayer *layer)
{
    UNREFERENCED_PARAMETER(buf);
    UNREFERENCED_PARAMETER(layer);

    return;
}

static uint32_t cirno_regression_unpersist_grad(MemBuf *buf, CirnoLayer *layer)
{
    UNREFERENCED_PARAMETER(buf);
    UNREFERENCED_PARAMETER(layer);

    return TRUE;
}

/* ------------------------------------------------------------------------ */
/* SOFTMAX_LAYER */
/* ------------------------------------------------------------------------ */

static CirnoVol *cirno_softmax_new(lua_State *L, int index, CirnoVol *prevVol,
    CirnoLayerOption *option, int newIndex, uint32_t nSnapshot)
{
    UNREFERENCED_PARAMETER(nSnapshot);

    CirnoLayer *curLayer = lua_newuserdata(L, sizeof(CirnoLayer));
    luaL_getmetatable(L, "cirno.layer_instance");
    lua_setmetatable(L, -2);

    curLayer->type = kCirnoLayerSoftmax;
    curLayer->inVol = prevVol;
    curLayer->outVol = cirno_vol_new(1, 1, 1, option->out, FALSE, 1);

    lua_rawseti(L, index - 1, newIndex);

    return curLayer->outVol;
}

static void cirno_softmax_free(CirnoLayer *layer)
{
    cirno_vol_free(layer->outVol);
}

static void cirno_softmax_forward(CirnoLayer *layer, uint8_t isTraining)
{
    UNREFERENCED_PARAMETER(isTraining);

    uint32_t i = 0;
    CirnoVol *inVol = layer->inVol;
    CirnoVol *outVol = layer->outVol;
    uint32_t n = inVol->n;
    double *inw = inVol->w;
    double *outw = outVol->w;
    double inputMax = -DBL_MAX, outSum = 0.0, outValue = 0.0;

    for (i = 0; i < n; ++i) {
        if (inputMax < *inw) {
            inputMax = *inw;
        }

        ++inw;
    }

    inw = inVol->w;

    for (i = 0; i < n; ++i) {
        outValue = exp(*(inw++) - inputMax);
        *(outw++) = outValue;
        outSum += outValue;
    }

    outw = outVol->w;

    for (i = 0; i < n; ++i) {
        *(outw++) /= outSum;
    }
}

static void cirno_softmax_backward(CirnoLayer *layer, uint32_t t, uint8_t resetHidden)
{
    UNREFERENCED_PARAMETER(layer);
    UNREFERENCED_PARAMETER(t);
    UNREFERENCED_PARAMETER(resetHidden);

    return;
}

static void cirno_softmax_clear_rin(CirnoLayer *layer)
{
    UNREFERENCED_PARAMETER(layer);

    return;
}

static double cirno_softmax_apply_grad(CirnoLayer *layer, CirnoTrainerOption *option)
{
    UNREFERENCED_PARAMETER(layer);
    UNREFERENCED_PARAMETER(option);

    return 0.0;
}

static void cirno_softmax_persist(MemBuf *buf, CirnoLayer *layer)
{
    mem_buf_write_uint32(buf, layer->type);
    cirno_vol_persist(buf, layer->outVol, FALSE);
}

static CirnoVol *cirno_softmax_unpersist(lua_State *L, int index, CirnoVol *prevVol,
    MemBuf *buf, int newIndex)
{
    CirnoLayer *curLayer = lua_newuserdata(L, sizeof(CirnoLayer));
    luaL_getmetatable(L, "cirno.layer_instance");
    lua_setmetatable(L, -2);

    curLayer->type = kCirnoLayerSoftmax;
    curLayer->inVol = prevVol;
    curLayer->outVol = cirno_vol_unpersist(buf, FALSE);

    lua_rawseti(L, index - 1, newIndex);

    return curLayer->outVol;
}

static void cirno_softmax_persist_grad(MemBuf *buf, CirnoLayer *layer)
{
    UNREFERENCED_PARAMETER(buf);
    UNREFERENCED_PARAMETER(layer);

    return;
}

static uint32_t cirno_softmax_unpersist_grad(MemBuf *buf, CirnoLayer *layer)
{
    UNREFERENCED_PARAMETER(buf);
    UNREFERENCED_PARAMETER(layer);

    return TRUE;
}

/* ------------------------------------------------------------------------ */
/* SVM_LAYER */
/* ------------------------------------------------------------------------ */

static CirnoVol *cirno_svm_new(lua_State *L, int index, CirnoVol *prevVol,
    CirnoLayerOption *option, int newIndex, uint32_t nSnapshot)
{
    UNREFERENCED_PARAMETER(option);
    UNREFERENCED_PARAMETER(nSnapshot);

    CirnoLayer *curLayer = lua_newuserdata(L, sizeof(CirnoLayer));
    luaL_getmetatable(L, "cirno.layer_instance");
    lua_setmetatable(L, -2);

    curLayer->type = kCirnoLayerSvm;
    curLayer->inVol = prevVol;
    curLayer->outVol = NULL;

    lua_rawseti(L, index - 1, newIndex);

    return curLayer->outVol;
}

static void cirno_svm_free(CirnoLayer *layer)
{
    UNREFERENCED_PARAMETER(layer);

    return;
}

static void cirno_svm_forward(CirnoLayer *layer, uint8_t isTraining)
{
    UNREFERENCED_PARAMETER(layer);
    UNREFERENCED_PARAMETER(isTraining);

    return;
}

static void cirno_svm_backward(CirnoLayer *layer, uint32_t t, uint8_t resetHidden)
{
    UNREFERENCED_PARAMETER(layer);
    UNREFERENCED_PARAMETER(t);
    UNREFERENCED_PARAMETER(resetHidden);

    return;
}

static void cirno_svm_clear_rin(CirnoLayer *layer)
{
    UNREFERENCED_PARAMETER(layer);

    return;
}

static double cirno_svm_apply_grad(CirnoLayer *layer, CirnoTrainerOption *option)
{
    UNREFERENCED_PARAMETER(layer);
    UNREFERENCED_PARAMETER(option);

    return 0.0;
}

static void cirno_svm_persist(MemBuf *buf, CirnoLayer *layer)
{
    mem_buf_write_uint32(buf, layer->type);
}

static CirnoVol *cirno_svm_unpersist(lua_State *L, int index, CirnoVol *prevVol,
    MemBuf *buf, int newIndex)
{
    UNREFERENCED_PARAMETER(buf);

    CirnoLayer *curLayer = lua_newuserdata(L, sizeof(CirnoLayer));
    luaL_getmetatable(L, "cirno.layer_instance");
    lua_setmetatable(L, -2);

    curLayer->type = kCirnoLayerSvm;
    curLayer->inVol = prevVol;
    curLayer->outVol = NULL;

    lua_rawseti(L, index - 1, newIndex);

    return curLayer->outVol;
}

static void cirno_svm_persist_grad(MemBuf *buf, CirnoLayer *layer)
{
    UNREFERENCED_PARAMETER(buf);
    UNREFERENCED_PARAMETER(layer);

    return;
}

static uint32_t cirno_svm_unpersist_grad(MemBuf *buf, CirnoLayer *layer)
{
    UNREFERENCED_PARAMETER(buf);
    UNREFERENCED_PARAMETER(layer);

    return TRUE;
}

/* ------------------------------------------------------------------------ */
/* PAD_LAYER */
/* ------------------------------------------------------------------------ */

static CirnoVol *cirno_pad_new(lua_State *L, int index, CirnoVol *prevVol,
    CirnoLayerOption *option, int newIndex, uint32_t nSnapshot)
{
    uint32_t padx = option->padx;
    uint32_t pady = option->pady;
    uint32_t padz = option->padz;

    CirnoLayer *curLayer = lua_newuserdata(L, sizeof(CirnoLayer));
    luaL_getmetatable(L, "cirno.layer_instance");
    lua_setmetatable(L, -2);

    curLayer->type = kCirnoLayerPad;
    curLayer->inVol = prevVol;
    curLayer->outVol = cirno_vol_new(prevVol->sx + padx * 2,
        prevVol->sy + pady * 2, prevVol->sz + padz * 2, prevVol->depth,
        FALSE, nSnapshot);
    curLayer->info.pad = malloc(sizeof(CirnoLayerInfoPad));
    curLayer->info.pad->padx = padx;
    curLayer->info.pad->pady = pady;
    curLayer->info.pad->padz = padz;

    lua_rawseti(L, index - 1, newIndex);

    return curLayer->outVol;
}

static void cirno_pad_free(CirnoLayer *layer)
{
    cirno_vol_free(layer->outVol);
    free(layer->info.pad);
}

static void cirno_pad_forward(CirnoLayer *layer, uint8_t isTraining)
{
    UNREFERENCED_PARAMETER(isTraining);

    uint32_t insx = layer->inVol->sx;
    uint32_t insy = layer->inVol->sy;
    uint32_t insz = layer->inVol->sz;
    uint32_t indepth = layer->inVol->depth;
    uint32_t outsx = layer->outVol->sx;
    uint32_t outsy = layer->outVol->sy;
    uint32_t sy = 0, sz = 0;
    double *inw = layer->inVol->w;
    double *outw = layer->outVol->w;
    uint32_t padx = layer->info.pad->padx;
    uint32_t pady = layer->info.pad->pady;
    uint32_t padz = layer->info.pad->padz;

    for (sz = 0; sz < insz; ++sz) {
        for (sy = 0; sy < insy; ++sy) {
            memcpy(
                outw + ((((sz + padz) * outsy + sy + pady) * outsx + padx) * indepth),
                inw + ((sz * insy + sy) * insx * indepth),
                sizeof(double) * insx * indepth
            );
        }
    }
}

static void cirno_pad_backward(CirnoLayer *layer, uint32_t t, uint8_t resetHidden)
{
    UNREFERENCED_PARAMETER(resetHidden);

    uint32_t insx = layer->inVol->sx;
    uint32_t insy = layer->inVol->sy;
    uint32_t insz = layer->inVol->sz;
    uint32_t indepth = layer->inVol->depth;
    uint32_t outsx = layer->outVol->sx;
    uint32_t outsy = layer->outVol->sy;
    uint32_t sy = 0, sz = 0;
    double *indw = NULL;
    double *outdw = NULL;
    uint32_t padx = layer->info.pad->padx;
    uint32_t pady = layer->info.pad->pady;
    uint32_t padz = layer->info.pad->padz;

    indw = cirno_vol_prev_dw(layer->inVol, t);
    outdw = cirno_vol_prev_dw(layer->outVol, t);

    for (sz = 0; sz < insz; ++sz) {
        for (sy = 0; sy < insy; ++sy) {
            memcpy(
                indw + ((sz * insy + sy) * insx * indepth),
                outdw + ((((sz + padz) * outsy + sy + pady) * outsx + padx) * indepth),
                sizeof(double) * insx * indepth
            );
        }
    }
}

static void cirno_pad_clear_rin(CirnoLayer *layer)
{
    UNREFERENCED_PARAMETER(layer);

    return;
}

static double cirno_pad_apply_grad(CirnoLayer *layer, CirnoTrainerOption *option)
{
    UNREFERENCED_PARAMETER(layer);
    UNREFERENCED_PARAMETER(option);

    return 0.0;
}

static void cirno_pad_persist(MemBuf *buf, CirnoLayer *layer)
{
    mem_buf_write_uint32(buf, layer->type);
    cirno_vol_persist(buf, layer->outVol, FALSE);
    mem_buf_write_uint32(buf, layer->info.pad->padx);
    mem_buf_write_uint32(buf, layer->info.pad->pady);
    mem_buf_write_uint32(buf, layer->info.pad->padz);
}

static CirnoVol *cirno_pad_unpersist(lua_State *L, int index, CirnoVol *prevVol,
    MemBuf *buf, int newIndex)
{
    uint32_t padx = 0, pady = 0, padz = 0;

    CirnoLayer *curLayer = lua_newuserdata(L, sizeof(CirnoLayer));
    luaL_getmetatable(L, "cirno.layer_instance");
    lua_setmetatable(L, -2);

    curLayer->type = kCirnoLayerPad;
    curLayer->inVol = prevVol;
    curLayer->outVol = cirno_vol_unpersist(buf, FALSE);

    padx = mem_buf_read_uint32(buf);
    pady = mem_buf_read_uint32(buf);
    padz = mem_buf_read_uint32(buf);

    curLayer->info.pad = malloc(sizeof(CirnoLayerInfoPad));
    curLayer->info.pad->padx = padx;
    curLayer->info.pad->pady = pady;
    curLayer->info.pad->padz = padz;

    lua_rawseti(L, index - 1, newIndex);

    return curLayer->outVol;
}

static void cirno_pad_persist_grad(MemBuf *buf, CirnoLayer *layer)
{
    UNREFERENCED_PARAMETER(buf);
    UNREFERENCED_PARAMETER(layer);

    return;
}

static uint32_t cirno_pad_unpersist_grad(MemBuf *buf, CirnoLayer *layer)
{
    UNREFERENCED_PARAMETER(buf);
    UNREFERENCED_PARAMETER(layer);

    return TRUE;
}

/* ------------------------------------------------------------------------ */
/* POOL_LAYER */
/* ------------------------------------------------------------------------ */

static CirnoVol *cirno_pool_new(lua_State *L, int index, CirnoVol *prevVol,
    CirnoLayerOption *option, int newIndex, uint32_t nSnapshot)
{
    uint32_t sx = option->sx;
    uint32_t sy = option->sy;
    uint32_t sz = option->sz;

    uint32_t outsx = prevVol->sx - sx + 1;
    uint32_t outsy = prevVol->sy - sy + 1;
    uint32_t outsz = prevVol->sz - sz + 1;
    uint32_t outdepth = prevVol->depth;

    uint32_t n = 0;

    if (prevVol->sx < sx || prevVol->sy < sy || prevVol->sz < sz) {
        luaL_error(L, "input size cannot be less than pool size");
    }

    CirnoLayer *curLayer = lua_newuserdata(L, sizeof(CirnoLayer));
    luaL_getmetatable(L, "cirno.layer_instance");
    lua_setmetatable(L, -2);

    curLayer->type = kCirnoLayerPool;
    curLayer->inVol = prevVol;
    curLayer->outVol = cirno_vol_new(outsx, outsy, outsz, outdepth, FALSE, nSnapshot);

    n = curLayer->outVol->n;

    curLayer->info.pool = malloc(sizeof(CirnoLayerInfoPool));
    curLayer->info.pool->nSnapshot = nSnapshot;
    curLayer->info.pool->sx = sx;
    curLayer->info.pool->sy = sy;
    curLayer->info.pool->sz = sz;
    curLayer->info.pool->switchxBase = malloc(sizeof(uint32_t) * n * nSnapshot);
    curLayer->info.pool->switchyBase = malloc(sizeof(uint32_t) * n * nSnapshot);
    curLayer->info.pool->switchzBase = malloc(sizeof(uint32_t) * n * nSnapshot);
    curLayer->info.pool->switchx = curLayer->info.pool->switchxBase;
    curLayer->info.pool->switchy = curLayer->info.pool->switchyBase;
    curLayer->info.pool->switchz = curLayer->info.pool->switchzBase;
    curLayer->info.pool->curSnapshot = 0;

    lua_rawseti(L, index - 1, newIndex);

    return curLayer->outVol;
}

static void cirno_pool_free(CirnoLayer *layer)
{
    cirno_vol_free(layer->outVol);
    free(layer->info.pool->switchxBase);
    free(layer->info.pool->switchyBase);
    free(layer->info.pool->switchzBase);
    free(layer->info.pool);
}

static void cirno_pool_forward(CirnoLayer *layer, uint8_t isTraining)
{
    UNREFERENCED_PARAMETER(isTraining);

    uint32_t insx = layer->inVol->sx;
    uint32_t insy = layer->inVol->sy;
    uint32_t indepth = layer->inVol->depth;
    uint32_t outsx = layer->outVol->sx;
    uint32_t outsy = layer->outVol->sy;
    uint32_t outsz = layer->outVol->sz;
    uint32_t outdepth = layer->outVol->depth;
    double val = 0.0, maxval = 0.0;
    uint32_t winx = 0, winy = 0, winz = 0;
    uint32_t sx = layer->info.pool->sx;
    uint32_t sy = layer->info.pool->sy;
    uint32_t sz = layer->info.pool->sz;
    CirnoLayerInfoPool *poolInfo = layer->info.pool;
    uint32_t *switchx = NULL, *switchy = NULL, *switchz = NULL;
    uint32_t d = 0, ox = 0, oy = 0, oz = 0;
    uint32_t fx = 0, fy = 0, fz = 0;
    double *inw = layer->inVol->w;
    double *outw = layer->outVol->w;

    poolInfo->curSnapshot =
        (poolInfo->curSnapshot + 1) % poolInfo->nSnapshot;

    poolInfo->switchx = poolInfo->switchxBase
        + (poolInfo->curSnapshot * layer->outVol->n);
    poolInfo->switchy = poolInfo->switchyBase
        + (poolInfo->curSnapshot * layer->outVol->n);
    poolInfo->switchz = poolInfo->switchzBase
        + (poolInfo->curSnapshot * layer->outVol->n);

    switchx = poolInfo->switchx;
    switchy = poolInfo->switchy;
    switchz = poolInfo->switchz;

    for (d = 0; d < outdepth; ++d) {
        for (oz = 0; oz < outsz; ++oz) {
            for (oy = 0; oy < outsy; ++oy) {
                for (ox = 0; ox < outsx; ++ox) {
                    maxval = -DBL_MAX;
                    winx = 0;
                    winy = 0;
                    winz = 0;

                    for (fz = 0; fz < sz; ++fz) {
                        for (fy = 0; fy < sy; ++fy) {
                            for (fx = 0; fx < sx; ++fx) {
                                val = inw[((((oz + fz) * insy + oy + fy) * insx + ox) * indepth)
                                    + fx * indepth + d];

                                if (val > maxval) {
                                    maxval = val;
                                    winx = fx;
                                    winy = fy;
                                    winz = oz;
                                }
                            }
                        }
                    }

                    *(switchx++) = winx + ox;
                    *(switchy++) = winy + oy;
                    *(switchz++) = winz + oz;

                    outw[((oz * outsy + oy) * outsx + ox) * outdepth + d] = maxval;
                }
            }
        }
    }
}

static void cirno_pool_backward(CirnoLayer *layer, uint32_t t, uint8_t resetHidden)
{
    UNREFERENCED_PARAMETER(resetHidden);

    uint32_t insx = layer->inVol->sx;
    uint32_t insy = layer->inVol->sy;
    uint32_t indepth = layer->inVol->depth;
    uint32_t outsx = layer->outVol->sx;
    uint32_t outsy = layer->outVol->sy;
    uint32_t outsz = layer->outVol->sz;
    uint32_t outdepth = layer->outVol->depth;
    uint32_t *switchx = NULL, *switchy = NULL, *switchz = NULL;
    uint32_t winx = 0, winy = 0, winz = 0;
    uint32_t d = 0, ox = 0, oy = 0, oz = 0;
    double *indw = NULL;
    double *outdw = NULL;
    uint32_t nSnapshot = layer->info.pool->nSnapshot;
    uint32_t curSnapshot = layer->info.pool->curSnapshot;
    uint32_t n = layer->outVol->n;
    uint32_t switchIndex = 0;

    switchIndex = (n * ((curSnapshot - t + nSnapshot) % nSnapshot));

    switchx = layer->info.pool->switchxBase + switchIndex;
    switchy = layer->info.pool->switchyBase + switchIndex;
    switchz = layer->info.pool->switchzBase + switchIndex;

    indw = cirno_vol_prev_dw(layer->inVol, t);
    outdw = cirno_vol_prev_dw(layer->outVol, t);

    memset(indw, 0, sizeof(double) * layer->inVol->n);

    for (d = 0; d < outdepth; ++d) {
        for (oz = 0; oz < outsz; ++oz) {
            for (oy = 0; oy < outsy; ++oy) {
                for (ox = 0; ox < outsx; ++ox) {
                    winx = *(switchx++);
                    winy = *(switchy++);
                    winz = *(switchz++);

                    indw[((winz * insy + winy) * insx + winx) * indepth + d] +=
                        outdw[((oz * outsy + oy) * outsx + ox) * outdepth + d];
                }
            }
        }
    }
}

static void cirno_pool_clear_rin(CirnoLayer *layer)
{
    UNREFERENCED_PARAMETER(layer);

    return;
}

static double cirno_pool_apply_grad(CirnoLayer *layer, CirnoTrainerOption *option)
{
    UNREFERENCED_PARAMETER(layer);
    UNREFERENCED_PARAMETER(option);

    return 0.0;
}

static void cirno_pool_persist(MemBuf *buf, CirnoLayer *layer)
{
    mem_buf_write_uint32(buf, layer->type);
    cirno_vol_persist(buf, layer->outVol, FALSE);
    mem_buf_write_uint32(buf, layer->info.pool->sx);
    mem_buf_write_uint32(buf, layer->info.pool->sy);
    mem_buf_write_uint32(buf, layer->info.pool->sz);
    mem_buf_write_uint32(buf, layer->info.pool->nSnapshot);
}

static CirnoVol *cirno_pool_unpersist(lua_State *L, int index, CirnoVol *prevVol,
    MemBuf *buf, int newIndex)
{
    uint32_t sx = 0, sy = 0, sz = 0;
    uint32_t n = 0, nSnapshot = 0;

    CirnoLayer *curLayer = lua_newuserdata(L, sizeof(CirnoLayer));
    luaL_getmetatable(L, "cirno.layer_instance");
    lua_setmetatable(L, -2);

    curLayer->type = kCirnoLayerPool;
    curLayer->inVol = prevVol;
    curLayer->outVol = cirno_vol_unpersist(buf, FALSE);

    sx = mem_buf_read_uint32(buf);
    sy = mem_buf_read_uint32(buf);
    sz = mem_buf_read_uint32(buf);

    nSnapshot = mem_buf_read_uint32(buf);

    n = curLayer->outVol->n;

    curLayer->info.pool = malloc(sizeof(CirnoLayerInfoPool));
    curLayer->info.pool->nSnapshot = nSnapshot;
    curLayer->info.pool->sx = sx;
    curLayer->info.pool->sy = sy;
    curLayer->info.pool->sz = sz;
    curLayer->info.pool->switchxBase = malloc(sizeof(uint32_t) * n * nSnapshot);
    curLayer->info.pool->switchyBase = malloc(sizeof(uint32_t) * n * nSnapshot);
    curLayer->info.pool->switchzBase = malloc(sizeof(uint32_t) * n * nSnapshot);
    curLayer->info.pool->switchx = curLayer->info.pool->switchxBase;
    curLayer->info.pool->switchy = curLayer->info.pool->switchyBase;
    curLayer->info.pool->switchz = curLayer->info.pool->switchzBase;
    curLayer->info.pool->curSnapshot = 0;

    lua_rawseti(L, index - 1, newIndex);

    return curLayer->outVol;
}

static void cirno_pool_persist_grad(MemBuf *buf, CirnoLayer *layer)
{
    UNREFERENCED_PARAMETER(buf);
    UNREFERENCED_PARAMETER(layer);

    return;
}

static uint32_t cirno_pool_unpersist_grad(MemBuf *buf, CirnoLayer *layer)
{
    UNREFERENCED_PARAMETER(buf);
    UNREFERENCED_PARAMETER(layer);

    return TRUE;
}

/* ------------------------------------------------------------------------ */
/* STRIDE_LAYER */
/* ------------------------------------------------------------------------ */

static CirnoVol *cirno_stride_new(lua_State *L, int index, CirnoVol *prevVol,
    CirnoLayerOption *option, int newIndex, uint32_t nSnapshot)
{
    uint32_t stride = option->stride;
    CirnoLayer *curLayer = lua_newuserdata(L, sizeof(CirnoLayer));
    luaL_getmetatable(L, "cirno.layer_instance");
    lua_setmetatable(L, -2);

    curLayer->type = kCirnoLayerStride;
    curLayer->inVol = prevVol;
    curLayer->outVol = cirno_vol_new((prevVol->sx + stride - 1) / stride,
        (prevVol->sy + stride - 1) / stride,
        (prevVol->sz + stride - 1) / stride, prevVol->depth,
        FALSE, nSnapshot);
    curLayer->info.stride = malloc(sizeof(CirnoLayerInfoStride));
    curLayer->info.stride->stride = stride;

    lua_rawseti(L, index - 1, newIndex);

    return curLayer->outVol;
}

static void cirno_stride_free(CirnoLayer *layer)
{
    cirno_vol_free(layer->outVol);
    free(layer->info.stride);
}

static void cirno_stride_forward(CirnoLayer *layer, uint8_t isTraining)
{
    UNREFERENCED_PARAMETER(isTraining);

    uint32_t stride = layer->info.stride->stride;
    uint32_t insx = layer->inVol->sx;
    uint32_t insy = layer->inVol->sy;
    uint32_t indepth = layer->inVol->depth;
    uint32_t outsx = layer->outVol->sx;
    uint32_t outsy = layer->outVol->sy;
    uint32_t outsz = layer->outVol->sz;
    uint32_t outdepth = layer->outVol->depth;
    uint32_t sx = 0, sy = 0, sz = 0;
    double *inw = layer->inVol->w;
    double *outw = layer->outVol->w;

    for (sz = 0; sz < outsz; ++sz) {
        for (sy = 0; sy < outsy; ++sy) {
            for (sx = 0; sx < outsx; ++sx) {
                memcpy(outw + (((sz * outsy + sy) * outsx * outdepth) + sx * outdepth),
                    inw + (((sz * insy + sy) * stride * insx * indepth) + sx * stride * indepth),
                    sizeof(double) * indepth);
            }
        }
    }
}

static void cirno_stride_backward(CirnoLayer *layer, uint32_t t, uint8_t resetHidden)
{
    UNREFERENCED_PARAMETER(resetHidden);

    uint32_t stride = layer->info.stride->stride;
    uint32_t insx = layer->inVol->sx;
    uint32_t insy = layer->inVol->sy;
    uint32_t indepth = layer->inVol->depth;
    uint32_t outsx = layer->outVol->sx;
    uint32_t outsy = layer->outVol->sy;
    uint32_t outsz = layer->outVol->sz;
    uint32_t outdepth = layer->outVol->depth;
    uint32_t sx = 0, sy = 0, sz = 0;
    double *indw = NULL;
    double *outdw = NULL;

    indw = cirno_vol_prev_dw(layer->inVol, t);
    outdw = cirno_vol_prev_dw(layer->outVol, t);

    for (sz = 0; sz < outsz; ++sz) {
        for (sy = 0; sy < outsy; ++sy) {
            for (sx = 0; sx < outsx; ++sx) {
                memcpy(indw + (((sz * insy + sy) * stride * insx * indepth) + sx * stride * indepth),
                    outdw + (((sz * outsy + sy) * outsx * outdepth) + sx * outdepth),
                    sizeof(double) * indepth);
            }
        }
    }
}

static void cirno_stride_clear_rin(CirnoLayer *layer)
{
    UNREFERENCED_PARAMETER(layer);

    return;
}

static double cirno_stride_apply_grad(CirnoLayer *layer, CirnoTrainerOption *option)
{
    UNREFERENCED_PARAMETER(layer);
    UNREFERENCED_PARAMETER(option);

    return 0.0;
}

static void cirno_stride_persist(MemBuf *buf, CirnoLayer *layer)
{
    mem_buf_write_uint32(buf, layer->type);
    cirno_vol_persist(buf, layer->outVol, FALSE);
    mem_buf_write_uint32(buf, layer->info.stride->stride);
}

static CirnoVol *cirno_stride_unpersist(lua_State *L, int index, CirnoVol *prevVol,
    MemBuf *buf, int newIndex)
{
    uint32_t stride = 0;

    CirnoLayer *curLayer = lua_newuserdata(L, sizeof(CirnoLayer));
    luaL_getmetatable(L, "cirno.layer_instance");
    lua_setmetatable(L, -2);


    curLayer->type = kCirnoLayerStride;
    curLayer->inVol = prevVol;
    curLayer->outVol = cirno_vol_unpersist(buf, FALSE);

    stride = mem_buf_read_uint32(buf);

    curLayer->info.stride = malloc(sizeof(CirnoLayerInfoStride));
    curLayer->info.stride->stride = stride;

    lua_rawseti(L, index - 1, newIndex);

    return curLayer->outVol;
}

static void cirno_stride_persist_grad(MemBuf *buf, CirnoLayer *layer)
{
    UNREFERENCED_PARAMETER(buf);
    UNREFERENCED_PARAMETER(layer);

    return;
}

static uint32_t cirno_stride_unpersist_grad(MemBuf *buf, CirnoLayer *layer)
{
    UNREFERENCED_PARAMETER(buf);
    UNREFERENCED_PARAMETER(layer);

    return TRUE;
}

/* ------------------------------------------------------------------------ */
/* CONV_LAYER */
/* ------------------------------------------------------------------------ */

static CirnoVol *cirno_conv_new(lua_State *L, int index, CirnoVol *prevVol,
    CirnoLayerOption *option, int newIndex, uint32_t nSnapshot)
{
    uint32_t sx = option->sx;
    uint32_t sy = option->sy;
    uint32_t sz = option->sz;

    uint32_t outsx = prevVol->sx - sx + 1;
    uint32_t outsy = prevVol->sy - sy + 1;
    uint32_t outsz = prevVol->sz - sz + 1;
    uint32_t outdepth = option->filters;

    uint32_t filtersize = sx * sy * sz * prevVol->depth;

    if (prevVol->sx < sx || prevVol->sy < sy || prevVol->sz < sz) {
        luaL_error(L, "input size cannot be less than conv filter");
    }

    CirnoLayer *curLayer = lua_newuserdata(L, sizeof(CirnoLayer));
    luaL_getmetatable(L, "cirno.layer_instance");
    lua_setmetatable(L, -2);

    curLayer->type = kCirnoLayerConv;
    curLayer->inVol = prevVol;
    curLayer->outVol = cirno_vol_new(outsx, outsy, outsz, outdepth, FALSE, nSnapshot);

    curLayer->info.conv = malloc(sizeof(CirnoLayerInfoConv));
    curLayer->info.conv->sx = sx;
    curLayer->info.conv->sy = sy;
    curLayer->info.conv->sz = sz;
    curLayer->info.conv->l1DecayMul = option->l1DecayMul;
    curLayer->info.conv->l2DecayMul = option->l2DecayMul;
    curLayer->info.conv->bias = cirno_vol_new(1, 1, 1, outdepth, TRUE, 1);
    curLayer->info.conv->filter =
        cirno_random_vol_new(filtersize, outdepth, 1, 1, filtersize, TRUE, 1);

    lua_rawseti(L, index - 1, newIndex);

    return curLayer->outVol;
}

static void cirno_conv_free(CirnoLayer *layer)
{
    cirno_vol_free(layer->outVol);
    cirno_vol_free(layer->info.conv->bias);
    cirno_vol_free(layer->info.conv->filter);
    free(layer->info.conv);
}

static void cirno_conv_forward(CirnoLayer *layer, uint8_t isTraining)
{
    UNREFERENCED_PARAMETER(isTraining);

    uint32_t insx = layer->inVol->sx;
    uint32_t insy = layer->inVol->sy;
    uint32_t indepth = layer->inVol->depth;
    uint32_t outsx = layer->outVol->sx;
    uint32_t outsy = layer->outVol->sy;
    uint32_t outsz = layer->outVol->sz;
    uint32_t outdepth = layer->outVol->depth;
    double sum = 0.0;
    uint32_t sx = layer->info.conv->sx;
    uint32_t sy = layer->info.conv->sy;
    uint32_t sz = layer->info.conv->sz;
    double *biasw = layer->info.conv->bias->w;
    double *filterw = layer->info.conv->filter->w;
    uint32_t d = 0, ox = 0, oy = 0, oz = 0;
    uint32_t fx = 0, fy = 0, fz = 0;
    uint32_t f = 0, i = 0;
    uint32_t filterbase = 0, filterindex = 0, inputindex = 0;
    double *inw = layer->inVol->w;
    double *outw = layer->outVol->w;

    for (d = 0; d < outdepth; ++d) {
        filterbase = d * sx * sy * sz * indepth;

        for (oz = 0; oz < outsz; ++oz) {
            for (oy = 0; oy < outsy; ++oy) {
                for (ox = 0; ox < outsx; ++ox) {
                    sum = 0.0;

                    for (fz = 0; fz < sz; ++fz) {
                        for (fy = 0; fy < sy; ++fy) {
                            inputindex = (((oz + fz) * insy + (oy + fy)) * insx + ox) * indepth;
                            filterindex = filterbase + (((fz * sy) + fy) * sx * indepth);
                            fx = (sx * indepth) % 4;

                            switch (fx) {
                                case 3:
                                    sum += filterw[filterindex + 2] * inw[inputindex + 2];
                                case 2:
                                    sum += filterw[filterindex + 1] * inw[inputindex + 1];
                                case 1:
                                    sum += filterw[filterindex] * inw[inputindex];
                            }

                            for (; fx < sx * indepth; fx += 4) {
                                f = filterindex + fx;
                                i = inputindex + fx;

                                sum += filterw[f] * inw[i] + filterw[f + 1] * inw[i + 1]
                                    + filterw[f + 2] * inw[i + 2] + filterw[f + 3] * inw[i + 3];
                            }
                        }
                    }

                    sum += biasw[d];
                    outw[(((oz * outsy) + oy) * outsx + ox) * outdepth + d] = sum;
                }
            }
        }
    }
}

static void cirno_conv_backward(CirnoLayer *layer, uint32_t t, uint8_t resetHidden)
{
    UNREFERENCED_PARAMETER(resetHidden);

    uint32_t insx = layer->inVol->sx;
    uint32_t insy = layer->inVol->sy;
    uint32_t indepth = layer->inVol->depth;
    uint32_t outsx = layer->outVol->sx;
    uint32_t outsy = layer->outVol->sy;
    uint32_t outsz = layer->outVol->sz;
    uint32_t outdepth = layer->outVol->depth;
    double chaingrad = 0.0;
    uint32_t sx = layer->info.conv->sx;
    uint32_t sy = layer->info.conv->sy;
    uint32_t sz = layer->info.conv->sz;
    double *biasdw = layer->info.conv->bias->dw;
    double *filterw = layer->info.conv->filter->w;
    double *filterdw = layer->info.conv->filter->dw;
    uint32_t d = 0, ox = 0, oy = 0, oz = 0;
    uint32_t fx = 0, fy = 0, fz = 0;
    uint32_t filterbase = 0, filterindex = 0, inputindex = 0;
    double *inw = NULL;
    double *indw = NULL;
    double *outdw = NULL;

    inw = cirno_vol_prev_w(layer->inVol, t);
    indw = cirno_vol_prev_dw(layer->inVol, t);
    outdw = cirno_vol_prev_dw(layer->outVol, t);

    memset(indw, 0, sizeof(double) * layer->inVol->n);

    for (d = 0; d < outdepth; ++d) {
        filterbase = d * sx * sy * sz * indepth;

        for (oz = 0; oz < outsz; ++oz) {
            for (oy = 0; oy < outsy; ++oy) {
                for (ox = 0; ox < outsx; ++ox) {
                    chaingrad = outdw[((oz * outsy + oy) * outsx + ox) * outdepth + d];

                    for (fz = 0; fz < sz; ++fz) {
                        for (fy = 0; fy < sy; ++fy) {
                            inputindex = (((oz + fz) * insy + (oy + fy)) * insx + ox) * indepth;
                            filterindex = filterbase + (((fz * sy) + fy) * sx * indepth);

                            for (fx = 0; fx < sx * indepth; ++fx) {
                                filterdw[filterindex + fx] += inw[inputindex + fx] * chaingrad;
                                indw[inputindex + fx] += filterw[filterindex + fx] * chaingrad;
                            }
                        }
                    }

                    biasdw[d] += chaingrad;
                }
            }
        }
    }

    ++(layer->info.conv->filter->k);
    ++(layer->info.conv->bias->k);
}

static void cirno_conv_clear_rin(CirnoLayer *layer)
{
    UNREFERENCED_PARAMETER(layer);

    return;
}

static double cirno_conv_apply_grad(CirnoLayer *layer, CirnoTrainerOption *option)
{
    double loss = 0.0;
    CirnoLayerInfoConv *conv = layer->info.conv;
    CirnoVol *filter = conv->filter;
    CirnoVol *bias = conv->bias;

    loss += cirno_trainer_apply(option, filter,
                                conv->l1DecayMul, conv->l2DecayMul);

    loss += cirno_trainer_apply(option, bias, 0.0, 0.0);

    return loss;
}

static void cirno_conv_persist(MemBuf *buf, CirnoLayer *layer)
{
    mem_buf_write_uint32(buf, layer->type);
    cirno_vol_persist(buf, layer->outVol, FALSE);
    mem_buf_write_uint32(buf, layer->info.conv->sx);
    mem_buf_write_uint32(buf, layer->info.conv->sy);
    mem_buf_write_uint32(buf, layer->info.conv->sz);
    mem_buf_write_double(buf, layer->info.conv->l1DecayMul);
    mem_buf_write_double(buf, layer->info.conv->l2DecayMul);
    cirno_vol_persist(buf, layer->info.conv->bias, TRUE);
    cirno_vol_persist(buf, layer->info.conv->filter, TRUE);
}

static CirnoVol *cirno_conv_unpersist(lua_State *L, int index, CirnoVol *prevVol,
    MemBuf *buf, int newIndex)
{
    uint32_t sx = 0, sy = 0, sz = 0;
    double l1DecayMul = 0.0, l2DecayMul = 0.0;

    CirnoLayer *curLayer = lua_newuserdata(L, sizeof(CirnoLayer));
    luaL_getmetatable(L, "cirno.layer_instance");
    lua_setmetatable(L, -2);

    curLayer->type = kCirnoLayerConv;
    curLayer->inVol = prevVol;
    curLayer->outVol = cirno_vol_unpersist(buf, FALSE);

    sx = mem_buf_read_uint32(buf);
    sy = mem_buf_read_uint32(buf);
    sz = mem_buf_read_uint32(buf);
    l1DecayMul = mem_buf_read_double(buf);
    l2DecayMul = mem_buf_read_double(buf);

    curLayer->info.conv = malloc(sizeof(CirnoLayerInfoConv));
    curLayer->info.conv->sx = sx;
    curLayer->info.conv->sy = sy;
    curLayer->info.conv->sz = sz;
    curLayer->info.conv->l1DecayMul = l1DecayMul;
    curLayer->info.conv->l2DecayMul = l2DecayMul;
    curLayer->info.conv->bias = cirno_vol_unpersist(buf, TRUE);
    curLayer->info.conv->filter = cirno_vol_unpersist(buf, TRUE);

    lua_rawseti(L, index - 1, newIndex);

    return curLayer->outVol;
}

static void cirno_conv_persist_grad(MemBuf *buf, CirnoLayer *layer)
{
    cirno_vol_persist_grad(buf, layer->info.conv->filter);
    cirno_vol_persist_grad(buf, layer->info.conv->bias);
}

static uint32_t cirno_conv_unpersist_grad(MemBuf *buf, CirnoLayer *layer)
{
    uint32_t result = TRUE;

    result = cirno_vol_unpersist_grad(buf, layer->info.conv->filter);

    if (result) {
        result = cirno_vol_unpersist_grad(buf, layer->info.conv->bias);
    }

    return result;
}

/* ------------------------------------------------------------------------ */
/* FULLYCONN_LAYER */
/* ------------------------------------------------------------------------ */

static CirnoVol *cirno_fullyconn_new(lua_State *L, int index, CirnoVol *prevVol,
    CirnoLayerOption *option, int newIndex, uint32_t nSnapshot)
{
    uint32_t inCount =
        prevVol->sx * prevVol->sy * prevVol->sz * prevVol->depth;
    uint32_t outCount = option->out;

    CirnoLayer *curLayer = lua_newuserdata(L, sizeof(CirnoLayer));
    luaL_getmetatable(L, "cirno.layer_instance");
    lua_setmetatable(L, -2);

    curLayer->type = kCirnoLayerFullyConn;
    curLayer->inVol = prevVol;
    curLayer->outVol = cirno_vol_new(1, 1, 1, outCount, FALSE, nSnapshot);
    curLayer->info.fullyConn = malloc(sizeof(CirnoLayerInfoFullyConn));
    curLayer->info.fullyConn->l1DecayMul = option->l1DecayMul;
    curLayer->info.fullyConn->l2DecayMul = option->l2DecayMul;
    curLayer->info.fullyConn->weight =
        cirno_random_vol_new(inCount, outCount, 1, 1, inCount, TRUE, 1);
    curLayer->info.fullyConn->bias = cirno_vol_new(1, 1, 1, outCount, TRUE, 1);

    lua_rawseti(L, index - 1, newIndex);

    return curLayer->outVol;
}

static void cirno_fullyconn_free(CirnoLayer *layer)
{
    cirno_vol_free(layer->outVol);
    cirno_vol_free(layer->info.fullyConn->weight);
    cirno_vol_free(layer->info.fullyConn->bias);
    free(layer->info.fullyConn);
}

static void cirno_fullyconn_forward(CirnoLayer *layer, uint8_t isTraining)
{
    UNREFERENCED_PARAMETER(isTraining);

    uint32_t i = 0, j = 0, wbase = 0;
    double sum = 0.0;
    CirnoVol *inVol = layer->inVol;
    CirnoVol *outVol = layer->outVol;
    CirnoLayerInfoFullyConn *fullyConnInfo = layer->info.fullyConn;
    CirnoVol *weight = fullyConnInfo->weight;
    CirnoVol *bias = fullyConnInfo->bias;
    double *inw = inVol->w;
    double *outw = outVol->w;
    double *weightw = weight->w;
    double *biasw = bias->w;
    uint32_t wsx = weight->sx;
    uint32_t wsy = weight->sy;

    for (i = 0; i < wsy; ++i) {
        sum = 0.0;

        j = wsx % 4;

        switch (j) {
            case 3: sum += weightw[wbase + 2] * inw[2];
            case 2: sum += weightw[wbase + 1] * inw[1];
            case 1: sum += weightw[wbase] * inw[0];
        }

        for (; j < wsx; j += 4) {
            sum += weightw[wbase + j] * inw[j]
                + weightw[wbase + j + 1] * inw[j + 1]
                + weightw[wbase + j + 2] * inw[j + 2]
                + weightw[wbase + j + 3] * inw[j + 3];
        }

        outw[i] = sum + biasw[i];
        wbase += wsx;
    }
}

static void cirno_fullyconn_backward(CirnoLayer *layer, uint32_t t, uint8_t resetHidden)
{
    UNREFERENCED_PARAMETER(resetHidden);

    uint32_t i = 0, j = 0;
    CirnoVol *inVol = layer->inVol;
    CirnoVol *outVol = layer->outVol;
    CirnoLayerInfoFullyConn *fullyConnInfo = layer->info.fullyConn;
    CirnoVol *weight = fullyConnInfo->weight;
    CirnoVol *bias = fullyConnInfo->bias;
    uint32_t n = inVol->n;
    double *inwb = NULL, *indwb = NULL;
    double *inw = NULL, *indw = NULL;
    double *outdw = NULL;
    double *weightw = weight->w;
    double *weightdw = weight->dw;
    double *biasdw = bias->dw;
    uint32_t wsx = weight->sx;
    uint32_t wsy = weight->sy;
    double outdwi = 0.0;

    inwb = cirno_vol_prev_w(inVol, t);
    indwb = cirno_vol_prev_dw(inVol, t);
    outdw = cirno_vol_prev_dw(outVol, t);

    memset(indwb, 0, sizeof(double) * n);

    for (i = 0; i < wsy; ++i) {
        outdwi = *(outdw++);

        *(biasdw++) += outdwi;

        inw = inwb;
        indw = indwb;

        for (j = 0; j < wsx; ++j) {
            *(indw++) += outdwi * *(weightw++);
            *(weightdw++) += outdwi * *(inw++);
        }
    }

    ++(fullyConnInfo->weight->k);
    ++(fullyConnInfo->bias->k);
}

static void cirno_fullyconn_clear_rin(CirnoLayer *layer)
{
    UNREFERENCED_PARAMETER(layer);

    return;
}

static double cirno_fullyconn_apply_grad(CirnoLayer *layer, CirnoTrainerOption *option)
{
    double loss = 0.0;
    CirnoLayerInfoFullyConn *fullyConn = layer->info.fullyConn;
    CirnoVol *weight = fullyConn->weight;
    CirnoVol *bias = fullyConn->bias;

    loss += cirno_trainer_apply(option, weight,
                                fullyConn->l1DecayMul, fullyConn->l2DecayMul);

    loss += cirno_trainer_apply(option, bias, 0.0, 0.0);

    return loss;
}

static void cirno_fullyconn_persist(MemBuf *buf, CirnoLayer *layer)
{
    mem_buf_write_uint32(buf, layer->type);
    cirno_vol_persist(buf, layer->outVol, FALSE);
    mem_buf_write_double(buf, layer->info.fullyConn->l1DecayMul);
    mem_buf_write_double(buf, layer->info.fullyConn->l2DecayMul);
    cirno_vol_persist(buf, layer->info.fullyConn->weight, TRUE);
    cirno_vol_persist(buf, layer->info.fullyConn->bias, TRUE);
}

static CirnoVol *cirno_fullyconn_unpersist(lua_State *L, int index, CirnoVol *prevVol,
    MemBuf *buf, int newIndex)
{
    CirnoLayer *curLayer = lua_newuserdata(L, sizeof(CirnoLayer));
    luaL_getmetatable(L, "cirno.layer_instance");
    lua_setmetatable(L, -2);

    curLayer->type = kCirnoLayerFullyConn;
    curLayer->inVol = prevVol;
    curLayer->outVol = cirno_vol_unpersist(buf, FALSE);

    curLayer->info.fullyConn = malloc(sizeof(CirnoLayerInfoFullyConn));
    curLayer->info.fullyConn->l1DecayMul = mem_buf_read_double(buf);
    curLayer->info.fullyConn->l2DecayMul = mem_buf_read_double(buf);
    curLayer->info.fullyConn->weight = cirno_vol_unpersist(buf, TRUE);
    curLayer->info.fullyConn->bias = cirno_vol_unpersist(buf, TRUE);

    lua_rawseti(L, index - 1, newIndex);

    return curLayer->outVol;
}

static void cirno_fullyconn_persist_grad(MemBuf *buf, CirnoLayer *layer)
{
    cirno_vol_persist_grad(buf, layer->info.fullyConn->weight);
    cirno_vol_persist_grad(buf, layer->info.fullyConn->bias);
}

static uint32_t cirno_fullyconn_unpersist_grad(MemBuf *buf, CirnoLayer *layer)
{
    uint32_t result = TRUE;

    result = cirno_vol_unpersist_grad(buf, layer->info.fullyConn->weight);

    if (result) {
        result = cirno_vol_unpersist_grad(buf, layer->info.fullyConn->bias);
    }

    return result;
}

/* ------------------------------------------------------------------------ */
/* DROPOUT_LAYER */
/* ------------------------------------------------------------------------ */

static CirnoVol *cirno_dropout_new(lua_State *L, int index, CirnoVol *prevVol,
    CirnoLayerOption *option, int newIndex, uint32_t nSnapshot)
{
    CirnoLayer *curLayer = lua_newuserdata(L, sizeof(CirnoLayer));
    luaL_getmetatable(L, "cirno.layer_instance");
    lua_setmetatable(L, -2);

    curLayer->type = kCirnoLayerDropout;
    curLayer->inVol = prevVol;
    curLayer->outVol = cirno_vol_new(prevVol->sx,
        prevVol->sy, prevVol->sz,
        prevVol->depth, FALSE, nSnapshot);
    curLayer->info.dropout = malloc(sizeof(CirnoLayerInfoDropout));
    curLayer->info.dropout->nSnapshot = nSnapshot;
    curLayer->info.dropout->drop = option->drop;
    curLayer->info.dropout->present = 1.0 - option->drop;
    curLayer->info.dropout->isDroppedBase =
        malloc(sizeof(uint8_t) * prevVol->n * nSnapshot);
    curLayer->info.dropout->isDropped =
        curLayer->info.dropout->isDroppedBase;
    curLayer->info.dropout->curSnapshot = 0;

    lua_rawseti(L, index - 1, newIndex);

    return curLayer->outVol;
}

static void cirno_dropout_free(CirnoLayer *layer)
{
    cirno_vol_free(layer->outVol);
    free(layer->info.dropout->isDroppedBase);
    free(layer->info.dropout);
}

static void cirno_dropout_forward(CirnoLayer *layer, uint8_t isTraining)
{
    uint32_t i = 0;
    CirnoVol *inVol = layer->inVol;
    CirnoVol *outVol = layer->outVol;
    CirnoLayerInfoDropout *dropoutInfo = layer->info.dropout;
    uint32_t n = inVol->n;
    double *inw = inVol->w;
    double *outw = outVol->w;
    double drop = dropoutInfo->drop;
    double present = dropoutInfo->present;
    uint8_t *isDropped = NULL;

    dropoutInfo->curSnapshot =
        (dropoutInfo->curSnapshot + 1) % dropoutInfo->nSnapshot;

    dropoutInfo->isDropped = dropoutInfo->isDroppedBase
        + (dropoutInfo->curSnapshot * n);

    isDropped = dropoutInfo->isDropped;

    if (isTraining) {
        for (i = 0; i < n; ++i) {
            if (runif() < drop) {
                *(isDropped++) = TRUE;
                *(outw++) = 0.0;
            }
            else {
                *(isDropped++) = FALSE;
                *(outw++) = *inw;
            }

            ++inw;
        }
    }
    else {
        for (i = 0; i < n; ++i) {
            *(outw++) = *(inw++) * present;
        }
    }
}

static void cirno_dropout_backward(CirnoLayer *layer, uint32_t t, uint8_t resetHidden)
{
    UNREFERENCED_PARAMETER(resetHidden);

    uint32_t i = 0;
    CirnoVol *inVol = layer->inVol;
    CirnoVol *outVol = layer->outVol;
    CirnoLayerInfoDropout *dropoutInfo = layer->info.dropout;
    uint32_t n = inVol->n;
    uint32_t nSnapshot = dropoutInfo->nSnapshot;
    uint32_t curSnapshot = dropoutInfo->curSnapshot;
    double *indw = NULL;
    double *outdw = NULL;
    uint8_t *isDropped = NULL;

    indw = cirno_vol_prev_dw(inVol, t);
    outdw = cirno_vol_prev_dw(outVol, t);

    isDropped = dropoutInfo->isDroppedBase +
        (n * ((curSnapshot - t + nSnapshot) % nSnapshot));

    for (i = 0; i < n; ++i) {
        if (*(isDropped++)) {
            *(indw++) = 0.0;
        }
        else {
            *(indw++) = *outdw;
        }

        ++outdw;
    }
}

static void cirno_dropout_clear_rin(CirnoLayer *layer)
{
    UNREFERENCED_PARAMETER(layer);

    return;
}

static double cirno_dropout_apply_grad(CirnoLayer *layer, CirnoTrainerOption *option)
{
    UNREFERENCED_PARAMETER(layer);
    UNREFERENCED_PARAMETER(option);

    return 0.0;
}

static void cirno_dropout_persist(MemBuf *buf, CirnoLayer *layer)
{
    mem_buf_write_uint32(buf, layer->type);
    cirno_vol_persist(buf, layer->outVol, FALSE);
    mem_buf_write_double(buf, layer->info.dropout->drop);
    mem_buf_write_uint32(buf, layer->info.dropout->nSnapshot);
}

static CirnoVol *cirno_dropout_unpersist(lua_State *L, int index, CirnoVol *prevVol,
    MemBuf *buf, int newIndex)
{
    double drop = 0.0;
    uint32_t nSnapshot = 0;

    CirnoLayer *curLayer = lua_newuserdata(L, sizeof(CirnoLayer));
    luaL_getmetatable(L, "cirno.layer_instance");
    lua_setmetatable(L, -2);

    curLayer->type = kCirnoLayerDropout;
    curLayer->inVol = prevVol;
    curLayer->outVol = cirno_vol_unpersist(buf, FALSE);

    drop = mem_buf_read_double(buf);
    nSnapshot = mem_buf_read_uint32(buf);

    curLayer->info.dropout = malloc(sizeof(CirnoLayerInfoDropout));
    curLayer->info.dropout->nSnapshot = nSnapshot;
    curLayer->info.dropout->drop = drop;
    curLayer->info.dropout->present = 1.0 - drop;
    curLayer->info.dropout->isDroppedBase =
        malloc(sizeof(uint8_t) * prevVol->n * nSnapshot);
    curLayer->info.dropout->isDropped =
        curLayer->info.dropout->isDroppedBase;
    curLayer->info.dropout->curSnapshot = 0;

    lua_rawseti(L, index - 1, newIndex);

    return curLayer->outVol;
}

static void cirno_dropout_persist_grad(MemBuf *buf, CirnoLayer *layer)
{
    UNREFERENCED_PARAMETER(buf);
    UNREFERENCED_PARAMETER(layer);

    return;
}

static uint32_t cirno_dropout_unpersist_grad(MemBuf *buf, CirnoLayer *layer)
{
    UNREFERENCED_PARAMETER(buf);
    UNREFERENCED_PARAMETER(layer);

    return TRUE;
}

/* ------------------------------------------------------------------------ */
/* TANH_LAYER */
/* ------------------------------------------------------------------------ */

static CirnoVol *cirno_tanh_new(lua_State *L, int index, CirnoVol *prevVol,
    CirnoLayerOption *option, int newIndex, uint32_t nSnapshot)
{
    UNREFERENCED_PARAMETER(option);

    CirnoLayer *curLayer = lua_newuserdata(L, sizeof(CirnoLayer));
    luaL_getmetatable(L, "cirno.layer_instance");
    lua_setmetatable(L, -2);

    curLayer->type = kCirnoLayerTanh;
    curLayer->inVol = prevVol;
    curLayer->outVol = cirno_vol_new(prevVol->sx,
        prevVol->sy, prevVol->sz,
        prevVol->depth, FALSE, nSnapshot);

    lua_rawseti(L, index - 1, newIndex);

    return curLayer->outVol;
}

static void cirno_tanh_free(CirnoLayer *layer)
{
    cirno_vol_free(layer->outVol);
}

static void cirno_tanh_forward(CirnoLayer *layer, uint8_t isTraining)
{
    UNREFERENCED_PARAMETER(isTraining);

    uint32_t i = 0;
    CirnoVol *inVol = layer->inVol;
    CirnoVol *outVol = layer->outVol;
    uint32_t n = inVol->n;
    double *inw = inVol->w;
    double *outw = outVol->w;

    for (i = 0; i < n; ++i) {
        *(outw++) = tanh(*(inw++));
    }
}

static void cirno_tanh_backward(CirnoLayer *layer, uint32_t t, uint8_t resetHidden)
{
    UNREFERENCED_PARAMETER(resetHidden);

    uint32_t i = 0;
    CirnoVol *inVol = layer->inVol;
    CirnoVol *outVol = layer->outVol;
    uint32_t n = inVol->n;
    double *indw = NULL;
    double *outw = NULL;
    double *outdw = NULL;

    indw = cirno_vol_prev_dw(inVol, t);
    outw = cirno_vol_prev_w(outVol, t);
    outdw = cirno_vol_prev_dw(outVol, t);

    for (i = 0; i < n; ++i) {
        *(indw++) = (1.0 - *outw * *outw) * *(outdw++);
        ++outw;
    }
}

static void cirno_tanh_clear_rin(CirnoLayer *layer)
{
    UNREFERENCED_PARAMETER(layer);

    return;
}

static double cirno_tanh_apply_grad(CirnoLayer *layer, CirnoTrainerOption *option)
{
    UNREFERENCED_PARAMETER(layer);
    UNREFERENCED_PARAMETER(option);

    return 0.0;
}

static void cirno_tanh_persist(MemBuf *buf, CirnoLayer *layer)
{
    mem_buf_write_uint32(buf, layer->type);
    cirno_vol_persist(buf, layer->outVol, FALSE);
}

static CirnoVol *cirno_tanh_unpersist(lua_State *L, int index, CirnoVol *prevVol,
    MemBuf *buf, int newIndex)
{
    CirnoLayer *curLayer = lua_newuserdata(L, sizeof(CirnoLayer));
    luaL_getmetatable(L, "cirno.layer_instance");
    lua_setmetatable(L, -2);

    curLayer->type = kCirnoLayerTanh;
    curLayer->inVol = prevVol;
    curLayer->outVol = cirno_vol_unpersist(buf, FALSE);

    lua_rawseti(L, index - 1, newIndex);

    return curLayer->outVol;
}

static void cirno_tanh_persist_grad(MemBuf *buf, CirnoLayer *layer)
{
    UNREFERENCED_PARAMETER(buf);
    UNREFERENCED_PARAMETER(layer);

    return;
}

static uint32_t cirno_tanh_unpersist_grad(MemBuf *buf, CirnoLayer *layer)
{
    UNREFERENCED_PARAMETER(buf);
    UNREFERENCED_PARAMETER(layer);

    return TRUE;
}

/* ------------------------------------------------------------------------ */
/* SIGMOID_LAYER */
/* ------------------------------------------------------------------------ */

static CirnoVol *cirno_sigmoid_new(lua_State *L, int index, CirnoVol *prevVol,
    CirnoLayerOption *option, int newIndex, uint32_t nSnapshot)
{
    UNREFERENCED_PARAMETER(option);

    CirnoLayer *curLayer = lua_newuserdata(L, sizeof(CirnoLayer));
    luaL_getmetatable(L, "cirno.layer_instance");
    lua_setmetatable(L, -2);

    curLayer->type = kCirnoLayerSigmoid;
    curLayer->inVol = prevVol;
    curLayer->outVol = cirno_vol_new(prevVol->sx,
        prevVol->sy, prevVol->sz,
        prevVol->depth, FALSE, nSnapshot);

    lua_rawseti(L, index - 1, newIndex);

    return curLayer->outVol;
}

static void cirno_sigmoid_free(CirnoLayer *layer)
{
    cirno_vol_free(layer->outVol);
}

static void cirno_sigmoid_forward(CirnoLayer *layer, uint8_t isTraining)
{
    UNREFERENCED_PARAMETER(isTraining);

    uint32_t i = 0;
    CirnoVol *inVol = layer->inVol;
    CirnoVol *outVol = layer->outVol;
    uint32_t n = inVol->n;
    double *inw = inVol->w;
    double *outw = outVol->w;

    for (i = 0; i < n; ++i) {
        *(outw++) = 1.0 / (1.0 + exp(- *(inw++)));
    }
}

static void cirno_sigmoid_backward(CirnoLayer *layer, uint32_t t, uint8_t resetHidden)
{
    UNREFERENCED_PARAMETER(resetHidden);

    uint32_t i = 0;
    CirnoVol *inVol = layer->inVol;
    CirnoVol *outVol = layer->outVol;
    uint32_t n = inVol->n;
    double *indw = NULL;
    double *outw = NULL;
    double *outdw = NULL;

    indw = cirno_vol_prev_dw(inVol, t);
    outw = cirno_vol_prev_w(outVol, t);
    outdw = cirno_vol_prev_dw(outVol, t);

    for (i = 0; i < n; ++i) {
        *(indw++) = *outw * (1.0 - *outw) * *(outdw++);
        ++outw;
    }
}

static void cirno_sigmoid_clear_rin(CirnoLayer *layer)
{
    UNREFERENCED_PARAMETER(layer);

    return;
}

static double cirno_sigmoid_apply_grad(CirnoLayer *layer, CirnoTrainerOption *option)
{
    UNREFERENCED_PARAMETER(layer);
    UNREFERENCED_PARAMETER(option);

    return 0.0;
}

static void cirno_sigmoid_persist(MemBuf *buf, CirnoLayer *layer)
{
    mem_buf_write_uint32(buf, layer->type);
    cirno_vol_persist(buf, layer->outVol, FALSE);
}

static CirnoVol *cirno_sigmoid_unpersist(lua_State *L, int index, CirnoVol *prevVol,
    MemBuf *buf, int newIndex)
{
    CirnoLayer *curLayer = lua_newuserdata(L, sizeof(CirnoLayer));
    luaL_getmetatable(L, "cirno.layer_instance");
    lua_setmetatable(L, -2);

    curLayer->type = kCirnoLayerSigmoid;
    curLayer->inVol = prevVol;
    curLayer->outVol = cirno_vol_unpersist(buf, FALSE);

    lua_rawseti(L, index - 1, newIndex);

    return curLayer->outVol;
}

static void cirno_sigmoid_persist_grad(MemBuf *buf, CirnoLayer *layer)
{
    UNREFERENCED_PARAMETER(buf);
    UNREFERENCED_PARAMETER(layer);

    return;
}

static uint32_t cirno_sigmoid_unpersist_grad(MemBuf *buf, CirnoLayer *layer)
{
    UNREFERENCED_PARAMETER(buf);
    UNREFERENCED_PARAMETER(layer);

    return TRUE;
}

/* ------------------------------------------------------------------------ */
/* RELU_LAYER */
/* ------------------------------------------------------------------------ */

static CirnoVol *cirno_relu_new(lua_State *L, int index, CirnoVol *prevVol,
    CirnoLayerOption *option, int newIndex, uint32_t nSnapshot)
{
    UNREFERENCED_PARAMETER(option);

    CirnoLayer *curLayer = lua_newuserdata(L, sizeof(CirnoLayer));
    luaL_getmetatable(L, "cirno.layer_instance");
    lua_setmetatable(L, -2);

    curLayer->type = kCirnoLayerRelu;
    curLayer->inVol = prevVol;
    curLayer->outVol = cirno_vol_new(prevVol->sx,
        prevVol->sy, prevVol->sz,
        prevVol->depth, FALSE, nSnapshot);

    lua_rawseti(L, index - 1, newIndex);

    return curLayer->outVol;
}

static void cirno_relu_free(CirnoLayer *layer)
{
    cirno_vol_free(layer->outVol);
}

static void cirno_relu_forward(CirnoLayer *layer, uint8_t isTraining)
{
    UNREFERENCED_PARAMETER(isTraining);

    uint32_t i = 0;
    CirnoVol *inVol = layer->inVol;
    CirnoVol *outVol = layer->outVol;
    uint32_t n = inVol->n;
    double *inw = inVol->w;
    double *outw = outVol->w;

    for (i = 0; i < n; ++i) {
        *(outw++) = (*inw < 0.0) ? 0.0 : *inw;
        ++inw;
    }
}

static void cirno_relu_backward(CirnoLayer *layer, uint32_t t, uint8_t resetHidden)
{
    UNREFERENCED_PARAMETER(resetHidden);

    uint32_t i = 0;
    CirnoVol *inVol = layer->inVol;
    CirnoVol *outVol = layer->outVol;
    uint32_t n = inVol->n;
    double *indw = NULL;
    double *outw = NULL;
    double *outdw = NULL;

    indw = cirno_vol_prev_dw(inVol, t);
    outw = cirno_vol_prev_w(outVol, t);
    outdw = cirno_vol_prev_dw(outVol, t);

    for (i = 0; i < n; ++i) {
        *(indw++) = (*(outw++) <= 0.0) ? 0.0 : *outdw;
        ++outdw;
    }
}

static void cirno_relu_clear_rin(CirnoLayer *layer)
{
    UNREFERENCED_PARAMETER(layer);

    return;
}

static double cirno_relu_apply_grad(CirnoLayer *layer, CirnoTrainerOption *option)
{
    UNREFERENCED_PARAMETER(layer);
    UNREFERENCED_PARAMETER(option);

    return 0.0;
}

static void cirno_relu_persist(MemBuf *buf, CirnoLayer *layer)
{
    mem_buf_write_uint32(buf, layer->type);
    cirno_vol_persist(buf, layer->outVol, FALSE);
}

static CirnoVol *cirno_relu_unpersist(lua_State *L, int index, CirnoVol *prevVol,
    MemBuf *buf, int newIndex)
{
    CirnoLayer *curLayer = lua_newuserdata(L, sizeof(CirnoLayer));
    luaL_getmetatable(L, "cirno.layer_instance");
    lua_setmetatable(L, -2);

    curLayer->type = kCirnoLayerRelu;
    curLayer->inVol = prevVol;
    curLayer->outVol = cirno_vol_unpersist(buf, FALSE);

    lua_rawseti(L, index - 1, newIndex);

    return curLayer->outVol;
}

static void cirno_relu_persist_grad(MemBuf *buf, CirnoLayer *layer)
{
    UNREFERENCED_PARAMETER(buf);
    UNREFERENCED_PARAMETER(layer);

    return;
}

static uint32_t cirno_relu_unpersist_grad(MemBuf *buf, CirnoLayer *layer)
{
    UNREFERENCED_PARAMETER(buf);
    UNREFERENCED_PARAMETER(layer);

    return TRUE;
}

/* ------------------------------------------------------------------------ */
/* LSTM_LAYER */
/* ------------------------------------------------------------------------ */

static CirnoVol *cirno_lstm_new(lua_State *L, int index, CirnoVol *prevVol,
    CirnoLayerOption *option, int newIndex, uint32_t nSnapshot)
{
    uint32_t zCount = 0;
    uint32_t hCount = 0;
    CirnoLayer *curLayer = lua_newuserdata(L, sizeof(CirnoLayer));
    luaL_getmetatable(L, "cirno.layer_instance");
    lua_setmetatable(L, -2);

    curLayer->type = kCirnoLayerLstm;
    curLayer->inVol = prevVol;
    curLayer->outVol = cirno_vol_new(1, 1, 1,
        option->out, FALSE, nSnapshot);
    curLayer->info.lstm = malloc(sizeof(CirnoLayerInfoLstm));
    curLayer->info.lstm->l1DecayMul = option->l1DecayMul;
    curLayer->info.lstm->l2DecayMul = option->l2DecayMul;
    curLayer->info.lstm->rin = cirno_vol_new(1, 1, 1,
        prevVol->n + (option->out + option->hidden) * 2, FALSE, nSnapshot);
    curLayer->info.lstm->rout = cirno_vol_new(1, 1, 1,
        (option->out + option->hidden) * 2, FALSE, nSnapshot);

    zCount = prevVol->n + option->out + option->hidden;
    hCount = option->out + option->hidden;

    curLayer->info.lstm->hsize = hCount;
    curLayer->info.lstm->weight =
        cirno_random_vol_new(zCount, hCount * 4, 1, 1, zCount, TRUE, 1);
    curLayer->info.lstm->bias = cirno_vol_new(1, 1, 1, hCount * 4, TRUE, 1);
    curLayer->info.lstm->figo = cirno_vol_new(1, 1, 1,
        hCount * 4, FALSE, nSnapshot);
    curLayer->info.lstm->afigo = cirno_vol_new(1, 1, 1,
        hCount * 4, FALSE, nSnapshot);

    lua_rawseti(L, index - 1, newIndex);

    return curLayer->outVol;
}

static void cirno_lstm_free(CirnoLayer *layer)
{
    cirno_vol_free(layer->outVol);
    cirno_vol_free(layer->info.lstm->rin);
    cirno_vol_free(layer->info.lstm->rout);
    cirno_vol_free(layer->info.lstm->weight);
    cirno_vol_free(layer->info.lstm->bias);
    cirno_vol_free(layer->info.lstm->figo);
    cirno_vol_free(layer->info.lstm->afigo);
    free(layer->info.lstm);
}

static void cirno_lstm_forward(CirnoLayer *layer, uint8_t isTraining)
{
    UNREFERENCED_PARAMETER(isTraining);

    uint32_t i = 0, j = 0, wbase = 0;
    double sum = 0.0;
    CirnoVol *inVol = layer->inVol;
    CirnoVol *outVol = layer->outVol;
    CirnoLayerInfoLstm *lstm = layer->info.lstm;
    CirnoVol *weight = lstm->weight;
    CirnoVol *bias = lstm->bias;
    double *weightw = weight->w;
    double *biasw = bias->w;
    double *nrinw = NULL, *rinw = NULL, *routw = NULL;
    double *figow = NULL, *afigow = NULL;
    double *incw = NULL, *outcw = NULL;
    double *afw = NULL, *aiw = NULL, *agw = NULL, *aow = NULL;
    uint32_t wsx = weight->sx;
    uint32_t wsy = weight->sy;
    uint32_t hsize = lstm->hsize;

    cirno_vol_next_snapshot(lstm->rin);
    cirno_vol_next_snapshot(lstm->rout);
    cirno_vol_next_snapshot(lstm->figo);
    cirno_vol_next_snapshot(lstm->afigo);

    rinw = lstm->rin->w;
    nrinw = cirno_vol_next_w(lstm->rin, 1);
    routw = lstm->rout->w;
    figow = lstm->figo->w;
    afigow = lstm->afigo->w;
    incw = rinw + inVol->n + hsize;
    outcw = routw + hsize;
    afw = afigow;
    aiw = afigow + hsize;
    agw = afigow + (hsize * 2);
    aow = afigow + (hsize * 3);

    memcpy(rinw, inVol->w, sizeof(double) * inVol->n);

    for (i = 0; i < wsy; ++i) {
        sum = 0.0;

        j = wsx % 4;

        switch (j) {
            case 3: sum += weightw[wbase + 2] * rinw[2];
            case 2: sum += weightw[wbase + 1] * rinw[1];
            case 1: sum += weightw[wbase] * rinw[0];
        }

        for (; j < wsx; j += 4) {
            sum += weightw[wbase + j] * rinw[j]
                + weightw[wbase + j + 1] * rinw[j + 1]
                + weightw[wbase + j + 2] * rinw[j + 2]
                + weightw[wbase + j + 3] * rinw[j + 3];
        }

        figow[i] = sum + biasw[i];
        wbase += wsx;
    }

    for (i = 0; i < hsize * 3; ++i) {
        *(afigow++) = 1.0 / (1.0 + exp(-(*(figow++))));
    }

    for (i = 0; i < hsize; ++i) {
        *(afigow++) = tanh(*(figow++));
    }

    for (i = 0; i < hsize; ++i) {
        *(outcw++) = (*(aiw++) * *(agw++)) + (*(incw++) * *(afw++));
    }

    outcw = routw + hsize;

    for (i = 0; i < hsize; ++i) {
        *(routw++) = *(outcw++) * *(aow++);
    }

    routw = lstm->rout->w;

    memcpy(outVol->w, routw, sizeof(double) * outVol->n);
    memcpy(nrinw + inVol->n, routw, sizeof(double) * lstm->rout->n);
}

static void cirno_lstm_backward(CirnoLayer *layer, uint32_t t, uint8_t resetHidden)
{
    uint32_t i = 0, j = 0;
    CirnoLayerInfoLstm *lstm = layer->info.lstm;
    CirnoVol *weight = lstm->weight;
    CirnoVol *bias = lstm->bias;
    uint32_t outn = layer->outVol->n;
    double *rinwb = NULL, *rindwb = NULL;
    double *rinw = NULL, *rindw = NULL, *routw = NULL;
    double *routdw = NULL, *proutdw = NULL;
    double *outdw = NULL, *indw = NULL;
    double *outcw = NULL, *incdw = NULL;
    double *incw = NULL, *figow = NULL, *afigow = NULL;
    double *figodw = NULL, *afigodw = NULL;
    double *afw = NULL, *aiw = NULL, *agw = NULL, *aow = NULL;
    double *afdw = NULL, *aidw = NULL, *agdw = NULL, *aodw = NULL;
    double *fw = NULL, *iw = NULL, *gw = NULL, *ow = NULL;
    double *weightw = weight->w;
    double *weightdw = weight->dw;
    double *biasdw = bias->dw;
    double ho = 0.0;
    uint32_t wsx = weight->sx;
    uint32_t wsy = weight->sy;
    uint32_t hsize = lstm->hsize;
    double outdwi = 0.0;

    rinwb = cirno_vol_prev_w(lstm->rin, t);
    rindwb = cirno_vol_prev_dw(lstm->rin, t);
    routw = cirno_vol_prev_w(lstm->rout, t);
    routdw = cirno_vol_prev_dw(lstm->rout, t);
    proutdw = cirno_vol_prev_dw(lstm->rout, t + 1);
    outdw = cirno_vol_prev_dw(layer->outVol, t);
    indw = cirno_vol_prev_dw(layer->inVol, t);
    figow = cirno_vol_prev_w(lstm->figo, t);
    afigow = cirno_vol_prev_w(lstm->afigo, t);
    figodw = cirno_vol_prev_dw(lstm->figo, t);
    afigodw = cirno_vol_prev_dw(lstm->afigo, t);

    fw = figow;
    iw = figow + hsize;
    gw = figow + (hsize * 2);
    ow = figow + (hsize * 3);

    afw = afigow;
    aiw = afigow + hsize;
    agw = afigow + (hsize * 2);
    aow = afigow + (hsize * 3);

    afdw = afigodw;
    aidw = afigodw + hsize;
    agdw = afigodw + (hsize * 2);
    aodw = afigodw + (hsize * 3);

    incw = rinwb + layer->inVol->n + hsize;
    incdw = rindwb + layer->inVol->n + hsize;
    outcw = routw  + hsize;

    if (resetHidden) {
        memset(routdw, 0, sizeof(double) * lstm->rout->n);
    }

    for (i = 0; i < outn; ++i) {
        *(routdw++) += *(outdw++);
    }

    routdw = cirno_vol_prev_dw(lstm->rout, t);

    for (i = 0; i < hsize; ++i) {
        *(aodw++) = *(routdw++) * *(outcw++);
    }

    routdw = cirno_vol_prev_dw(lstm->rout, t);

    for (i = 0; i < hsize; ++i) {
        ho = *(routdw++) * *(ow++);

        *(agdw++) = ho * *(aiw++);
        *(aidw++) = ho * *(agw++);
        *(afdw++) = ho * *(incw++);
        *(incdw++) = ho * *(afw++);
    }

    for (i = 0; i < hsize * 3; ++i) {
        *(figodw++) = *(afigow) * (1.0 - *(afigow)) * *(afigodw);
        ++afigodw;
    }

    for (i = 0; i < hsize; ++i) {
        *(figodw++) = (1.0 - *(afigow) * *(afigow)) * *(afigodw);
        ++afigodw;
    }

    memset(rindwb, 0, sizeof(double) * lstm->rin->n);

    figodw = cirno_vol_prev_dw(lstm->figo, t);

    for (i = 0; i < wsy; ++i) {
        outdwi = *(figodw++);

        *(biasdw++) += outdwi;

        rinw = rinwb;
        rindw = rindwb;

        for (j = 0; j < wsx; ++j) {
            *(rindw++) += outdwi * *(weightw++);
            *(weightdw++) += outdwi * *(rinw++);
        }
    }

    memcpy(proutdw, rindwb + layer->inVol->n, sizeof(double) * lstm->rout->n);
    memcpy(indw, rindwb, sizeof(double) * layer->inVol->n);

    ++(lstm->weight->k);
    ++(lstm->bias->k);
}

static void cirno_lstm_clear_rin(CirnoLayer *layer)
{
    double *rinw = layer->info.lstm->rin->w;
    double *nrinw = cirno_vol_next_w(layer->info.lstm->rin, 1);
    uint32_t n = layer->info.lstm->rin->n;

    memset(rinw, 0, sizeof(double) * n);
    memset(nrinw, 0, sizeof(double) * n);
}

static double cirno_lstm_apply_grad(CirnoLayer *layer, CirnoTrainerOption *option)
{
    double loss = 0.0;
    CirnoLayerInfoLstm *lstm = layer->info.lstm;
    CirnoVol *weight = lstm->weight;
    CirnoVol *bias = lstm->bias;

    loss += cirno_trainer_apply(option, weight,
                                lstm->l1DecayMul, lstm->l2DecayMul);

    loss += cirno_trainer_apply(option, bias, 0.0, 0.0);

    return loss;
}

static void cirno_lstm_persist(MemBuf *buf, CirnoLayer *layer)
{
    mem_buf_write_uint32(buf, layer->type);
    cirno_vol_persist(buf, layer->outVol, FALSE);
    mem_buf_write_double(buf, layer->info.lstm->l1DecayMul);
    mem_buf_write_double(buf, layer->info.lstm->l2DecayMul);
    mem_buf_write_uint32(buf, layer->info.lstm->hsize);
    cirno_vol_persist(buf, layer->info.lstm->weight, TRUE);
    cirno_vol_persist(buf, layer->info.lstm->bias, TRUE);
    cirno_vol_persist(buf, layer->info.lstm->rin, FALSE);
    cirno_vol_persist(buf, layer->info.lstm->rout, FALSE);
    cirno_vol_persist(buf, layer->info.lstm->figo, FALSE);
    cirno_vol_persist(buf, layer->info.lstm->afigo, FALSE);
}

static CirnoVol *cirno_lstm_unpersist(lua_State *L, int index, CirnoVol *prevVol,
    MemBuf *buf, int newIndex)
{
    CirnoLayer *curLayer = lua_newuserdata(L, sizeof(CirnoLayer));
    luaL_getmetatable(L, "cirno.layer_instance");
    lua_setmetatable(L, -2);

    curLayer->type = kCirnoLayerLstm;
    curLayer->inVol = prevVol;
    curLayer->outVol = cirno_vol_unpersist(buf, FALSE);

    curLayer->info.lstm = malloc(sizeof(CirnoLayerInfoLstm));
    curLayer->info.lstm->l1DecayMul = mem_buf_read_double(buf);
    curLayer->info.lstm->l2DecayMul = mem_buf_read_double(buf);
    curLayer->info.lstm->hsize = mem_buf_read_uint32(buf);
    curLayer->info.lstm->weight = cirno_vol_unpersist(buf, TRUE);
    curLayer->info.lstm->bias = cirno_vol_unpersist(buf, TRUE);
    curLayer->info.lstm->rin = cirno_vol_unpersist(buf, FALSE);
    curLayer->info.lstm->rout = cirno_vol_unpersist(buf, FALSE);
    curLayer->info.lstm->figo = cirno_vol_unpersist(buf, FALSE);
    curLayer->info.lstm->afigo = cirno_vol_unpersist(buf, FALSE);

    lua_rawseti(L, index - 1, newIndex);

    return curLayer->outVol;
}

static void cirno_lstm_persist_grad(MemBuf *buf, CirnoLayer *layer)
{
    cirno_vol_persist_grad(buf, layer->info.lstm->weight);
    cirno_vol_persist_grad(buf, layer->info.lstm->bias);
}

static uint32_t cirno_lstm_unpersist_grad(MemBuf *buf, CirnoLayer *layer)
{
    uint32_t result = TRUE;

    result = cirno_vol_unpersist_grad(buf, layer->info.lstm->weight);

    if (result) {
        result = cirno_vol_unpersist_grad(buf, layer->info.lstm->bias);
    }

    return result;
}

/* ------------------------------------------------------------------------ */
/* MDPAGENT_LAYER */
/* ------------------------------------------------------------------------ */

static CirnoVol *cirno_mdpagent_new(lua_State *L, int index, CirnoVol *prevVol,
    CirnoLayerOption *option, int newIndex, uint32_t nSnapshot)
{
    CirnoLayer *curLayer = lua_newuserdata(L, sizeof(CirnoLayer));
    luaL_getmetatable(L, "cirno.layer_instance");
    lua_setmetatable(L, -2);

    curLayer->type = kCirnoLayerMdpAgent;
    curLayer->inVol = prevVol;
    curLayer->outVol = NULL;
    curLayer->info.mdpAgent = malloc(sizeof(CirnoLayerInfoMdpAgent));
    curLayer->info.mdpAgent->nSnapshot = nSnapshot;
    curLayer->info.mdpAgent->gamma = option->gamma;
    curLayer->info.mdpAgent->eps = option->eps;
    curLayer->info.mdpAgent->rewards = malloc(sizeof(double) * nSnapshot);
    curLayer->info.mdpAgent->actions = malloc(sizeof(uint32_t) * nSnapshot);
    curLayer->info.mdpAgent->curSnapshot = 0;

    lua_rawseti(L, index - 1, newIndex);

    return curLayer->outVol;
}

static void cirno_mdpagent_free(CirnoLayer *layer)
{
    free(layer->info.mdpAgent->rewards);
    free(layer->info.mdpAgent->actions);
    free(layer->info.mdpAgent);

    return;
}

static void cirno_mdpagent_forward(CirnoLayer *layer, uint8_t isTraining)
{
    UNREFERENCED_PARAMETER(layer);
    UNREFERENCED_PARAMETER(isTraining);

    return;
}

static void cirno_mdpagent_backward(CirnoLayer *layer, uint32_t t, uint8_t resetHidden)
{
    UNREFERENCED_PARAMETER(layer);
    UNREFERENCED_PARAMETER(t);
    UNREFERENCED_PARAMETER(resetHidden);

    return;
}

static void cirno_mdpagent_clear_rin(CirnoLayer *layer)
{
    UNREFERENCED_PARAMETER(layer);

    return;
}

static double cirno_mdpagent_apply_grad(CirnoLayer *layer, CirnoTrainerOption *option)
{
    UNREFERENCED_PARAMETER(layer);
    UNREFERENCED_PARAMETER(option);

    return 0.0;
}

static void cirno_mdpagent_persist(MemBuf *buf, CirnoLayer *layer)
{
    mem_buf_write_uint32(buf, layer->type);
    mem_buf_write_uint32(buf, layer->info.mdpAgent->nSnapshot);
    mem_buf_write_double(buf, layer->info.mdpAgent->gamma);
    mem_buf_write_double(buf, layer->info.mdpAgent->eps);

    return;
}

static CirnoVol *cirno_mdpagent_unpersist(lua_State *L, int index, CirnoVol *prevVol,
    MemBuf *buf, int newIndex)
{
    uint32_t nSnapshot = 0;
    CirnoLayer *curLayer = lua_newuserdata(L, sizeof(CirnoLayer));
    luaL_getmetatable(L, "cirno.layer_instance");
    lua_setmetatable(L, -2);

    nSnapshot = mem_buf_read_uint32(buf);

    curLayer->type = kCirnoLayerMdpAgent;
    curLayer->inVol = prevVol;
    curLayer->outVol = NULL;
    curLayer->info.mdpAgent = malloc(sizeof(CirnoLayerInfoMdpAgent));
    curLayer->info.mdpAgent->nSnapshot = nSnapshot;
    curLayer->info.mdpAgent->gamma = mem_buf_read_double(buf);
    curLayer->info.mdpAgent->eps = mem_buf_read_double(buf);
    curLayer->info.mdpAgent->rewards = malloc(sizeof(double) * nSnapshot);
    curLayer->info.mdpAgent->actions = malloc(sizeof(uint32_t) * nSnapshot);
    curLayer->info.mdpAgent->curSnapshot = 0;

    lua_rawseti(L, index - 1, newIndex);

    return curLayer->outVol;
}

static void cirno_mdpagent_persist_grad(MemBuf *buf, CirnoLayer *layer)
{
    UNREFERENCED_PARAMETER(buf);
    UNREFERENCED_PARAMETER(layer);

    return;
}

static uint32_t cirno_mdpagent_unpersist_grad(MemBuf *buf, CirnoLayer *layer)
{
    UNREFERENCED_PARAMETER(buf);
    UNREFERENCED_PARAMETER(layer);

    return TRUE;
}

/* ------------------------------------------------------------------------ */
/* RECURRENT_LAYER */
/* ------------------------------------------------------------------------ */

static CirnoVol *cirno_recurrent_new(lua_State *L, int index, CirnoVol *prevVol,
    CirnoLayerOption *option, int newIndex, uint32_t nSnapshot)
{
    uint32_t inCount = 0;
    uint32_t outCount = 0;
    CirnoLayer *curLayer = lua_newuserdata(L, sizeof(CirnoLayer));
    luaL_getmetatable(L, "cirno.layer_instance");
    lua_setmetatable(L, -2);

    curLayer->type = kCirnoLayerRecurrent;
    curLayer->inVol = prevVol;
    curLayer->outVol = cirno_vol_new(1, 1, 1,
        option->out, FALSE, nSnapshot);
    curLayer->info.recurrent = malloc(sizeof(CirnoLayerInfoRecurrent));
    curLayer->info.recurrent->l1DecayMul = option->l1DecayMul;
    curLayer->info.recurrent->l2DecayMul = option->l2DecayMul;
    curLayer->info.recurrent->rin = cirno_vol_new(1, 1, 1,
        prevVol->n + option->out + option->hidden, FALSE, nSnapshot);
    curLayer->info.recurrent->rout = cirno_vol_new(1, 1, 1,
        option->out + option->hidden, FALSE, nSnapshot);

    inCount = curLayer->info.recurrent->rin->n;
    outCount = curLayer->info.recurrent->rout->n;

    curLayer->info.recurrent->weight =
        cirno_random_vol_new(inCount, outCount, 1, 1, inCount, TRUE, 1);
    curLayer->info.recurrent->bias = cirno_vol_new(1, 1, 1, outCount, TRUE, 1);

    lua_rawseti(L, index - 1, newIndex);

    return curLayer->outVol;
}

static void cirno_recurrent_free(CirnoLayer *layer)
{
    cirno_vol_free(layer->outVol);
    cirno_vol_free(layer->info.recurrent->rin);
    cirno_vol_free(layer->info.recurrent->rout);
    cirno_vol_free(layer->info.recurrent->weight);
    cirno_vol_free(layer->info.recurrent->bias);
    free(layer->info.recurrent);
}

static void cirno_recurrent_forward(CirnoLayer *layer, uint8_t isTraining)
{
    UNREFERENCED_PARAMETER(isTraining);

    uint32_t i = 0, j = 0, wbase = 0;
    double sum = 0.0;
    CirnoVol *inVol = layer->inVol;
    CirnoVol *outVol = layer->outVol;
    CirnoLayerInfoRecurrent *recurrent = layer->info.recurrent;
    CirnoVol *weight = recurrent->weight;
    CirnoVol *bias = recurrent->bias;
    double *weightw = weight->w;
    double *biasw = bias->w;
    double *nrinw = NULL, *rinw = NULL, *routw = NULL;
    uint32_t wsx = weight->sx;
    uint32_t wsy = weight->sy;

    cirno_vol_next_snapshot(recurrent->rin);
    cirno_vol_next_snapshot(recurrent->rout);

    rinw = recurrent->rin->w;
    nrinw = cirno_vol_next_w(recurrent->rin, 1);
    routw = recurrent->rout->w;

    memcpy(rinw, inVol->w, sizeof(double) * inVol->n);

    for (i = 0; i < wsy; ++i) {
        sum = 0.0;

        j = wsx % 4;

        switch (j) {
            case 3: sum += weightw[wbase + 2] * rinw[2];
            case 2: sum += weightw[wbase + 1] * rinw[1];
            case 1: sum += weightw[wbase] * rinw[0];
        }

        for (; j < wsx; j += 4) {
            sum += weightw[wbase + j] * rinw[j]
                + weightw[wbase + j + 1] * rinw[j + 1]
                + weightw[wbase + j + 2] * rinw[j + 2]
                + weightw[wbase + j + 3] * rinw[j + 3];
        }

        routw[i] = sum + biasw[i];
        wbase += wsx;
    }

    memcpy(outVol->w, routw, sizeof(double) * outVol->n);
    memcpy(nrinw + inVol->n, routw, sizeof(double) * recurrent->rout->n);
}

static void cirno_recurrent_backward(CirnoLayer *layer, uint32_t t, uint8_t resetHidden)
{
    uint32_t i = 0, j = 0;
    CirnoLayerInfoRecurrent *recurrent = layer->info.recurrent;
    CirnoVol *weight = recurrent->weight;
    CirnoVol *bias = recurrent->bias;
    uint32_t outn = layer->outVol->n;
    double *rinwb = NULL, *rindwb = NULL;
    double *rinw = NULL, *rindw = NULL;
    double *routdw = NULL, *proutdw = NULL;
    double *outdw = NULL, *indw = NULL;
    double *weightw = weight->w;
    double *weightdw = weight->dw;
    double *biasdw = bias->dw;
    uint32_t wsx = weight->sx;
    uint32_t wsy = weight->sy;
    double outdwi = 0.0;

    rinwb = cirno_vol_prev_w(recurrent->rin, t);
    rindwb = cirno_vol_prev_dw(recurrent->rin, t);
    routdw = cirno_vol_prev_dw(recurrent->rout, t);
    proutdw = cirno_vol_prev_dw(recurrent->rout, t + 1);
    outdw = cirno_vol_prev_dw(layer->outVol, t);
    indw = cirno_vol_prev_dw(layer->inVol, t);

    if (resetHidden) {
        memset(routdw, 0, sizeof(double) * recurrent->rout->n);
    }

    for (i = 0; i < outn; ++i) {
        *(routdw++) += *(outdw++);
    }

    memset(rindwb, 0, sizeof(double) * recurrent->rin->n);

    routdw = cirno_vol_prev_dw(recurrent->rout, t);

    for (i = 0; i < wsy; ++i) {
        outdwi = *(routdw++);

        *(biasdw++) += outdwi;

        rinw = rinwb;
        rindw = rindwb;

        for (j = 0; j < wsx; ++j) {
            *(rindw++) += outdwi * *(weightw++);
            *(weightdw++) += outdwi * *(rinw++);
        }
    }

    memcpy(proutdw, rindwb + layer->inVol->n, sizeof(double) * recurrent->rout->n);
    memcpy(indw, rindwb, sizeof(double) * layer->inVol->n);

    ++(recurrent->weight->k);
    ++(recurrent->bias->k);
}

static void cirno_recurrent_clear_rin(CirnoLayer *layer)
{
    double *rinw = layer->info.recurrent->rin->w;
    double *nrinw = cirno_vol_next_w(layer->info.recurrent->rin, 1);
    uint32_t n = layer->info.recurrent->rin->n;

    memset(rinw, 0, sizeof(double) * n);
    memset(nrinw, 0, sizeof(double) * n);
}

static double cirno_recurrent_apply_grad(CirnoLayer *layer, CirnoTrainerOption *option)
{
    double loss = 0.0;
    CirnoLayerInfoRecurrent *recurrent = layer->info.recurrent;
    CirnoVol *weight = recurrent->weight;
    CirnoVol *bias = recurrent->bias;

    loss += cirno_trainer_apply(option, weight,
                                recurrent->l1DecayMul, recurrent->l2DecayMul);

    loss += cirno_trainer_apply(option, bias, 0.0, 0.0);

    return loss;
}

static void cirno_recurrent_persist(MemBuf *buf, CirnoLayer *layer)
{
    mem_buf_write_uint32(buf, layer->type);
    cirno_vol_persist(buf, layer->outVol, FALSE);
    mem_buf_write_double(buf, layer->info.recurrent->l1DecayMul);
    mem_buf_write_double(buf, layer->info.recurrent->l2DecayMul);
    cirno_vol_persist(buf, layer->info.recurrent->weight, TRUE);
    cirno_vol_persist(buf, layer->info.recurrent->bias, TRUE);
    cirno_vol_persist(buf, layer->info.recurrent->rin, FALSE);
    cirno_vol_persist(buf, layer->info.recurrent->rout, FALSE);
}

static CirnoVol *cirno_recurrent_unpersist(lua_State *L, int index, CirnoVol *prevVol,
    MemBuf *buf, int newIndex)
{
    CirnoLayer *curLayer = lua_newuserdata(L, sizeof(CirnoLayer));
    luaL_getmetatable(L, "cirno.layer_instance");
    lua_setmetatable(L, -2);

    curLayer->type = kCirnoLayerRecurrent;
    curLayer->inVol = prevVol;
    curLayer->outVol = cirno_vol_unpersist(buf, FALSE);

    curLayer->info.recurrent = malloc(sizeof(CirnoLayerInfoRecurrent));
    curLayer->info.recurrent->l1DecayMul = mem_buf_read_double(buf);
    curLayer->info.recurrent->l2DecayMul = mem_buf_read_double(buf);
    curLayer->info.recurrent->weight = cirno_vol_unpersist(buf, TRUE);
    curLayer->info.recurrent->bias = cirno_vol_unpersist(buf, TRUE);
    curLayer->info.recurrent->rin = cirno_vol_unpersist(buf, FALSE);
    curLayer->info.recurrent->rout = cirno_vol_unpersist(buf, FALSE);

    lua_rawseti(L, index - 1, newIndex);

    return curLayer->outVol;
}

static void cirno_recurrent_persist_grad(MemBuf *buf, CirnoLayer *layer)
{
    cirno_vol_persist_grad(buf, layer->info.recurrent->weight);
    cirno_vol_persist_grad(buf, layer->info.recurrent->bias);
}

static uint32_t cirno_recurrent_unpersist_grad(MemBuf *buf, CirnoLayer *layer)
{
    uint32_t result = TRUE;

    result = cirno_vol_unpersist_grad(buf, layer->info.recurrent->weight);

    if (result) {
        result = cirno_vol_unpersist_grad(buf, layer->info.recurrent->bias);
    }

    return result;
}

/* ------------------------------------------------------------------------ */
/* NOISE_LAYER */
/* ------------------------------------------------------------------------ */

static CirnoVol *cirno_noise_new(lua_State *L, int index, CirnoVol *prevVol,
    CirnoLayerOption *option, int newIndex, uint32_t nSnapshot)
{
    CirnoLayer *curLayer = lua_newuserdata(L, sizeof(CirnoLayer));
    luaL_getmetatable(L, "cirno.layer_instance");
    lua_setmetatable(L, -2);

    curLayer->type = kCirnoLayerNoise;
    curLayer->inVol = prevVol;
    curLayer->outVol = cirno_vol_new(prevVol->sx,
        prevVol->sy, prevVol->sz,
        prevVol->depth, FALSE, nSnapshot);
    curLayer->info.noise = malloc(sizeof(CirnoLayerInfoNoise));
    curLayer->info.noise->noiseType = option->noise;
    curLayer->info.noise->nw = option->nw;
    curLayer->info.noise->nsd = option->nsd;

    lua_rawseti(L, index - 1, newIndex);

    return curLayer->outVol;
}

static void cirno_noise_free(CirnoLayer *layer)
{
    cirno_vol_free(layer->outVol);
    free(layer->info.noise);
}

static void cirno_noise_forward(CirnoLayer *layer, uint8_t isTraining)
{
    uint32_t i = 0;
    CirnoVol *inVol = layer->inVol;
    CirnoVol *outVol = layer->outVol;
    uint32_t n = inVol->n;
    double *inw = inVol->w;
    double *outw = outVol->w;
    double rwidth = layer->info.noise->nw * 2;
    double nsd = layer->info.noise->nsd;

    if (isTraining) {
        switch (layer->info.noise->noiseType) {
            case kCirnoNoiseUniform: {
                for (i = 0; i < n; ++i) {
                    *(outw++) = *(inw++) + ((runif() - 0.5) * rwidth);
                }
            }
            break;
            case kCirnoNoiseRUniform: {
                for (i = 0; i < n; ++i) {
                    *(outw++) = *(inw++) * (((runif() - 0.5) * rwidth) + 1.0);
                }
            }
            break;
            case kCirnoNoiseGaussian: {
                for (i = 0; i < n; ++i) {
                    *(outw++) = *(inw++) + (rnorm() * nsd);
                }
            }
            break;
            case kCirnoNoiseRGaussian: {
                for (i = 0; i < n; ++i) {
                    *(outw++) = *(inw++) * ((rnorm() * nsd) + 1.0);
                }
            }
            break;
            default: {
            }
            break;
        }
    }
    else {
        memcpy(outw, inw, sizeof(double) * n);
    }
}

static void cirno_noise_backward(CirnoLayer *layer, uint32_t t, uint8_t resetHidden)
{
    UNREFERENCED_PARAMETER(resetHidden);

    CirnoVol *inVol = layer->inVol;
    CirnoVol *outVol = layer->outVol;
    uint32_t n = inVol->n;
    double *indw = NULL;
    double *outdw = NULL;

    indw = cirno_vol_prev_dw(inVol, t);
    outdw = cirno_vol_prev_dw(outVol, t);

    memcpy(indw, outdw, sizeof(double) * n);
}

static void cirno_noise_clear_rin(CirnoLayer *layer)
{
    UNREFERENCED_PARAMETER(layer);

    return;
}

static double cirno_noise_apply_grad(CirnoLayer *layer, CirnoTrainerOption *option)
{
    UNREFERENCED_PARAMETER(layer);
    UNREFERENCED_PARAMETER(option);
    return 0.0;
}

static void cirno_noise_persist(MemBuf *buf, CirnoLayer *layer)
{
    mem_buf_write_uint32(buf, layer->type);
    cirno_vol_persist(buf, layer->outVol, FALSE);
    mem_buf_write_uint32(buf, layer->info.noise->noiseType);
    mem_buf_write_double(buf, layer->info.noise->nw);
    mem_buf_write_double(buf, layer->info.noise->nsd);
}

static CirnoVol *cirno_noise_unpersist(lua_State *L, int index, CirnoVol *prevVol,
    MemBuf *buf, int newIndex)
{
    CirnoLayer *curLayer = lua_newuserdata(L, sizeof(CirnoLayer));
    luaL_getmetatable(L, "cirno.layer_instance");
    lua_setmetatable(L, -2);

    curLayer->type = kCirnoLayerNoise;
    curLayer->inVol = prevVol;
    curLayer->outVol = cirno_vol_unpersist(buf, FALSE);
    curLayer->info.noise = malloc(sizeof(CirnoLayerInfoNoise));
    curLayer->info.noise->noiseType = mem_buf_read_uint32(buf);
    curLayer->info.noise->nw = mem_buf_read_double(buf);
    curLayer->info.noise->nsd = mem_buf_read_double(buf);

    lua_rawseti(L, index - 1, newIndex);

    return curLayer->outVol;
}

static void cirno_noise_persist_grad(MemBuf *buf, CirnoLayer *layer)
{
    UNREFERENCED_PARAMETER(buf);
    UNREFERENCED_PARAMETER(layer);
}

static uint32_t cirno_noise_unpersist_grad(MemBuf *buf, CirnoLayer *layer)
{
    UNREFERENCED_PARAMETER(buf);
    UNREFERENCED_PARAMETER(layer);
    return TRUE;
}

/* ------------------------------------------------------------------------ */
/* LAYER_OPTION */
/* ------------------------------------------------------------------------ */

static int cirno_is_layer_option(lua_State *L, int index)
{
    int result = 0;

    lua_getmetatable(L, index);
    luaL_getmetatable(L, "cirno.layer_option");

    result = lua_rawequal(L, -1, -2);

    lua_pop(L, 2);

    return result;
}

static int cirno_layer_option_new(lua_State *L)
{
    CirnoLayerOption *layerOption =
        lua_newuserdata(L, sizeof(CirnoLayerOption));
    const char *strLayerType = NULL;
    const char *strActivation = NULL;
    const char *strNoise = NULL;
    const char *optionKey = NULL;

    luaL_getmetatable(L, "cirno.layer_option");
    lua_setmetatable(L, -2);

    lua_getfield(L, -2, "type");

    if (lua_isnil(L, -1)) {
        luaL_error(L, "type must be string");
    }
    else {
        strLayerType = luaL_checkstring(L, -1);
    }

    lua_getfield(L, -3, "act");

    if (!lua_isnil(L, -1)) {
        strActivation = luaL_checkstring(L, -1);
    }

    lua_getfield(L, -4, "noise");

    if (!lua_isnil(L, -1)) {
        strNoise = luaL_checkstring(L, -1);
    }

    if (!strcmp(strLayerType, "conv")) {
        layerOption->type = kCirnoLayerConv;
    }
    else if(!strcmp(strLayerType, "pool")) {
        layerOption->type = kCirnoLayerPool;
    }
    else if(!strcmp(strLayerType, "fc")) {
        layerOption->type = kCirnoLayerFullyConn;
    }
    else if(!strcmp(strLayerType, "input")) {
        layerOption->type = kCirnoLayerInput;
    }
    else if(!strcmp(strLayerType, "regression")) {
        layerOption->type = kCirnoLayerRegression;
    }
    else if(!strcmp(strLayerType, "softmax")) {
        layerOption->type = kCirnoLayerSoftmax;
    }
    else if(!strcmp(strLayerType, "svm")) {
        layerOption->type = kCirnoLayerSvm;
    }
    else if(!strcmp(strLayerType, "lstm")) {
        layerOption->type = kCirnoLayerLstm;
    }
    else if(!strcmp(strLayerType, "mdpagent")) {
        layerOption->type = kCirnoLayerMdpAgent;
    }
    else if(!strcmp(strLayerType, "recurrent")) {
        layerOption->type = kCirnoLayerRecurrent;
    }
    else {
        lua_pushfstring(L, "unknown layer type : %s", strLayerType);
        lua_error(L);
    }

    if (strActivation == NULL) {
        layerOption->act = kCirnoActivationNull;
    }
    else if (!strcmp(strActivation, "relu")) {
        layerOption->act = kCirnoActivationRelu;
    }
    else if (!strcmp(strActivation, "sigmoid")) {
        layerOption->act = kCirnoActivationSigmoid;
    }
    else if (!strcmp(strActivation, "tanh")) {
        layerOption->act = kCirnoActivationTanh;
    }
    else {
        lua_pushfstring(L, "unknown act : %s", strActivation);
        lua_error(L);
    }

    if (strNoise == NULL) {
        layerOption->noise = kCirnoNoiseNull;
    }
    else if (!strcmp(strNoise, "unif")) {
        layerOption->noise = kCirnoNoiseUniform;
    }
    else if (!strcmp(strNoise, "runif")) {
        layerOption->noise = kCirnoNoiseRUniform;
    }
    else if (!strcmp(strNoise, "gauss")) {
        layerOption->noise = kCirnoNoiseGaussian;
    }
    else if (!strcmp(strNoise, "rgauss")) {
        layerOption->noise = kCirnoNoiseRGaussian;
    }
    else {
        lua_pushfstring(L, "unknown noise : %s", strNoise);
        lua_error(L);
    }

    lua_pop(L, 3);

    layerOption->drop = 0.0;
    layerOption->l1DecayMul = 0.0;
    layerOption->l2DecayMul = 1.0;
    layerOption->out = 1;
    layerOption->hidden = 0;
    layerOption->sx = 1;
    layerOption->sy = 1;
    layerOption->sz = 1;
    layerOption->depth = 1;
    layerOption->filters = 1;
    layerOption->stride = 1;
    layerOption->padx = 0;
    layerOption->pady = 0;
    layerOption->padz = 0;
    layerOption->gamma = 0.8;
    layerOption->eps = 0.05;
    layerOption->nw = 0.1;
    layerOption->nsd = 0.1;

    lua_pushnil(L);

    while (lua_next(L, -3)) {
        optionKey = luaL_checkstring(L, -2);

        if (!strcmp(optionKey, "drop")) {
            layerOption->drop = (double)luaL_checknumber(L, -1);
        }
        else if (!strcmp(optionKey, "l1_decay_mul")) {
            layerOption->l1DecayMul = (double)luaL_checknumber(L, -1);
        }
        else if (!strcmp(optionKey, "l2_decay_mul")) {
            layerOption->l2DecayMul = (double)luaL_checknumber(L, -1);
        }
        else if (!strcmp(optionKey, "out")) {
            layerOption->out = (uint32_t)luaL_checkint(L, -1);
        }
        else if (!strcmp(optionKey, "hidden")) {
            layerOption->hidden = (uint32_t)luaL_checkint(L, -1);
        }
        else if (!strcmp(optionKey, "sx")) {
            layerOption->sx = (uint32_t)luaL_checkint(L, -1);
        }
        else if (!strcmp(optionKey, "sy")) {
            layerOption->sy = (uint32_t)luaL_checkint(L, -1);
        }
        else if (!strcmp(optionKey, "sz")) {
            layerOption->sz = (uint32_t)luaL_checkint(L, -1);
        }
        else if (!strcmp(optionKey, "depth")) {
            layerOption->depth = (uint32_t)luaL_checkint(L, -1);
        }
        else if (!strcmp(optionKey, "filters")) {
            layerOption->filters = (uint32_t)luaL_checkint(L, -1);
        }
        else if (!strcmp(optionKey, "stride")) {
            layerOption->stride = (uint32_t)luaL_checkint(L, -1);
        }
        else if (!strcmp(optionKey, "padx")) {
            layerOption->padx = (uint32_t)luaL_checkint(L, -1);
        }
        else if (!strcmp(optionKey, "pady")) {
            layerOption->pady = (uint32_t)luaL_checkint(L, -1);
        }
        else if (!strcmp(optionKey, "padz")) {
            layerOption->padz = (uint32_t)luaL_checkint(L, -1);
        }
        else if (!strcmp(optionKey, "gamma")) {
            layerOption->gamma = (double)luaL_checknumber(L, -1);
        }
        else if (!strcmp(optionKey, "eps")) {
            layerOption->eps = (double)luaL_checknumber(L, -1);
        }
        else if (!strcmp(optionKey, "nw")) {
            layerOption->nw = (double)luaL_checknumber(L, -1);
        }
        else if (!strcmp(optionKey, "nsd")) {
            layerOption->nsd = (double)luaL_checknumber(L, -1);
        }
        else if (!strcmp(optionKey, "act")) {
            ;
        }
        else if (!strcmp(optionKey, "type")) {
            ;
        }
        else if (!strcmp(optionKey, "noise")) {
            ;
        }
        else {
            lua_pushfstring(L, "unknown layer option : %s", optionKey);
            lua_error(L);
        }

        lua_pop(L, 1);
    }

    return 1;
}

/* ------------------------------------------------------------------------ */
/* TRAINER_OPTION */
/* ------------------------------------------------------------------------ */

static int cirno_is_trainer_option(lua_State *L, int index)
{
    int result = 0;

    lua_getmetatable(L, index);
    luaL_getmetatable(L, "cirno.trainer_option");

    result = lua_rawequal(L, -1, -2);

    lua_pop(L, 2);

    return result;
}

static int cirno_trainer_option_new(lua_State *L)
{
    CirnoTrainerOption *trainerOption =
        lua_newuserdata(L, sizeof(CirnoTrainerOption));
    const char *strTrainMethod = NULL;
    const char *optionKey = NULL;

    luaL_getmetatable(L, "cirno.trainer_option");
    lua_setmetatable(L, -2);

    lua_getfield(L, -2, "method");

    if (!lua_isnil(L, -1)) {
        strTrainMethod = luaL_checkstring(L, -1);
    }

    if (strTrainMethod == NULL) {
        trainerOption->trainMethod = kCirnoTrainerSgd;
    }
    else if (!strcmp(strTrainMethod, "adadelta")) {
        trainerOption->trainMethod = kCirnoTrainerAdaDelta;
    }
    else if (!strcmp(strTrainMethod, "adagrad")) {
        trainerOption->trainMethod = kCirnoTrainerAdaGrad;
    }
    else if (!strcmp(strTrainMethod, "adam")) {
        trainerOption->trainMethod = kCirnoTrainerAdam;
    }
    else if (!strcmp(strTrainMethod, "nesterov")) {
        trainerOption->trainMethod = kCirnoTrainerNesterov;
    }
    else if (!strcmp(strTrainMethod, "sgd")) {
        trainerOption->trainMethod = kCirnoTrainerSgd;
    }
    else if (!strcmp(strTrainMethod, "windowgrad")) {
        trainerOption->trainMethod = kCirnoTrainerWindowGrad;
    }
    else {
        lua_pushfstring(L, "unknown trainer type : %s", strTrainMethod);
        lua_error(L);
    }

    lua_pop(L, 1);

    trainerOption->lr = 0.01;
    trainerOption->l1Decay = 0.0;
    trainerOption->l2Decay = 0.0;
    trainerOption->momentum = 0.9;
    trainerOption->ro = 0.95;
    trainerOption->eps = 1e-8;
    trainerOption->beta1 = 0.9;
    trainerOption->beta2 = 0.999;
    trainerOption->maxGrad = 10.0;
    trainerOption->batch = 1;
    trainerOption->manual = FALSE;
    trainerOption->bpttStep = 0;

    lua_pushnil(L);

    while (lua_next(L, -3)) {
        optionKey = luaL_checkstring(L, -2);

        if (!strcmp(optionKey, "lr")) {
            trainerOption->lr = (double)luaL_checknumber(L, -1);
        }
        else if (!strcmp(optionKey, "l1_decay")) {
            trainerOption->l1Decay = (double)luaL_checknumber(L, -1);
        }
        else if (!strcmp(optionKey, "l2_decay")) {
            trainerOption->l2Decay = (double)luaL_checknumber(L, -1);
        }
        else if (!strcmp(optionKey, "momentum")) {
            trainerOption->momentum = (double)luaL_checknumber(L, -1);
        }
        else if (!strcmp(optionKey, "ro")) {
            trainerOption->ro = (double)luaL_checknumber(L, -1);
        }
        else if (!strcmp(optionKey, "eps")) {
            trainerOption->eps = (double)luaL_checknumber(L, -1);
        }
        else if (!strcmp(optionKey, "beta1")) {
            trainerOption->beta1 = (double)luaL_checknumber(L, -1);
        }
        else if (!strcmp(optionKey, "beta2")) {
            trainerOption->beta2 = (double)luaL_checknumber(L, -1);
        }
        else if (!strcmp(optionKey, "max_grad")) {
            trainerOption->maxGrad = (double)luaL_checknumber(L, -1);
        }
        else if (!strcmp(optionKey, "batch")) {
            trainerOption->batch = (uint32_t)luaL_checkint(L, -1);
        }
        else if (!strcmp(optionKey, "manual")) {
            trainerOption->manual = (uint32_t)lua_toboolean(L, -1);
        }
        else if (!strcmp(optionKey, "bptt_step")) {
            trainerOption->bpttStep = (uint32_t)luaL_checkint(L, -1);
        }
        else if (!strcmp(optionKey, "method")) {
            ;
        }
        else {
            lua_pushfstring(L, "unknown trainer option : %s", optionKey);
            lua_error(L);
        }

        lua_pop(L, 1);
    }

    trainerOption->k = 0;

    return 1;
}

static void cirno_trainer_option_persist(MemBuf *buf, CirnoTrainerOption *option)
{
    mem_buf_write_uint32(buf, option->trainMethod);
    mem_buf_write_double(buf, option->lr);
    mem_buf_write_double(buf, option->l1Decay);
    mem_buf_write_double(buf, option->l2Decay);
    mem_buf_write_double(buf, option->momentum);
    mem_buf_write_double(buf, option->ro);
    mem_buf_write_double(buf, option->eps);
    mem_buf_write_double(buf, option->beta1);
    mem_buf_write_double(buf, option->beta2);
    mem_buf_write_double(buf, option->maxGrad);
    mem_buf_write_uint32(buf, option->batch);
    mem_buf_write_uint32(buf, option->manual);
    mem_buf_write_uint32(buf, option->bpttStep);
}

static int cirno_trainer_option_unpersist(lua_State *L, MemBuf *buf)
{
    CirnoTrainerOption *trainerOption =
        lua_newuserdata(L, sizeof(CirnoTrainerOption));

    luaL_getmetatable(L, "cirno.trainer_option");
    lua_setmetatable(L, -2);

    trainerOption->trainMethod = mem_buf_read_uint32(buf);
    trainerOption->lr = mem_buf_read_double(buf);
    trainerOption->l1Decay = mem_buf_read_double(buf);
    trainerOption->l2Decay = mem_buf_read_double(buf);
    trainerOption->momentum = mem_buf_read_double(buf);
    trainerOption->ro = mem_buf_read_double(buf);
    trainerOption->eps = mem_buf_read_double(buf);
    trainerOption->beta1 = mem_buf_read_double(buf);
    trainerOption->beta2 = mem_buf_read_double(buf);
    trainerOption->maxGrad = mem_buf_read_double(buf);
    trainerOption->batch = mem_buf_read_uint32(buf);
    trainerOption->manual = mem_buf_read_uint32(buf);
    trainerOption->bpttStep = mem_buf_read_uint32(buf);
    trainerOption->k = 0;

    return 1;
}

/* ------------------------------------------------------------------------ */
/* TRAINER_APPLY */
/* ------------------------------------------------------------------------ */

static double cirno_trainer_apply(CirnoTrainerOption *option, CirnoVol *vol,
    double l1DecayMul, double l2DecayMul)
{
    uint32_t i = 0;
    double l1DecayLoss = 0.0, l2DecayLoss = 0.0;
    double l1Grad = 0.0, l2Grad = 0.0;
    double gij = 0.0, dx = 0.0;
    double biasCorr1 = 0.0, biasCorr2 = 0.0;
    double *p = vol->w;
    double *g = vol->dw;
    double *gsum = vol->gsum;
    double *xsum = vol->xsum;
    uint32_t n = vol->n;
    double lr = option->lr;
    double l1Decay = option->l1Decay * l1DecayMul;
    double l2Decay = option->l2Decay * l2DecayMul;
    uint32_t batch = vol->k;
    double momentum = option->momentum;
    double ro = option->ro;
    double eps = option->eps;
    double beta1 = option->beta1;
    double beta2 = option->beta2;
    double maxGrad = option->maxGrad;
    uint64_t k = option->k;

    if (batch == 0)
    {
        return 0.0;
    }

    switch (option->trainMethod) {
        case kCirnoTrainerAdaDelta: {
            for (i = 0; i < n; ++i) {
                l1DecayLoss += l1Decay * fabs(*p);
                l2DecayLoss += l2Decay * *p * *p / 2.0;

                l1Grad = l1Decay * (*p > 0.0 ? 1.0 : -1.0);
                l2Grad = l2Decay * *p;

                gij = (l2Grad + l1Grad + *g) / batch;

                *gsum = ro * *gsum + (1.0 - ro) * gij * gij;
                dx = -sqrt((*xsum + eps) / (*gsum + eps)) * gij;

                if (dx > maxGrad) { dx = maxGrad; }
                else if (dx < -maxGrad) { dx = -maxGrad; }

                *xsum = ro * *xsum + (1.0 - ro) * dx * dx;
                *p += dx;

                ++p; ++g; ++gsum; ++xsum;
            }
        }
        break;
        case kCirnoTrainerAdaGrad: {
            for (i = 0; i < n; ++i) {
                l1DecayLoss += l1Decay * fabs(*p);
                l2DecayLoss += l2Decay * *p * *p / 2.0;

                l1Grad = l1Decay * (*p > 0.0 ? 1.0 : -1.0);
                l2Grad = l2Decay * *p;

                gij = (l2Grad + l1Grad + *g) / batch;

                *gsum += gij * gij;
                dx = -lr / sqrt(*gsum + eps) * gij;

                if (dx > maxGrad) { dx = maxGrad; }
                else if (dx < -maxGrad) { dx = -maxGrad; }

                *p += dx;

                ++p; ++g; ++gsum;
            }
        }
        break;
        case kCirnoTrainerAdam: {
            for (i = 0; i < n; ++i) {
                l1DecayLoss += l1Decay * fabs(*p);
                l2DecayLoss += l2Decay * *p * *p / 2.0;

                l1Grad = l1Decay * (*p > 0.0 ? 1.0 : -1.0);
                l2Grad = l2Decay * *p;

                gij = (l2Grad + l1Grad + *g) / batch;

                *gsum = *gsum * beta1 + (1.0 - beta1) * gij;
                *xsum = *xsum * beta2 + (1.0 - beta2) * gij * gij;
                biasCorr1 = *gsum * (1.0 - pow(beta1, k));
                biasCorr2 = *xsum * (1.0 - pow(beta2, k));
                dx = -lr * biasCorr1 / (sqrt(biasCorr2) + eps);

                if (dx > maxGrad) { dx = maxGrad; }
                else if (dx < -maxGrad) { dx = -maxGrad; }

                *p += dx;

                ++p; ++g; ++gsum; ++xsum;
            }
        }
        break;
        case kCirnoTrainerNesterov: {
            for (i = 0; i < n; ++i) {
                l1DecayLoss += l1Decay * fabs(*p);
                l2DecayLoss += l2Decay * *p * *p / 2.0;

                l1Grad = l1Decay * (*p > 0 ? 1.0 : -1.0);
                l2Grad = l2Decay * *p;

                gij = (l2Grad + l1Grad + *g) / batch;

                dx = *gsum;

                *gsum = *gsum * momentum + lr * gij;
                dx = momentum * dx - (1.0 + momentum) * *gsum;

                if (dx > maxGrad) { dx = maxGrad; }
                else if (dx < -maxGrad) { dx = -maxGrad; }

                *p += dx;

                ++p; ++g; ++gsum;
            }
        }
        break;
        case kCirnoTrainerSgd: {
            for (i = 0; i < n; ++i) {
                l1DecayLoss += l1Decay * fabs(*p);
                l2DecayLoss += l2Decay * *p * *p / 2.0;

                l1Grad = l1Decay * (*p > 0.0 ? 1.0 : -1.0);
                l2Grad = l2Decay * *p;

                gij = (l2Grad + l1Grad + *g) / batch;

                if (momentum > 0.0) {
                    dx = momentum * *gsum - lr * gij;

                    if (dx > maxGrad) { dx = maxGrad; }
                    else if (dx < -maxGrad) { dx = -maxGrad; }

                    *gsum = dx;
                    *p += dx;
                } else {
                    if (gij > maxGrad) { gij = maxGrad; }
                    else if (gij < -maxGrad) { gij = -maxGrad; }

                    *p += -lr * gij;
                }

                ++p; ++g; ++gsum;
            }
        }
        break;
        case kCirnoTrainerWindowGrad: {
            for (i = 0; i < n; ++i) {
                l1DecayLoss += l1Decay * fabs(*p);
                l2DecayLoss += l2Decay * *p * *p / 2.0;

                l1Grad = l1Decay * (*p > 0.0 ? 1.0 : -1.0);
                l2Grad = l2Decay * *p;

                gij = (l2Grad + l1Grad + *g) / batch;

                *gsum = ro * *gsum + (1.0 - ro) * gij * gij;
                dx = -lr / sqrt(*gsum + eps) * gij;

                if (dx > maxGrad) { dx = maxGrad; }
                else if (dx < -maxGrad) { dx = -maxGrad; }

                *p += dx;

                ++p; ++g; ++gsum;
            }
        }
        break;
        default: {
        }
        break;
    }

    memset(vol->dw, 0, sizeof(double) * n);
    vol->k = 0;

    return l1DecayLoss + l2DecayLoss;
}

/* ------------------------------------------------------------------------ */
/* LAYER_INSTANCE */
/* ------------------------------------------------------------------------ */

static int cirno_layer_gc(lua_State *L)
{
    CirnoLayer *layer = luaL_checkudata(L, -1, "cirno.layer_instance");
    cirno_layer_free layerFreeFunc = cirnoFreeFuncs[layer->type];

    if (layerFreeFunc != NULL) {
        layerFreeFunc(layer);
    }

    return 0;
}

/* ------------------------------------------------------------------------ */
/* NETWORK */
/* ------------------------------------------------------------------------ */

static int cirno_is_network(lua_State *L, int index)
{
    int result = 0;

    lua_getmetatable(L, index);
    luaL_getmetatable(L, "cirno.network_instance");

    result = lua_rawequal(L, -1, -2);

    lua_pop(L, 2);

    return result;
}

static int cirno_network_new(lua_State *L)
{
    int i = 0;
    int optionLen = lua_objlen(L, -1);
    int curLayerIndex = 0;
    uint32_t nSnapshot = 0;
    uint32_t hasMdpAgent = 0;
    CirnoVol *prevOutVol = NULL;
    CirnoLayer *curLayer = NULL;
    CirnoTrainerOption *trainerOption = NULL;
    CirnoLayerOption* layerOption = NULL;

    lua_newtable(L);
    luaL_getmetatable(L, "cirno.network_instance");
    lua_setmetatable(L, -2);

    lua_newtable(L);

    for (i = 1; i <= optionLen; ++i) {
        lua_rawgeti(L, -3, i);

        if (cirno_is_trainer_option(L, -1)) {
            trainerOption = luaL_checkudata(L, -1, "cirno.trainer_option");
            lua_setfield(L, -3, "trainer");
        }
        else {
            layerOption = luaL_checkudata(L, -1, "cirno.layer_option");

            if (layerOption->type == kCirnoLayerMdpAgent) {
                hasMdpAgent = 1;
            }

            lua_pop(L, 1);
        }

    }

    if (trainerOption != NULL) {
        nSnapshot = trainerOption->bpttStep + hasMdpAgent + 1;
    }
    else {
        nSnapshot = hasMdpAgent + 1;
    }

    for (i = 1; i <= optionLen; ++i) {
        lua_rawgeti(L, -3, i);

        if (cirno_is_layer_option(L, -1)) {
            layerOption = luaL_checkudata(L, -1, "cirno.layer_option");

            if (curLayerIndex == 0 &&
                layerOption->type != kCirnoLayerInput) {
                luaL_error(L, "first layer must be input");
            }

            if (curLayerIndex != 0 && prevOutVol == NULL) {
                luaL_error(L, "cannot add layer after (regression, softmax, svm)");
            }

            if (layerOption->padx != 0
                || layerOption->pady != 0 || layerOption->padz != 0) {
                prevOutVol = cirno_pad_new(L, -2,
                    prevOutVol, layerOption, ++curLayerIndex, nSnapshot);
            }

            if (layerOption->stride != 1) {
                prevOutVol = cirno_stride_new(L, -2,
                    prevOutVol, layerOption, ++curLayerIndex, nSnapshot);
            }

            switch (layerOption->type) {
                case kCirnoLayerConv: {
                    prevOutVol = cirno_conv_new(L, -2,
                        prevOutVol, layerOption, ++curLayerIndex, nSnapshot);
                }
                break;
                case kCirnoLayerPool: {
                    prevOutVol = cirno_pool_new(L, -2,
                        prevOutVol, layerOption, ++curLayerIndex, nSnapshot);
                }
                break;
                case kCirnoLayerFullyConn: {
                    prevOutVol = cirno_fullyconn_new(L, -2,
                        prevOutVol, layerOption, ++curLayerIndex, nSnapshot);
                }
                break;
                case kCirnoLayerInput: {
                    prevOutVol = cirno_input_new(L, -2,
                        prevOutVol, layerOption, ++curLayerIndex, nSnapshot);
                }
                break;
                case kCirnoLayerRegression: {
                    prevOutVol = cirno_fullyconn_new(L, -2,
                        prevOutVol, layerOption, ++curLayerIndex, nSnapshot);
                    prevOutVol = cirno_regression_new(L, -2,
                        prevOutVol, layerOption, ++curLayerIndex, nSnapshot);
                }
                break;
                case kCirnoLayerSoftmax: {
                    prevOutVol = cirno_fullyconn_new(L, -2,
                        prevOutVol, layerOption, ++curLayerIndex, nSnapshot);
                    prevOutVol = cirno_softmax_new(L, -2,
                        prevOutVol, layerOption, ++curLayerIndex, nSnapshot);
                }
                break;
                case kCirnoLayerSvm: {
                    prevOutVol = cirno_fullyconn_new(L, -2,
                        prevOutVol, layerOption, ++curLayerIndex, nSnapshot);
                    prevOutVol = cirno_svm_new(L, -2,
                        prevOutVol, layerOption, ++curLayerIndex, nSnapshot);
                }
                break;
                case kCirnoLayerLstm: {
                    prevOutVol = cirno_lstm_new(L, -2,
                        prevOutVol, layerOption, ++curLayerIndex, nSnapshot);
                }
                break;
                case kCirnoLayerMdpAgent: {
                    prevOutVol = cirno_fullyconn_new(L, -2,
                        prevOutVol, layerOption, ++curLayerIndex, nSnapshot);
                    prevOutVol = cirno_mdpagent_new(L, -2,
                        prevOutVol, layerOption, ++curLayerIndex, nSnapshot);
                }
                break;
                case kCirnoLayerRecurrent: {
                    prevOutVol = cirno_recurrent_new(L, -2,
                        prevOutVol, layerOption, ++curLayerIndex, nSnapshot);
                }
                break;
                default: {
                    lua_pushfstring(L, "unknown layer type %d\n", layerOption->type);
                    lua_error(L);
                }
                break;
            }

            if (layerOption->act != kCirnoActivationNull) {
                if (prevOutVol == NULL) {
                    luaL_error(L, "(regression, softmax, svm) cannot have act");
                }

                switch (layerOption->act) {
                    case kCirnoActivationRelu: {
                        prevOutVol = cirno_relu_new(L, -2,
                            prevOutVol, layerOption, ++curLayerIndex, nSnapshot);
                    }
                    break;
                    case kCirnoActivationSigmoid: {
                        prevOutVol = cirno_sigmoid_new(L, -2,
                            prevOutVol, layerOption, ++curLayerIndex, nSnapshot);
                    }
                    break;
                    case kCirnoActivationTanh: {
                        prevOutVol = cirno_tanh_new(L, -2,
                            prevOutVol, layerOption, ++curLayerIndex, nSnapshot);
                    }
                    break;
                    default: {
                        lua_pushfstring(L, "unknown act %d\n", layerOption->act);
                        lua_error(L);
                    }
                    break;
                }
            }

            if (layerOption->noise != kCirnoNoiseNull) {
                prevOutVol = cirno_noise_new(L, -2,
                    prevOutVol, layerOption, ++curLayerIndex, nSnapshot);
            }

            if (layerOption->drop != 0.0) {
                if (prevOutVol == NULL) {
                    luaL_error(L, "(regression, softmax, svm) cannot have drop_prob");
                }

                prevOutVol = cirno_dropout_new(L, -2,
                    prevOutVol, layerOption, ++curLayerIndex, nSnapshot);
            }

            lua_pop(L, 1);
        }
        else if (cirno_is_trainer_option(L, -1)) {
            lua_pop(L, 1);
        }
        else {
            luaL_error(L, "invalid data in network definition");
        }
    }

    lua_rawgeti(L, -1, curLayerIndex);
    curLayer = luaL_checkudata(L, -1, "cirno.layer_instance");

    if (curLayer->type != kCirnoLayerRegression &&
        curLayer->type != kCirnoLayerSoftmax &&
        curLayer->type != kCirnoLayerSvm &&
        curLayer->type != kCirnoLayerMdpAgent) {
        luaL_error(L, "last layer must be (regression, softmax, svm, mdpagent)");
    }

    lua_pop(L, 1);

    lua_setfield(L, -2, "layers");

    lua_getfield(L, -1, "trainer");

    if (lua_isnil(L, -1)) {
        lua_newtable(L);
        cirno_trainer_option_new(L);
        lua_setfield(L, -4, "trainer");
        lua_pop(L, 1);
    }

    lua_pop(L, 1);

    return 1;
}

static CirnoVol *cirno_network_forward(lua_State *L,
    int networkIndex, int inputIndex, uint8_t isTraining,
    uint32_t *lastLayerType, uint32_t *action)
{
    uint32_t i = 0;
    uint32_t layerLen = 0;
    uint32_t inputLen = 0;
    uint32_t n = 0;
    double *w = NULL;
    double maxq = -DBL_MAX;
    uint32_t maxi = 0;
    CirnoLayer *curLayer = NULL;
    CirnoVol *outVol = NULL;
    CirnoLayerInfoMdpAgent *mdpAgent = NULL;

    if (!cirno_is_network(L, networkIndex)) {
        luaL_error(L, "argument is not network");
    }

    inputLen = lua_objlen(L, inputIndex);

    lua_getfield(L, networkIndex, "layers");

    layerLen = lua_objlen(L, -1);

    if (layerLen == 0) {
        luaL_error(L, "no layers");
    }

    lua_rawgeti(L, -1, 1);

    curLayer = luaL_checkudata(L, -1, "cirno.layer_instance");

    if (isTraining == TRUE) {
        cirno_vol_next_snapshot(curLayer->outVol);
    }

    n = curLayer->outVol->n;
    w = curLayer->outVol->w;

    if (inputLen != n) {
        lua_pushfstring(L,
            "input length mismatch expected = %d, input = %d", n, inputLen);
        lua_error(L);
    }

    for (i = 1; i <= n; ++i) {
        lua_rawgeti(L, inputIndex - 2, i);
        *(w++) = lua_tonumber(L, -1);
        lua_pop(L, 1);
    }

    lua_pop(L, 1);

    for (i = 2; i <= layerLen; ++i) {
        lua_rawgeti(L, -1, i);
        curLayer = luaL_checkudata(L, -1, "cirno.layer_instance");

        if (isTraining == TRUE && i != layerLen) {
            cirno_vol_next_snapshot(curLayer->outVol);
        }

        cirnoForwardFuncs[curLayer->type](curLayer, isTraining);
        lua_pop(L, 1);
    }

    lua_rawgeti(L, -1, layerLen);
    curLayer = luaL_checkudata(L, -1, "cirno.layer_instance");

    switch (curLayer->type) {
        case kCirnoLayerRegression: {
            outVol = curLayer->inVol;
        }
        break;
        case kCirnoLayerSoftmax: {
            outVol = curLayer->outVol;
        }
        break;
        case kCirnoLayerSvm: {
            outVol = curLayer->inVol;
        }
        break;
        case kCirnoLayerMdpAgent: {
            mdpAgent = curLayer->info.mdpAgent;

            if (isTraining && (runif() < mdpAgent->eps)) {
                *action = mt19937_rand() % curLayer->inVol->n;
            }
            else {
                w = curLayer->inVol->w;
                n = curLayer->inVol->n;

                for (i = 0; i < n; ++i) {
                    if (maxq < w[i]) {
                        maxq = w[i];
                        maxi = i;
                    }
                }

                *action = maxi;
            }

            if (isTraining) {
                mdpAgent->curSnapshot =
                    (mdpAgent->curSnapshot + 1) % mdpAgent->nSnapshot;
                mdpAgent->actions[mdpAgent->curSnapshot] = *action;
            }
        }
        break;
        default: {
            luaL_error(L, "invalid last layer");
        }
        break;
    }

    *lastLayerType = curLayer->type;

    lua_pop(L, 2);

    return outVol;
}

static double cirno_network_backward_with_desired(lua_State *L,
    CirnoTrainerOption *option, int networkIndex, int desiredIndex)
{
    uint32_t i = 0;
    uint32_t layerLen = 0;
    uint32_t desiredLen = 0;
    uint32_t n = 0;
    uint32_t nSnapshot = option->bpttStep + 1;
    uint32_t t = 0;
    double *w = NULL;
    double *dw = NULL;
    double loss = 0.0;
    double desiredVal = 0.0;
    double dy = 0.0;
    CirnoLayer *curLayer = NULL;
    CirnoVol *lossVol = NULL;

    if (!cirno_is_network(L, networkIndex)) {
        luaL_error(L, "argument is not network");
    }

    desiredLen = lua_objlen(L, desiredIndex);

    lua_getfield(L, networkIndex, "layers");

    layerLen = lua_objlen(L, -1);

    if (layerLen == 0) {
        luaL_error(L, "no layers");
    }

    lua_rawgeti(L, -1, layerLen);

    curLayer = luaL_checkudata(L, -1, "cirno.layer_instance");

    if (curLayer->type == kCirnoLayerRegression) {
        lossVol = curLayer->inVol;
        n = lossVol->n;
        w = lossVol->w;
        dw = lossVol->dw;

        if (desiredLen != n) {
            lua_pushfstring(L,
                "desired length mismatch expected = %d, desired = %d",
                n, desiredLen);
            lua_error(L);
        }

        for (i = 1; i <= n; ++i) {
            lua_rawgeti(L, desiredIndex - 2, i);
            desiredVal = lua_tonumber(L, -1);

            dy = *(w++) - desiredVal;
            *(dw++) = dy;

            loss += 0.5 * dy * dy;

            lua_pop(L, 1);
        }
    }
    else {
        luaL_error(L, "invalid last layer");
    }

    lua_pop(L, 1);

    for (t = 0; t < nSnapshot; ++t) {
        for (i = layerLen; i >= 2; --i) {
            lua_rawgeti(L, -1, i);
            curLayer = luaL_checkudata(L, -1, "cirno.layer_instance");
            cirnoBackwardFuncs[curLayer->type](curLayer, t, t == 0);
            lua_pop(L, 1);
        }
    }

    lua_pop(L, 1);

    return loss;
}

static double cirno_network_backward_with_class(lua_State *L,
    CirnoTrainerOption *option, int networkIndex, uint32_t classNo)
{
    uint32_t i = 0;
    uint32_t layerLen = 0;
    uint32_t n = 0;
    uint32_t nSnapshot = option->bpttStep + 1;
    uint32_t t = 0;
    double *inw = NULL;
    double *indw = NULL;
    double *outw = NULL;
    double loss = 0.0;
    double indicator = 0.0;
    double classScore = 0.0;
    double classDiff = 0.0;
    CirnoLayer *curLayer = NULL;

    --classNo;

    if (!cirno_is_network(L, networkIndex)) {
        luaL_error(L, "argument is not network");
    }

    lua_getfield(L, networkIndex, "layers");

    layerLen = lua_objlen(L, -1);

    if (layerLen == 0) {
        luaL_error(L, "no layers");
    }

    lua_rawgeti(L, -1, layerLen);

    curLayer = luaL_checkudata(L, -1, "cirno.layer_instance");

    n = curLayer->inVol->n;
    inw = curLayer->inVol->w;
    indw = curLayer->inVol->dw;

    if (classNo >= n) {
        luaL_error(L, "invalid desired class number");
    }

    switch (curLayer->type) {
        case kCirnoLayerSoftmax: {
            outw = curLayer->outVol->w;

            for (i = 0; i < n; ++i) {
                indicator = (i == classNo) ? 1.0 : 0.0;

                *(indw++) = -(indicator - *(outw++));
            }

            loss = 1.0 - curLayer->outVol->w[classNo];
        }
        break;
        case kCirnoLayerSvm: {
            memset(indw, 0, sizeof(double) * n);
            classScore = inw[classNo];

            --inw; --indw;

            for (i = 0; i < n; ++i) {
                ++inw; ++indw;

                if (i == classNo) {
                    continue;
                }

                classDiff = -classScore + *inw + 1.0; /* margin == 1.0 */

                if (classDiff > 0.0) {
                    *indw += 1.0;
                    curLayer->inVol->dw[classNo] -= 1.0;
                    loss += classDiff;
                }
            }
        }
        break;
        default: {
            luaL_error(L, "invalid last layer");
        }
        break;
    }

    lua_pop(L, 1);

    for (t = 0; t < nSnapshot; ++t) {
        for (i = layerLen; i >= 2; --i) {
            lua_rawgeti(L, -1, i);
            curLayer = luaL_checkudata(L, -1, "cirno.layer_instance");
            cirnoBackwardFuncs[curLayer->type](curLayer, t, t == 0);
            lua_pop(L, 1);
        }
    }

    lua_pop(L, 1);

    return loss;
}

static double cirno_network_backward_with_reward(lua_State *L,
    CirnoTrainerOption *option, int networkIndex, double reward)
{
    uint32_t i = 0;
    uint32_t layerLen = 0;
    uint32_t curSnapshot = 0;
    uint32_t prevSnapshot = 0;
    uint32_t paction = 0;
    uint32_t n = 0;
    uint32_t nSnapshot = option->bpttStep + 2;
    uint32_t t = 0;
    double *pindw = NULL;
    double *inw = NULL;
    double *pinw = NULL;
    double maxq = -DBL_MAX;
    double loss = 0.0;
    double dy = 0.0;
    double preward = 0.0;
    CirnoLayer *curLayer = NULL;
    CirnoLayerInfoMdpAgent *mdpAgent = NULL;

    if (!cirno_is_network(L, networkIndex)) {
        luaL_error(L, "argument is not network");
    }

    lua_getfield(L, networkIndex, "layers");

    layerLen = lua_objlen(L, -1);

    if (layerLen == 0) {
        luaL_error(L, "no layers");
    }

    lua_rawgeti(L, -1, layerLen);

    curLayer = luaL_checkudata(L, -1, "cirno.layer_instance");

    if (curLayer->type != kCirnoLayerMdpAgent) {
        luaL_error(L, "layer must be mdpagent");
    }

    mdpAgent = curLayer->info.mdpAgent;

    n = curLayer->inVol->n;
    curSnapshot = mdpAgent->curSnapshot;
    prevSnapshot = (curSnapshot - 1 + nSnapshot) % nSnapshot;

    mdpAgent->rewards[curSnapshot] = reward;

    inw = cirno_vol_prev_w(curLayer->inVol, 0);
    pinw = cirno_vol_prev_w(curLayer->inVol, 1);
    pindw = cirno_vol_prev_dw(curLayer->inVol, 1);

    preward = mdpAgent->rewards[prevSnapshot];
    paction = mdpAgent->actions[prevSnapshot];

    for (i = 0; i < n; ++i) {
        if (maxq < inw[i]) {
            maxq = inw[i];
        }
    }

    memset(pindw, 0, sizeof(double) * n);

    dy = pinw[paction] - (preward + mdpAgent->gamma * maxq);
    pindw[paction] = dy;

    loss = 0.5 * dy * dy;

    lua_pop(L, 1);

    for (t = 1; t < nSnapshot; ++t) {
        for (i = layerLen; i >= 2; --i) {
            lua_rawgeti(L, -1, i);
            curLayer = luaL_checkudata(L, -1, "cirno.layer_instance");
            cirnoBackwardFuncs[curLayer->type](curLayer, t, t == 1);
            lua_pop(L, 1);
        }
    }

    lua_pop(L, 1);

    return loss;
}

static double cirno_network_backward_with_index_value(lua_State *L,
    CirnoTrainerOption *option, int networkIndex, uint32_t index, double value)
{
    uint32_t i = 0;
    uint32_t layerLen = 0;
    uint32_t n = 0;
    uint32_t nSnapshot = option->bpttStep + 1;
    uint32_t t = 0;
    double loss = 0.0;
    double dy = 0.0;
    CirnoLayer *curLayer = NULL;

    --index;

    if (!cirno_is_network(L, networkIndex)) {
        luaL_error(L, "argument is not network");
    }

    lua_getfield(L, networkIndex, "layers");

    layerLen = lua_objlen(L, -1);

    if (layerLen == 0) {
        luaL_error(L, "no layers");
    }

    lua_rawgeti(L, -1, layerLen);

    curLayer = luaL_checkudata(L, -1, "cirno.layer_instance");

    n = curLayer->inVol->n;

    if (index >= n) {
        luaL_error(L, "invalid index");
    }

    memset(curLayer->inVol->dw, 0, sizeof(double) * n);

    if (curLayer->type == kCirnoLayerRegression) {
        dy = curLayer->inVol->w[index] - value;
        curLayer->inVol->dw[index] = dy;
        loss = 0.5 * dy * dy;
    }
    else {
        luaL_error(L, "invalid last layer");
    }

    lua_pop(L, 1);

    for (t = 0; t < nSnapshot; ++t) {
        for (i = layerLen; i >= 2;--i) {
            lua_rawgeti(L, -1, i);
            curLayer = luaL_checkudata(L, -1, "cirno.layer_instance");
            cirnoBackwardFuncs[curLayer->type](curLayer, t, t == 0);
            lua_pop(L, 1);
        }
    }

    lua_pop(L, 1);

    return loss;
}

static double cirno_network_apply_grad_inner(lua_State *L,
    CirnoTrainerOption *option, int networkIndex)
{
    uint32_t i = 0;
    uint32_t layerLen = 0;
    double loss = 0.0;
    CirnoLayer *curLayer = NULL;

    if (!cirno_is_network(L, networkIndex)) {
        luaL_error(L, "argument is not network");
    }

    lua_getfield(L, networkIndex, "layers");

    layerLen = lua_objlen(L, -1);

    if (layerLen == 0) {
        luaL_error(L, "no layers");
    }

    for (i = 2; i < layerLen; ++i) {
        lua_rawgeti(L, -1, i);
        curLayer = luaL_checkudata(L, -1, "cirno.layer_instance");
        loss += cirnoApplyGradFuncs[curLayer->type](curLayer, option);
        lua_pop(L, 1);
    }

    lua_pop(L, 1);

    return loss;
}

static int cirno_network_predict(lua_State *L)
{
    uint32_t i = 0;
    uint32_t n = 0;
    double *w = NULL;
    uint32_t lastLayerType = 0;
    CirnoVol *outVol = NULL;
    uint32_t action = 0;

    outVol = cirno_network_forward(L, -2, -1, FALSE, &lastLayerType, &action);

    if (lastLayerType == kCirnoLayerMdpAgent) {
        lua_pushinteger(L, action + 1);
    }
    else {
        n = outVol->n;
        w = outVol->w;

        lua_newtable(L);

        for (i = 0; i < n; ++i) {
            lua_pushnumber(L, w[i]);
            lua_rawseti(L, -2, i + 1);
        }
    }

    return 1;
}

static int cirno_network_train(lua_State *L)
{
    double loss = 0;
    CirnoTrainerOption *option = NULL;
    int networkIndex = 0;
    int inputIndex = 0;
    int arg3Index = 0;
    int arg4Index = 0;
    int arg3Val = 0;
    double arg4Val = 0.0;
    double reward = 0.0;
    uint32_t argLen = lua_gettop(L);
    uint32_t lastLayerType = 0;
    uint32_t action = 0;
    uint32_t mdpStep = 0;

    if (argLen != 3 && argLen != 4) {
        luaL_error(L, "invalid argument count");
    }

    networkIndex = -argLen;
    inputIndex = -argLen + 1;
    arg3Index = -argLen + 2;
    arg4Index = -argLen + 3;

    lua_getfield(L, networkIndex, "trainer");

    if (lua_isnil(L, -1)) {
        luaL_error(L, "network does not have trainer");
    }

    option = luaL_checkudata(L, -1, "cirno.trainer_option");

    cirno_network_forward(L, networkIndex - 1, inputIndex - 1, TRUE, &lastLayerType, &action);

    ++(option->k);

    if (lastLayerType == kCirnoLayerMdpAgent) {
        mdpStep = 1;
    }
    else {
        mdpStep = 0;
    }

    if (option->k >= option->bpttStep + mdpStep) {
        if (argLen == 3) {
            if (lua_istable(L, arg3Index - 1)) {
                loss = cirno_network_backward_with_desired(L, option,
                    networkIndex - 1, arg3Index - 1);
            }
            else if (lua_isnumber(L, arg3Index - 1)) {
                if (lastLayerType == kCirnoLayerMdpAgent) {
                    reward = (double)lua_tonumber(L, arg3Index - 1);

                    loss = cirno_network_backward_with_reward(L, option,
                        networkIndex - 1, reward);
                }
                else {
                    arg3Val = lua_tointeger(L, arg3Index - 1);

                    loss = cirno_network_backward_with_class(L, option,
                        networkIndex - 1, (uint32_t)arg3Val);
                }
            }
            else {
                luaL_error(L, "invalid argument type");
            }
        }
        else {
            if (!lua_isnumber(L, arg3Index - 1)) {
                luaL_error(L, "invalid argument type");
            }
            if (!lua_isnumber(L, arg4Index - 1)) {
                luaL_error(L, "invalid argument type");
            }

            arg3Val = lua_tointeger(L, arg3Index - 1);
            arg4Val = lua_tonumber(L, arg4Index - 1);

            loss = cirno_network_backward_with_index_value(L, option,
                networkIndex - 1, (uint32_t)arg3Val, arg4Val);
        }

        if ((!option->manual) && ((option->k) % option->batch == 0)) {
            loss += cirno_network_apply_grad_inner(L, option, networkIndex - 1);
        }
    }

    lua_pop(L, 1);

    if (lastLayerType == kCirnoLayerMdpAgent) {
        lua_pushinteger(L, action + 1);
        lua_pushnumber(L, loss);

        return 2;
    }

    lua_pushnumber(L, loss);

    return 1;
}

static int cirno_network_clear_rin(lua_State *L)
{
    uint32_t i = 0;
    uint32_t layerLen = 0;
    CirnoLayer *curLayer = NULL;

    if (!cirno_is_network(L, -1)) {
        luaL_error(L, "argument is not network");
    }

    lua_getfield(L, -1, "layers");

    layerLen = lua_objlen(L, -1);

    if (layerLen == 0) {
        luaL_error(L, "no layers");
    }

    for (i = 1; i <= layerLen; ++i) {
        lua_rawgeti(L, -1, i);
        curLayer = luaL_checkudata(L, -1, "cirno.layer_instance");
        cirnoClearRinFuncs[curLayer->type](curLayer);
        lua_pop(L, 1);
    }

    lua_pop(L, 1);

    return 0;
}

static int cirno_network_persist(lua_State *L)
{
    uint32_t i = 0;
    uint32_t layerLen = 0;
    CirnoTrainerOption *option = NULL;
    CirnoLayer *curLayer = NULL;
    MemBuf *buf = mem_buf_write_new();

    if (!cirno_is_network(L, -1)) {
        luaL_error(L, "argument is not network");
    }

    lua_getfield(L, -1, "trainer");

    if (lua_isnil(L, -1)) {
        luaL_error(L, "network does not have trainer");
    }

    option = luaL_checkudata(L, -1, "cirno.trainer_option");

    cirno_trainer_option_persist(buf, option);

    lua_pop(L, 1);

    lua_getfield(L, -1, "layers");

    layerLen = lua_objlen(L, -1);

    mem_buf_write_uint32(buf, layerLen);

    if (layerLen == 0) {
        luaL_error(L, "no layers");
    }

    for (i = 1; i <= layerLen; ++i) {
        lua_rawgeti(L, -1, i);
        curLayer = luaL_checkudata(L, -1, "cirno.layer_instance");
        cirnoPersistFuncs[curLayer->type](buf, curLayer);
        lua_pop(L, 1);
    }

    lua_pop(L, 1);

    mem_buf_toluastring(L, buf);
    mem_buf_free(buf);

    return 1;
}

static int cirno_network_unpersist(lua_State *L)
{
    const char *data = NULL;
    size_t dataLen = 0;
    uint32_t layerLen = 0;
    uint32_t type = 0;
    uint32_t i = 0;
    MemBuf *buf = NULL;
    CirnoVol *prevOutVol = NULL;

    data = luaL_checklstring(L, -1, &dataLen);

    buf = mem_buf_read_new(data, dataLen);

    lua_newtable(L);
    luaL_getmetatable(L, "cirno.network_instance");
    lua_setmetatable(L, -2);

    cirno_trainer_option_unpersist(L, buf);
    lua_setfield(L, -2, "trainer");

    lua_newtable(L);
    layerLen = mem_buf_read_uint32(buf);

    for (i = 1; i <= layerLen; ++i) {
        type = mem_buf_read_uint32(buf);
        prevOutVol = cirnoUnpersistFuncs[type](L, -1,
            prevOutVol, buf, i);
    }

    lua_setfield(L, -2, "layers");
    mem_buf_free(buf);

    return 1;
}

static int cirno_network_persist_grad(lua_State *L)
{
    uint32_t i = 0;
    uint32_t layerLen = 0;
    CirnoTrainerOption *option = NULL;
    CirnoLayer *curLayer = NULL;
    MemBuf *buf = mem_buf_write_new();

    if (!cirno_is_network(L, -1)) {
        luaL_error(L, "argument is not network");
    }

    lua_getfield(L, -1, "trainer");

    if (lua_isnil(L, -1)) {
        luaL_error(L, "network does not have trainer");
    }

    option = luaL_checkudata(L, -1, "cirno.trainer_option");

    if (option->manual != TRUE) {
        luaL_error(L, "persist_grad is only available on manual_apply");
    }

    mem_buf_write_uint32(buf, option->k);
    option->k = 0;

    lua_pop(L, 1);

    lua_getfield(L, -1, "layers");
    layerLen = lua_objlen(L, -1);
    mem_buf_write_uint32(buf, layerLen);

    if (layerLen == 0) {
        luaL_error(L, "no layers");
    }

    for (i = 1; i <= layerLen; ++i) {
        lua_rawgeti(L, -1, i);
        curLayer = luaL_checkudata(L, -1, "cirno.layer_instance");
        cirnoPersistGradFuncs[curLayer->type](buf, curLayer);
        lua_pop(L, 1);
    }

    lua_pop(L, 1);

    mem_buf_toluastring(L, buf);
    mem_buf_free(buf);

    return 1;
}

static int cirno_network_unpersist_grad(lua_State *L)
{
    const char *data = NULL;
    uint32_t i = 0;
    uint32_t layerLen = 0;
    uint32_t bufn = 0;
    size_t dataLen = 0;
    CirnoTrainerOption *option = NULL;
    CirnoLayer *curLayer = NULL;
    MemBuf *buf = NULL;

    if (!cirno_is_network(L, -2)) {
        luaL_error(L, "argument is not network");
    }

    data = luaL_checklstring(L, -1, &dataLen);
    buf = mem_buf_read_new(data, dataLen);

    lua_getfield(L, -2, "trainer");

    if (lua_isnil(L, -1)) {
        luaL_error(L, "network does not have trainer");
    }

    option = luaL_checkudata(L, -1, "cirno.trainer_option");

    if (option->manual != TRUE) {
        luaL_error(L, "unpersist_grad is only available on manual_apply");
    }

    option->k += mem_buf_read_uint32(buf);

    lua_pop(L, 1);

    lua_getfield(L, -2, "layers");
    layerLen = lua_objlen(L, -1);
    bufn = mem_buf_read_uint32(buf);

    if (layerLen == 0) {
        luaL_error(L, "no layers");
    }

    if (layerLen != bufn) {
        luaL_error(L, "layer length mismatch");
    }

    for (i = 1; i <= layerLen; ++i) {
        lua_rawgeti(L, -1, i);
        curLayer = luaL_checkudata(L, -1, "cirno.layer_instance");
        if (cirnoUnpersistGradFuncs[curLayer->type](buf, curLayer) == FALSE) {
            luaL_error(L, "unpersist grad in layer failed");
        }
        lua_pop(L, 1);
    }

    lua_pop(L, 1);

    mem_buf_free(buf);

    return 0;
}

static int cirno_network_apply_grad(lua_State *L)
{
    CirnoTrainerOption *option = NULL;
    double loss = 0.0;

    if (!cirno_is_network(L, -1)) {
        luaL_error(L, "argument is not network");
    }

    lua_getfield(L, -1, "trainer");

    if (lua_isnil(L, -1)) {
        luaL_error(L, "network does not have trainer");
    }

    option = luaL_checkudata(L, -1, "cirno.trainer_option");

    if (option->manual != TRUE) {
        luaL_error(L, "apply_grad is only available on manual_apply");
    }

    loss = cirno_network_apply_grad_inner(L, option, -2);

    lua_pop(L, 1);

    lua_pushnumber(L, loss);

    return 1;
}

/* ------------------------------------------------------------------------ */
/* INIT */
/* ------------------------------------------------------------------------ */

static luaL_reg cirnoFuncs[] = {
    { "layer", cirno_layer_option_new },
    { "trainer", cirno_trainer_option_new },
    { "network", cirno_network_new },
    { "unpersist", cirno_network_unpersist },
    { NULL, NULL },
};

int luaopen_cirno(lua_State *L)
{
    if (g_mtIndex > kMtConstN) {
        mt19937_seed(time(NULL));
    }

    luaL_newmetatable(L, "cirno.layer_option");
    lua_pop(L, 1);

    luaL_newmetatable(L, "cirno.layer_instance");
    lua_pushcfunction(L, cirno_layer_gc);
    lua_setfield(L, -2, "__gc");
    lua_pop(L, 1);

    luaL_newmetatable(L, "cirno.trainer_option");
    lua_pop(L, 1);

    luaL_newmetatable(L, "cirno.network_instance");
    lua_pushvalue(L, -1);
    lua_setfield(L, -2, "__index");
    lua_pushcfunction(L, cirno_network_predict);
    lua_setfield(L, -2, "predict");
    lua_pushcfunction(L, cirno_network_train);
    lua_setfield(L, -2, "train");
    lua_pushcfunction(L, cirno_network_clear_rin);
    lua_setfield(L, -2, "clear_rin");
    lua_pushcfunction(L, cirno_network_persist);
    lua_setfield(L, -2, "persist");
    lua_pushcfunction(L, cirno_network_persist_grad);
    lua_setfield(L, -2, "persist_grad");
    lua_pushcfunction(L, cirno_network_unpersist_grad);
    lua_setfield(L, -2, "unpersist_grad");
    lua_pushcfunction(L, cirno_network_apply_grad);
    lua_setfield(L, -2, "apply_grad");
    lua_pop(L, 1);

    cirnoFreeFuncs[kCirnoLayerConv]                = cirno_conv_free;
    cirnoFreeFuncs[kCirnoLayerDropout]             = cirno_dropout_free;
    cirnoFreeFuncs[kCirnoLayerFullyConn]           = cirno_fullyconn_free;
    cirnoFreeFuncs[kCirnoLayerInput]               = cirno_input_free;
    cirnoFreeFuncs[kCirnoLayerPad]                 = cirno_pad_free;
    cirnoFreeFuncs[kCirnoLayerPool]                = cirno_pool_free;
    cirnoFreeFuncs[kCirnoLayerRegression]          = cirno_regression_free;
    cirnoFreeFuncs[kCirnoLayerRelu]                = cirno_relu_free;
    cirnoFreeFuncs[kCirnoLayerSigmoid]             = cirno_sigmoid_free;
    cirnoFreeFuncs[kCirnoLayerSoftmax]             = cirno_softmax_free;
    cirnoFreeFuncs[kCirnoLayerStride]              = cirno_stride_free;
    cirnoFreeFuncs[kCirnoLayerSvm]                 = cirno_svm_free;
    cirnoFreeFuncs[kCirnoLayerTanh]                = cirno_tanh_free;
    cirnoFreeFuncs[kCirnoLayerLstm]                = cirno_lstm_free;
    cirnoFreeFuncs[kCirnoLayerMdpAgent]            = cirno_mdpagent_free;
    cirnoFreeFuncs[kCirnoLayerRecurrent]           = cirno_recurrent_free;
    cirnoFreeFuncs[kCirnoLayerNoise]               = cirno_noise_free;

    cirnoForwardFuncs[kCirnoLayerConv]             = cirno_conv_forward;
    cirnoForwardFuncs[kCirnoLayerDropout]          = cirno_dropout_forward;
    cirnoForwardFuncs[kCirnoLayerFullyConn]        = cirno_fullyconn_forward;
    cirnoForwardFuncs[kCirnoLayerInput]            = cirno_input_forward;
    cirnoForwardFuncs[kCirnoLayerPad]              = cirno_pad_forward;
    cirnoForwardFuncs[kCirnoLayerPool]             = cirno_pool_forward;
    cirnoForwardFuncs[kCirnoLayerRegression]       = cirno_regression_forward;
    cirnoForwardFuncs[kCirnoLayerRelu]             = cirno_relu_forward;
    cirnoForwardFuncs[kCirnoLayerSigmoid]          = cirno_sigmoid_forward;
    cirnoForwardFuncs[kCirnoLayerSoftmax]          = cirno_softmax_forward;
    cirnoForwardFuncs[kCirnoLayerStride]           = cirno_stride_forward;
    cirnoForwardFuncs[kCirnoLayerSvm]              = cirno_svm_forward;
    cirnoForwardFuncs[kCirnoLayerTanh]             = cirno_tanh_forward;
    cirnoForwardFuncs[kCirnoLayerLstm]             = cirno_lstm_forward;
    cirnoForwardFuncs[kCirnoLayerMdpAgent]         = cirno_mdpagent_forward;
    cirnoForwardFuncs[kCirnoLayerRecurrent]        = cirno_recurrent_forward;
    cirnoForwardFuncs[kCirnoLayerNoise]            = cirno_noise_forward;

    cirnoBackwardFuncs[kCirnoLayerConv]            = cirno_conv_backward;
    cirnoBackwardFuncs[kCirnoLayerDropout]         = cirno_dropout_backward;
    cirnoBackwardFuncs[kCirnoLayerFullyConn]       = cirno_fullyconn_backward;
    cirnoBackwardFuncs[kCirnoLayerInput]           = cirno_input_backward;
    cirnoBackwardFuncs[kCirnoLayerPad]             = cirno_pad_backward;
    cirnoBackwardFuncs[kCirnoLayerPool]            = cirno_pool_backward;
    cirnoBackwardFuncs[kCirnoLayerRegression]      = cirno_regression_backward;
    cirnoBackwardFuncs[kCirnoLayerRelu]            = cirno_relu_backward;
    cirnoBackwardFuncs[kCirnoLayerSigmoid]         = cirno_sigmoid_backward;
    cirnoBackwardFuncs[kCirnoLayerSoftmax]         = cirno_softmax_backward;
    cirnoBackwardFuncs[kCirnoLayerStride]          = cirno_stride_backward;
    cirnoBackwardFuncs[kCirnoLayerSvm]             = cirno_svm_backward;
    cirnoBackwardFuncs[kCirnoLayerTanh]            = cirno_tanh_backward;
    cirnoBackwardFuncs[kCirnoLayerLstm]            = cirno_lstm_backward;
    cirnoBackwardFuncs[kCirnoLayerMdpAgent]        = cirno_mdpagent_backward;
    cirnoBackwardFuncs[kCirnoLayerRecurrent]       = cirno_recurrent_backward;
    cirnoBackwardFuncs[kCirnoLayerNoise]           = cirno_noise_backward;

    cirnoClearRinFuncs[kCirnoLayerConv]            = cirno_conv_clear_rin;
    cirnoClearRinFuncs[kCirnoLayerDropout]         = cirno_dropout_clear_rin;
    cirnoClearRinFuncs[kCirnoLayerFullyConn]       = cirno_fullyconn_clear_rin;
    cirnoClearRinFuncs[kCirnoLayerInput]           = cirno_input_clear_rin;
    cirnoClearRinFuncs[kCirnoLayerPad]             = cirno_pad_clear_rin;
    cirnoClearRinFuncs[kCirnoLayerPool]            = cirno_pool_clear_rin;
    cirnoClearRinFuncs[kCirnoLayerRegression]      = cirno_regression_clear_rin;
    cirnoClearRinFuncs[kCirnoLayerRelu]            = cirno_relu_clear_rin;
    cirnoClearRinFuncs[kCirnoLayerSigmoid]         = cirno_sigmoid_clear_rin;
    cirnoClearRinFuncs[kCirnoLayerSoftmax]         = cirno_softmax_clear_rin;
    cirnoClearRinFuncs[kCirnoLayerStride]          = cirno_stride_clear_rin;
    cirnoClearRinFuncs[kCirnoLayerSvm]             = cirno_svm_clear_rin;
    cirnoClearRinFuncs[kCirnoLayerTanh]            = cirno_tanh_clear_rin;
    cirnoClearRinFuncs[kCirnoLayerLstm]            = cirno_lstm_clear_rin;
    cirnoClearRinFuncs[kCirnoLayerMdpAgent]        = cirno_mdpagent_clear_rin;
    cirnoClearRinFuncs[kCirnoLayerRecurrent]       = cirno_recurrent_clear_rin;
    cirnoClearRinFuncs[kCirnoLayerNoise]           = cirno_noise_clear_rin;

    cirnoApplyGradFuncs[kCirnoLayerConv]           = cirno_conv_apply_grad;
    cirnoApplyGradFuncs[kCirnoLayerDropout]        = cirno_dropout_apply_grad;
    cirnoApplyGradFuncs[kCirnoLayerFullyConn]      = cirno_fullyconn_apply_grad;
    cirnoApplyGradFuncs[kCirnoLayerInput]          = cirno_input_apply_grad;
    cirnoApplyGradFuncs[kCirnoLayerPad]            = cirno_pad_apply_grad;
    cirnoApplyGradFuncs[kCirnoLayerPool]           = cirno_pool_apply_grad;
    cirnoApplyGradFuncs[kCirnoLayerRegression]     = cirno_regression_apply_grad;
    cirnoApplyGradFuncs[kCirnoLayerRelu]           = cirno_relu_apply_grad;
    cirnoApplyGradFuncs[kCirnoLayerSigmoid]        = cirno_sigmoid_apply_grad;
    cirnoApplyGradFuncs[kCirnoLayerSoftmax]        = cirno_softmax_apply_grad;
    cirnoApplyGradFuncs[kCirnoLayerStride]         = cirno_stride_apply_grad;
    cirnoApplyGradFuncs[kCirnoLayerSvm]            = cirno_svm_apply_grad;
    cirnoApplyGradFuncs[kCirnoLayerTanh]           = cirno_tanh_apply_grad;
    cirnoApplyGradFuncs[kCirnoLayerLstm]           = cirno_lstm_apply_grad;
    cirnoApplyGradFuncs[kCirnoLayerMdpAgent]       = cirno_mdpagent_apply_grad;
    cirnoApplyGradFuncs[kCirnoLayerRecurrent]      = cirno_recurrent_apply_grad;
    cirnoApplyGradFuncs[kCirnoLayerNoise]          = cirno_noise_apply_grad;

    cirnoPersistFuncs[kCirnoLayerConv]             = cirno_conv_persist;
    cirnoPersistFuncs[kCirnoLayerDropout]          = cirno_dropout_persist;
    cirnoPersistFuncs[kCirnoLayerFullyConn]        = cirno_fullyconn_persist;
    cirnoPersistFuncs[kCirnoLayerInput]            = cirno_input_persist;
    cirnoPersistFuncs[kCirnoLayerPad]              = cirno_pad_persist;
    cirnoPersistFuncs[kCirnoLayerPool]             = cirno_pool_persist;
    cirnoPersistFuncs[kCirnoLayerRegression]       = cirno_regression_persist;
    cirnoPersistFuncs[kCirnoLayerRelu]             = cirno_relu_persist;
    cirnoPersistFuncs[kCirnoLayerSigmoid]          = cirno_sigmoid_persist;
    cirnoPersistFuncs[kCirnoLayerSoftmax]          = cirno_softmax_persist;
    cirnoPersistFuncs[kCirnoLayerStride]           = cirno_stride_persist;
    cirnoPersistFuncs[kCirnoLayerSvm]              = cirno_svm_persist;
    cirnoPersistFuncs[kCirnoLayerTanh]             = cirno_tanh_persist;
    cirnoPersistFuncs[kCirnoLayerLstm]             = cirno_lstm_persist;
    cirnoPersistFuncs[kCirnoLayerMdpAgent]         = cirno_mdpagent_persist;
    cirnoPersistFuncs[kCirnoLayerRecurrent]        = cirno_recurrent_persist;
    cirnoPersistFuncs[kCirnoLayerNoise]            = cirno_noise_persist;

    cirnoUnpersistFuncs[kCirnoLayerConv]           = cirno_conv_unpersist;
    cirnoUnpersistFuncs[kCirnoLayerDropout]        = cirno_dropout_unpersist;
    cirnoUnpersistFuncs[kCirnoLayerFullyConn]      = cirno_fullyconn_unpersist;
    cirnoUnpersistFuncs[kCirnoLayerInput]          = cirno_input_unpersist;
    cirnoUnpersistFuncs[kCirnoLayerPad]            = cirno_pad_unpersist;
    cirnoUnpersistFuncs[kCirnoLayerPool]           = cirno_pool_unpersist;
    cirnoUnpersistFuncs[kCirnoLayerRegression]     = cirno_regression_unpersist;
    cirnoUnpersistFuncs[kCirnoLayerRelu]           = cirno_relu_unpersist;
    cirnoUnpersistFuncs[kCirnoLayerSigmoid]        = cirno_sigmoid_unpersist;
    cirnoUnpersistFuncs[kCirnoLayerSoftmax]        = cirno_softmax_unpersist;
    cirnoUnpersistFuncs[kCirnoLayerStride]         = cirno_stride_unpersist;
    cirnoUnpersistFuncs[kCirnoLayerSvm]            = cirno_svm_unpersist;
    cirnoUnpersistFuncs[kCirnoLayerTanh]           = cirno_tanh_unpersist;
    cirnoUnpersistFuncs[kCirnoLayerLstm]           = cirno_lstm_unpersist;
    cirnoUnpersistFuncs[kCirnoLayerMdpAgent]       = cirno_mdpagent_unpersist;
    cirnoUnpersistFuncs[kCirnoLayerRecurrent]      = cirno_recurrent_unpersist;
    cirnoUnpersistFuncs[kCirnoLayerNoise]          = cirno_noise_unpersist;

    cirnoPersistGradFuncs[kCirnoLayerConv]         = cirno_conv_persist_grad;
    cirnoPersistGradFuncs[kCirnoLayerDropout]      = cirno_dropout_persist_grad;
    cirnoPersistGradFuncs[kCirnoLayerFullyConn]    = cirno_fullyconn_persist_grad;
    cirnoPersistGradFuncs[kCirnoLayerInput]        = cirno_input_persist_grad;
    cirnoPersistGradFuncs[kCirnoLayerPad]          = cirno_pad_persist_grad;
    cirnoPersistGradFuncs[kCirnoLayerPool]         = cirno_pool_persist_grad;
    cirnoPersistGradFuncs[kCirnoLayerRegression]   = cirno_regression_persist_grad;
    cirnoPersistGradFuncs[kCirnoLayerRelu]         = cirno_relu_persist_grad;
    cirnoPersistGradFuncs[kCirnoLayerSigmoid]      = cirno_sigmoid_persist_grad;
    cirnoPersistGradFuncs[kCirnoLayerSoftmax]      = cirno_softmax_persist_grad;
    cirnoPersistGradFuncs[kCirnoLayerStride]       = cirno_stride_persist_grad;
    cirnoPersistGradFuncs[kCirnoLayerSvm]          = cirno_svm_persist_grad;
    cirnoPersistGradFuncs[kCirnoLayerTanh]         = cirno_tanh_persist_grad;
    cirnoPersistGradFuncs[kCirnoLayerLstm]         = cirno_lstm_persist_grad;
    cirnoPersistGradFuncs[kCirnoLayerMdpAgent]     = cirno_mdpagent_persist_grad;
    cirnoPersistGradFuncs[kCirnoLayerRecurrent]    = cirno_recurrent_persist_grad;
    cirnoPersistGradFuncs[kCirnoLayerNoise]        = cirno_noise_persist_grad;

    cirnoUnpersistGradFuncs[kCirnoLayerConv]       = cirno_conv_unpersist_grad;
    cirnoUnpersistGradFuncs[kCirnoLayerDropout]    = cirno_dropout_unpersist_grad;
    cirnoUnpersistGradFuncs[kCirnoLayerFullyConn]  = cirno_fullyconn_unpersist_grad;
    cirnoUnpersistGradFuncs[kCirnoLayerInput]      = cirno_input_unpersist_grad;
    cirnoUnpersistGradFuncs[kCirnoLayerPad]        = cirno_pad_unpersist_grad;
    cirnoUnpersistGradFuncs[kCirnoLayerPool]       = cirno_pool_unpersist_grad;
    cirnoUnpersistGradFuncs[kCirnoLayerRegression] = cirno_regression_unpersist_grad;
    cirnoUnpersistGradFuncs[kCirnoLayerRelu]       = cirno_relu_unpersist_grad;
    cirnoUnpersistGradFuncs[kCirnoLayerSigmoid]    = cirno_sigmoid_unpersist_grad;
    cirnoUnpersistGradFuncs[kCirnoLayerSoftmax]    = cirno_softmax_unpersist_grad;
    cirnoUnpersistGradFuncs[kCirnoLayerStride]     = cirno_stride_unpersist_grad;
    cirnoUnpersistGradFuncs[kCirnoLayerSvm]        = cirno_svm_unpersist_grad;
    cirnoUnpersistGradFuncs[kCirnoLayerTanh]       = cirno_tanh_unpersist_grad;
    cirnoUnpersistGradFuncs[kCirnoLayerLstm]       = cirno_lstm_unpersist_grad;
    cirnoUnpersistGradFuncs[kCirnoLayerMdpAgent]   = cirno_mdpagent_unpersist_grad;
    cirnoUnpersistGradFuncs[kCirnoLayerRecurrent]  = cirno_recurrent_unpersist_grad;
    cirnoUnpersistGradFuncs[kCirnoLayerNoise]      = cirno_noise_unpersist_grad;

    luaL_register(L, "cirno", cirnoFuncs);

    return 1;
}

