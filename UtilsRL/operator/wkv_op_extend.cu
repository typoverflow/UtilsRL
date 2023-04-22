#include <stdio.h>
#include <assert.h>

#define MIN_VALUE (-1e38)

template <typename F>
__global__ void kernel_forward(
    const int B, 
    const int T, 
    const int C,
    const F *__restrict__ const _w, 
    const F *__restrict__ const _u, 
    const F *__restrict__ const _k, 
    const F *__restrict__ const _v,
    const F *__restrict__ const _h1, 
    const F *__restrict__ const _h2,
    F *__restrict__ const _y, 
    F *__restrict__ const _oh1, 
    F *__restrict__ const _oh2, 
){
    const int idx = blockIdx.x * blockDim.x + threadIdx.x;
    const int _b = idx / C;
    const int _c = idx % C;
    const int _offset = _b * T * C + _c;
    const int _hist_offset = _b * C + _c; 

    F u = _u[_c];
    F w = _w[_c];
    F aa, bb, pp;
    aa = (_h1 == NULL)? 0:_h1[_hist_offset]; 
    bb = (_h2 == NULL)? 0:_h2[_hist_offset*2]; 
    pp = (_h2 == NULL)? MIN_VALUE:_h2[_hist_offset*2+1]; 

    const F *__restrict__ const k = _k + _offset;
    const F *__restrict__ const v = _v + _offset;
    F *__restrict__ const y = _y + _offset;

    // aa and bb are running sums divided by exp(pp) (to avoid overflow)
    for (int i = 0; i < T; i++) {
        const int ii = i * C;
        const F kk = k[ii];
        const F vv = v[ii];

        F ww = u + kk;
        F p = max(pp, ww);
        F e1 = exp(pp - p);
        F e2 = exp(ww - p);
        y[ii] = (e1 * aa + e2 * vv) / (e1 * bb + e2);
        
        ww = w + pp;
        p = max(ww, kk);
        e1 = exp(ww - p);
        e2 = exp(kk - p);
        aa = e1 * aa + e2 * vv;
        bb = e1 * bb + e2;
        pp = p;
    }

    _oh1[_hist_offset] = aa; 
    _oh2[_hist_offset*2] = bb; 
    _oh3[_hist_offset*2+1] = pp; 
}

void cuda_forward(
    int B, int T, int C, 
    float *w, float *u, float *k, float *v, 
    float *h1, float *h2, float *y, 
    float *oh1, float *oh2
) {
    dim3 threadsPerBlock( min(C, 32) ); // requires --maxrregcount 60 for optimal performance
    assert(B * C % threadsPerBlock.x == 0);
    dim3 numBlocks(B * C / threadsPerBlock.x);
    kernel_forward<<<numBlocks, threadsPerBlock>>>(B, T, C, w, u, k, v, h1, h2, y, oh1, oh2);
}


template <typename F>
__global__ void kernel_backward(
    const int B, 
    const int T, 
    const int C,
    const F *__restrict__ const _w, 
    const F *__restrict__ const _u, 
    const F *__restrict__ const _k, 
    const F *__restrict__ const _v,
    const F *__restrict__ const _h1, 
    const F *__restrict__ const _h2,
    const F *__restrict__ const _y, 
    const F *__restrict__ const _gy, 
    const F *__restrict__ const _goh1, 
    const F *__restrict__ const _goh2, 
    F *__restrict__ const _gw, 
    F *__restrict__ const _gu, 
    F *__restrict__ const _gk, 
    F *__restrict__ const _gv, 
    F *__restrict__ const _gh1, 
    F *__restrict__ const _gh2
) {
    const int idx = blockIdx.x * blockDim.x + threadIdx.x;
    const int _b = idx / C;
    const int _c = idx % C;
    const int _offset = _b * T * C + _c;
    const int _hist_offset = _b * C + _c; 

    F u = _u[_c];
    F w = _w[_c];
    const F *__restrict__ const k = _k + _offset;
    const F *__restrict__ const v = _v + _offset;
    const F *__restrict__ const y = _y + _offset;
    const F *__restrict__ const gy = _gy + _offset;
    F *__restrict__ const gk = _gk + _offset;
    F *__restrict__ const gv = _gv + _offset;

    F q[Tmax], r[Tmax];
    F gw = 0, gu = 0, ga = 0, gb = 0; 
    F aa, bb, pp; 
    aa = (_h1 == NULL)? 0:_h1[_hist_offset]; 
    bb = (_h2 == NULL)? 0:_h2[_hist_offset*2]; 
    pp = (_h2 == NULL)? MIN_VALUE:_h2[_hist_offset*2+1]; 
    for (int i = 0; i < T; i++) {
        const int ii = i * C;
        const F kk = k[ii];
        const F vv = v[ii];
        const F yy = y[ii];

        F ww = u + kk;
        F p = max(pp, ww);
        F e1 = exp(pp - p);
        F e2 = exp(ww - p);
        const F qq = gy[ii] / (e1 * bb + e2);
        gw += (ga - gb * yy) * e1 * qq;
        gu += (vv - yy) * e2 * qq;
        q[i] = qq;
        r[i] = ww - p;

        ww = w + pp;
        p = max(ww, kk);
        e1 = exp(ww - p);
        e2 = exp(kk - p);
        ga = e1 * (aa + ga);
        gb = e1 * (bb + gb);
        aa = e1 * aa + e2 * vv;
        bb = e1 * bb + e2;
        pp = p;
    }

    F gaa = 0, gbb = 0, gpp = MIN_VALUE; 
    if (_goh1 != NULL && _goh2 != NULL) {
        gaa = _goh1[_hist_offset]; 
        gbb = _goh2[_hist_offset*2]; 
        gpp = _goh2[_hist_offset*2+1];
        if (gaa == 0 && gbb == 0) gpp = MIN_VALUE;
    }

    // below is back-propagating gradients through time
    gw += (gaa * ga + gbb * gb);
    for (int i = T - 1; i >= 0; i--) {
        const int ii = i * C;
        const F kk = k[ii];
        const F vv = v[ii];
        const F yy = y[ii];
        const F qq = q[i];
        const F rr = r[i];

        F e1 = qq * exp(rr);
        F e2 = exp(kk + gpp);
        gk[ii] = e1 * (vv - yy) + e2 * (gaa * vv + gbb);
        gv[ii] = e1 + e2 * gaa;

        const F ww = w + gpp;
        const F www = rr - u - kk;
        const F p = max(ww, www);
        e1 = exp(ww - p);
        e2 = qq * exp(www - p);
        gaa = e1 * gaa + e2;
        gbb = e1 * gbb - e2 * yy;
        gpp = p;
    }

    if (_gh1 != NULL and _gh2 != NULL) {
        _gh1[_hist_offset] = gaa; 
        _gh2[_hist_offset*2] = gbb; 
        _gh2[_hist_offset*2+1] = gpp; 
    }

    const int _offsetBC = _b * C + _c;
    _gw[_offsetBC] = gw * _w[_c]; // multiply by w because of w -> -exp(w) in python forward()
    _gu[_offsetBC] = gu;
}

void cuda_backward(
    int B, int T, int C, 
    float *w, float *u, float *k, float *v,
    float *h1, float *h2, float *y, 
    float *gy, float *goh1, float *goh2, 
    float *gw, float *gu, float *gk, float *gv, 
    float *gh1, float* gh2
) {
    dim3 threadsPerBlock( min(C, 32) ); // requires --maxrregcount 60 for optimal performance
    assert(B * C % threadsPerBlock.x == 0);
    dim3 numBlocks(B * C / threadsPerBlock.x);
    kernel_backward<<<numBlocks, threadsPerBlock>>>(B, T, C, w, u, k, v, h1, h2, y, gy, goh1, goh2, gw, gu, gk, gv, gh1, gh2);
}