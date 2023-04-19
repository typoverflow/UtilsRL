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
    const F *__restrict__ const _h3, 
    F *__restrict__ const _y, 
    F *__restrict__ const _oh1, 
    F *__restrict__ const _oh2, 
    F *__restrict__ const _oh3
){
    const int idx = blockIdx.x * blockDim.x + threadIdx.x;
    const int _b = idx / C;
    const int _c = idx % C;
    const int _offset = _b * T * C + _c;
    const int _hist_offset = _b * C + _c; 

    F u = _u[_c];
    F w = _w[_c];
    F h1 = _h1[_hist_offset]; 
    F h2 = _h2[_hist_offset]; 
    F h3 = _h3[_hist_offset]; 
    const F *__restrict__ const k = _k + _offset;
    const F *__restrict__ const v = _v + _offset;
    F *__restrict__ const y = _y + _offset;

    // aa and bb are running sums divided by exp(pp) (to avoid overflow)
    F aa = h1, bb = h2, pp = h3;
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
    _oh2[_hist_offset] = bb; 
    _oh3[_hist_offset] = pp; 
}

void cuda_forward(
    int B, int T, int C, 
    float *w, float *u, float *k, float *v, 
    float *h1, float *h2, float *h3, float *y, 
    float *oh1, float *oh2, float *oh3
) {
    dim3 threadsPerBlock( min(C, 32) ); // requires --maxrregcount 60 for optimal performance
    assert(B * C % threadsPerBlock.x == 0);
    dim3 numBlocks(B * C / threadsPerBlock.x);
    kernel_forward<<<numBlocks, threadsPerBlock>>>(B, T, C, w, u, k, v, h1, h2, h3, y, oh1, oh2, oh3);
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
    const F *__restrict__ const _h3, 
    const F *__restrict__ const _y, 
    const F *__restrict__ const _gy,
    F *__restrict__ const _gw, 
    F *__restrict__ const _gu, 
    F *__restrict__ const _gk, 
    F *__restrict__ const _gv
) {
    const int idx = blockIdx.x * blockDim.x + threadIdx.x;
    const int _b = idx / C;
    const int _c = idx % C;
    const int _offset = _b * T * C + _c;
    const int _hist_offset = _b * C + _c; 

    F u = _u[_c];
    F w = _w[_c];
    F h1 = _h1[_hist_offset]; 
    F h2 = _h2[_hist_offset]; 
    F h3 = _h3[_hist_offset]; 
    const F *__restrict__ const k = _k + _offset;
    const F *__restrict__ const v = _v + _offset;
    const F *__restrict__ const y = _y + _offset;
    const F *__restrict__ const gy = _gy + _offset;
    F *__restrict__ const gk = _gk + _offset;
    F *__restrict__ const gv = _gv + _offset;

    F q[Tmax], r[Tmax];

    F gw = 0, gu = 0, aa = h1, bb = h2, ga = 0, gb = 0, pp = h3;
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
    const int _offsetBC = _b * C + _c;
    _gw[_offsetBC] = gw * _w[_c]; // multiply by w because of w -> -exp(w) in python forward()
    _gu[_offsetBC] = gu;

    aa = h1, bb = h2, pp = h3;
    for (int i = T - 1; i >= 0; i--) {
        const int ii = i * C;
        const F kk = k[ii];
        const F vv = v[ii];
        const F yy = y[ii];
        const F qq = q[i];
        const F rr = r[i];

        F e1 = qq * exp(rr);
        F e2 = exp(kk + pp);
        gk[ii] = e1 * (vv - yy) + e2 * (aa * vv + bb);
        gv[ii] = e1 + e2 * aa;

        const F ww = w + pp;
        const F www = rr - u - kk;
        const F p = max(ww, www);
        e1 = exp(ww - p);
        e2 = qq * exp(www - p);
        aa = e1 * aa + e2;
        bb = e1 * bb - e2 * yy;
        pp = p;
    }
}

void cuda_backward(
    int B, int T, int C, 
    float *w, float *u, float *k, float *v,
    float *h1, float *h2, float *h3, float *y, 
    float *gy, float *gw, float *gu, float *gk, float *gv
) {
    dim3 threadsPerBlock( min(C, 32) ); // requires --maxrregcount 60 for optimal performance
    assert(B * C % threadsPerBlock.x == 0);
    dim3 numBlocks(B * C / threadsPerBlock.x);
    kernel_backward<<<numBlocks, threadsPerBlock>>>(B, T, C, w, u, k, v, h1, h2, h3, y, gy, gw, gu, gk, gv);
}