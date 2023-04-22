#include <torch/extension.h>

void cuda_forward(
    int B, int T, int C, 
    float *w, float *u, float *k, float *v, 
    float *h1, float *h2, float *y, 
    float *oh1, float *oh2
); 

void cuda_backward(
    int B, int T, int C, 
    float *w, float *u, float *k, float *v,
    float *h1, float *h2, float *y, 
    float *gy, float *goh1, float *goh2, 
    float *gw, float *gu, float *gk, float *gv, 
    float *gh1, float *gh2
); 

void forward(
    int64_t B, int64_t T, int64_t C, 
    torch::Tensor &w, 
    torch::Tensor &u, 
    torch::Tensor &k, 
    torch::Tensor &v, 
    torch::Tensor &h1, 
    torch::Tensor &h2, 
    torch::Tensor &y, 
    torch::Tensor &oh1, 
    torch::Tensor &oh2
) {
    cuda_forward(
        B, T, C, 
        w.data_ptr<float>(), 
        u.data_ptr<float>(), 
        k.data_ptr<float>(), 
        v.data_ptr<float>(), 
        h1.data_ptr<float>(), 
        h2.data_ptr<float>(),
        y.data_ptr<float>(), 
        oh1.data_ptr<float>(), 
        oh2.data_ptr<float>()
    );
}

void backward(
    int64_t B, int64_t T, int64_t C, 
    torch::Tensor &w, 
    torch::Tensor &u, 
    torch::Tensor &k, 
    torch::Tensor &v, 
    torch::Tensor &h1, 
    torch::Tensor &h2, 
    torch::Tensor &y, 
    torch::Tensor &gy, 
    torch::Tensor &goh1, 
    torch::Tensor &goh2, 
    torch::Tensor &gw, 
    torch::Tensor &gu, 
    torch::Tensor &gk, 
    torch::Tensor &gv, 
    torch::Tensor &gh1, 
    torch::Tensor &gh2
) {
    cuda_backward(
        B, T, C, 
        w.data_ptr<float>(), 
        u.data_ptr<float>(), 
        k.data_ptr<float>(), 
        v.data_ptr<float>(), 
        h1.data_ptr<float>(), 
        h2.data_ptr<float>(),
        y.data_ptr<float>(), 
        gy.data_ptr<float>(), 
        goh1.data_ptr<float>(), 
        goh2.data_ptr<float>(), 
        gw.data_ptr<float>(), 
        gu.data_ptr<float>(), 
        gk.data_ptr<float>(), 
        gv.data_ptr<float>(), 
        gh1.data_ptr<float>(), 
        gh2.data_ptr<float>()
    );
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("forward", &forward, "wkv forward");
    m.def("backward", &backward, "wkv backward");
}

TORCH_LIBRARY(wkv_extend, m) {
    m.def("forward", forward);
    m.def("backward", backward);
}