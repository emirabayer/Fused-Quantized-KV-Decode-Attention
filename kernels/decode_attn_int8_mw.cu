
#include <torch/extension.h>
#include <c10/cuda/CUDAException.h>
#include <cuda_fp16.h>
#include <cuda_runtime.h>
template <int D, int W>
__global__ void decode_attn_int8_mw_kernel(
    const __half* __restrict__ q, const int8_t* __restrict__ K, const int8_t* __restrict__ V,
    const __half* __restrict__ Ks, const __half* __restrict__ Vs, __half* __restrict__ o,
    int N, float scale){
    const int head=blockIdx.x, lane=threadIdx.x&31, warp=threadIdx.x>>5, ELEMS=D/32;
    const __half* qh=q+(size_t)head*D; const int8_t* Kh=K+(size_t)head*N*D;
    const int8_t* Vh=V+(size_t)head*N*D; const __half* Ksh=Ks+(size_t)head*N;
    const __half* Vsh=Vs+(size_t)head*N; __half* oh=o+(size_t)head*D;
    float q_reg[ELEMS];
    #pragma unroll
    for(int e=0;e<ELEMS;++e) q_reg[e]=__half2float(qh[lane+e*32]);
    float m_run=-INFINITY,l_run=0.0f,o_run[ELEMS];
    #pragma unroll
    for(int e=0;e<ELEMS;++e) o_run[e]=0.0f;
    for(int i=warp;i<N;i+=W){
        const int8_t* Ki=Kh+(size_t)i*D; float ksc=__half2float(Ksh[i]); float partial=0.0f;
        #pragma unroll
        for(int e=0;e<ELEMS;++e) partial+=q_reg[e]*((float)Ki[lane+e*32]*ksc);
        #pragma unroll
        for(int off=16;off>0;off>>=1) partial+=__shfl_down_sync(0xffffffff,partial,off);
        float s=__shfl_sync(0xffffffff,partial,0)*scale;
        float m_new=fmaxf(m_run,s), beta=__expf(m_run-m_new), p=__expf(s-m_new);
        l_run=beta*l_run+p; const int8_t* Vi=Vh+(size_t)i*D; float vsc=__half2float(Vsh[i]);
        #pragma unroll
        for(int e=0;e<ELEMS;++e) o_run[e]=beta*o_run[e]+p*((float)Vi[lane+e*32]*vsc);
        m_run=m_new;
    }
    __shared__ float sm[W],sl[W],so[W][D];
    if(lane==0){sm[warp]=m_run;sl[warp]=l_run;}
    #pragma unroll
    for(int e=0;e<ELEMS;++e) so[warp][lane+e*32]=o_run[e];
    __syncthreads();
    if(warp==0){
        float m_c=-INFINITY,l_c=0.0f,o_c[ELEMS];
        #pragma unroll
        for(int e=0;e<ELEMS;++e) o_c[e]=0.0f;
        #pragma unroll
        for(int w=0;w<W;++w){ float m_w=sm[w],m_new=fmaxf(m_c,m_w);
            float a=__expf(m_c-m_new),b=__expf(m_w-m_new); l_c=a*l_c+b*sl[w];
            #pragma unroll
            for(int e=0;e<ELEMS;++e) o_c[e]=a*o_c[e]+b*so[w][lane+e*32]; m_c=m_new; }
        #pragma unroll
        for(int e=0;e<ELEMS;++e) oh[lane+e*32]=__float2half(o_c[e]/l_c);
    }
}
torch::Tensor decode_attn_int8_mw(torch::Tensor q, torch::Tensor K, torch::Tensor V,
    torch::Tensor Ks, torch::Tensor Vs, int64_t W){
    const int B=K.size(0),N=K.size(1),D=K.size(2);
    TORCH_CHECK(K.scalar_type()==torch::kChar&&V.scalar_type()==torch::kChar,"int8");
    auto o=torch::empty({B,D},q.options()); const float scale=1.0f/sqrtf((float)D);
    dim3 grid(B),block(32*W);
    #define L(Dv,Wv) decode_attn_int8_mw_kernel<Dv,Wv><<<grid,block>>>((const __half*)q.data_ptr<at::Half>(),(const int8_t*)K.data_ptr<int8_t>(),(const int8_t*)V.data_ptr<int8_t>(),(const __half*)Ks.data_ptr<at::Half>(),(const __half*)Vs.data_ptr<at::Half>(),(__half*)o.data_ptr<at::Half>(),N,scale)
    TORCH_CHECK(D==64||D==128,"D");
    if(D==64){switch(W){case 2:L(64,2);break;case 4:L(64,4);break;case 8:L(64,8);break;default:TORCH_CHECK(false,"W");}}
    else{switch(W){case 2:L(128,2);break;case 4:L(128,4);break;case 8:L(128,8);break;default:TORCH_CHECK(false,"W");}}
    #undef L
    TORCH_CHECK(cudaGetLastError()==cudaSuccess,"launch"); return o;
}
PYBIND11_MODULE(TORCH_EXTENSION_NAME,m){m.def("decode_attn_int8_mw",&decode_attn_int8_mw,"int8 fused");}
