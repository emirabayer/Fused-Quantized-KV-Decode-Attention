
#include <torch/extension.h>
#include <c10/cuda/CUDAException.h>
#include <cuda_fp16.h>
#include <cuda_runtime.h>
// int4 KV (packed 2/byte), per-token symmetric scales, multi-warp. D=64 only.
// lane t owns byte t -> elems (2t lo-nibble, 2t+1 hi-nibble).
template <int W>
__global__ void decode_attn_int4_kernel(
    const __half* __restrict__ q, const uint8_t* __restrict__ K, const uint8_t* __restrict__ V,
    const __half* __restrict__ Ks, const __half* __restrict__ Vs, __half* __restrict__ o,
    int N, float scale){
    const int D=64, H=32; // H = packed bytes per token = D/2
    const int head=blockIdx.x, lane=threadIdx.x&31, warp=threadIdx.x>>5;
    const __half* qh=q+(size_t)head*D; const uint8_t* Kh=K+(size_t)head*N*H;
    const uint8_t* Vh=V+(size_t)head*N*H; const __half* Ksh=Ks+(size_t)head*N;
    const __half* Vsh=Vs+(size_t)head*N; __half* oh=o+(size_t)head*D;
    float q0=__half2float(qh[2*lane+0]), q1=__half2float(qh[2*lane+1]);
    float m_run=-INFINITY,l_run=0.0f,o0=0.0f,o1=0.0f;
    for(int i=warp;i<N;i+=W){
        unsigned int kb=Kh[(size_t)i*H + lane]; float ksc=__half2float(Ksh[i]);
        int k0=kb&0xF; if(k0>=8)k0-=16; int k1=(kb>>4)&0xF; if(k1>=8)k1-=16;
        float partial=q0*((float)k0*ksc)+q1*((float)k1*ksc);
        #pragma unroll
        for(int off=16;off>0;off>>=1) partial+=__shfl_down_sync(0xffffffff,partial,off);
        float s=__shfl_sync(0xffffffff,partial,0)*scale;
        float m_new=fmaxf(m_run,s), beta=__expf(m_run-m_new), p=__expf(s-m_new);
        l_run=beta*l_run+p;
        unsigned int vb=Vh[(size_t)i*H + lane]; float vsc=__half2float(Vsh[i]);
        int v0=vb&0xF; if(v0>=8)v0-=16; int v1=(vb>>4)&0xF; if(v1>=8)v1-=16;
        o0=beta*o0+p*((float)v0*vsc); o1=beta*o1+p*((float)v1*vsc);
        m_run=m_new;
    }
    __shared__ float sm[W],sl[W],so[W][64];
    if(lane==0){sm[warp]=m_run;sl[warp]=l_run;}
    so[warp][2*lane+0]=o0; so[warp][2*lane+1]=o1;
    __syncthreads();
    if(warp==0){
        float m_c=-INFINITY,l_c=0.0f,c0=0.0f,c1=0.0f;
        #pragma unroll
        for(int w=0;w<W;++w){ float m_w=sm[w],m_new=fmaxf(m_c,m_w);
            float a=__expf(m_c-m_new),b=__expf(m_w-m_new); l_c=a*l_c+b*sl[w];
            c0=a*c0+b*so[w][2*lane+0]; c1=a*c1+b*so[w][2*lane+1]; m_c=m_new; }
        oh[2*lane+0]=__float2half(c0/l_c); oh[2*lane+1]=__float2half(c1/l_c);
    }
}
torch::Tensor decode_attn_int4(torch::Tensor q, torch::Tensor K, torch::Tensor V,
    torch::Tensor Ks, torch::Tensor Vs, int64_t W){
    TORCH_CHECK(K.scalar_type()==torch::kByte&&V.scalar_type()==torch::kByte,"packed uint8");
    const int B=K.size(0),N=K.size(1),H=K.size(2); TORCH_CHECK(H==32,"D=64 packed -> 32 bytes");
    auto o=torch::empty({B,64},q.options()); const float scale=1.0f/sqrtf(64.0f);
    dim3 grid(B),block(32*W);
    #define L(Wv) decode_attn_int4_kernel<Wv><<<grid,block>>>((const __half*)q.data_ptr<at::Half>(),(const uint8_t*)K.data_ptr<uint8_t>(),(const uint8_t*)V.data_ptr<uint8_t>(),(const __half*)Ks.data_ptr<at::Half>(),(const __half*)Vs.data_ptr<at::Half>(),(__half*)o.data_ptr<at::Half>(),N,scale)
    switch(W){case 2:L(2);break;case 4:L(4);break;case 8:L(8);break;default:TORCH_CHECK(false,"W");}
    #undef L
    TORCH_CHECK(cudaGetLastError()==cudaSuccess,"launch"); return o;
}
PYBIND11_MODULE(TORCH_EXTENSION_NAME,m){m.def("decode_attn_int4",&decode_attn_int4,"int4 KV");}
