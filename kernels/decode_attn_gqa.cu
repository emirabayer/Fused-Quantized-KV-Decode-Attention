
#include <torch/extension.h>
#include <c10/cuda/CUDAException.h>
#include <cuda_fp16.h>
#include <cuda_runtime.h>
// GQA-aware fp16: grid = B*Hq blocks; block maps to (b,h); reads KV head h/G directly.
// q:[B,Hq,D]  K,V:[B,Hkv,N,D]  o:[B,Hq,D]
template <int D, int W>
__global__ void decode_attn_gqa_kernel(
    const __half* __restrict__ q, const __half* __restrict__ K, const __half* __restrict__ V,
    __half* __restrict__ o, int N, int Hq, int Hkv, int G, float scale){
    const int blk=blockIdx.x;            // 0 .. B*Hq-1
    const int b=blk/Hq, h=blk%Hq, kvh=h/G;
    const int lane=threadIdx.x&31, warp=threadIdx.x>>5, ELEMS=D/32;
    const __half* qh=q+((size_t)b*Hq + h)*D;
    const __half* Kh=K+(((size_t)b*Hkv + kvh)*N)*D;
    const __half* Vh=V+(((size_t)b*Hkv + kvh)*N)*D;
    __half* oh=o+((size_t)b*Hq + h)*D;
    float q_reg[ELEMS];
    #pragma unroll
    for(int e=0;e<ELEMS;++e) q_reg[e]=__half2float(qh[lane+e*32]);
    float m_run=-INFINITY,l_run=0.0f,o_run[ELEMS];
    #pragma unroll
    for(int e=0;e<ELEMS;++e) o_run[e]=0.0f;
    for(int i=warp;i<N;i+=W){
        const __half* Ki=Kh+(size_t)i*D; float partial=0.0f;
        #pragma unroll
        for(int e=0;e<ELEMS;++e) partial+=q_reg[e]*__half2float(Ki[lane+e*32]);
        #pragma unroll
        for(int off=16;off>0;off>>=1) partial+=__shfl_down_sync(0xffffffff,partial,off);
        float s=__shfl_sync(0xffffffff,partial,0)*scale;
        float m_new=fmaxf(m_run,s), beta=__expf(m_run-m_new), p=__expf(s-m_new);
        l_run=beta*l_run+p; const __half* Vi=Vh+(size_t)i*D;
        #pragma unroll
        for(int e=0;e<ELEMS;++e) o_run[e]=beta*o_run[e]+p*__half2float(Vi[lane+e*32]);
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
            float a=__expf(m_c-m_new),b2=__expf(m_w-m_new); l_c=a*l_c+b2*sl[w];
            #pragma unroll
            for(int e=0;e<ELEMS;++e) o_c[e]=a*o_c[e]+b2*so[w][lane+e*32]; m_c=m_new; }
        #pragma unroll
        for(int e=0;e<ELEMS;++e) oh[lane+e*32]=__float2half(o_c[e]/l_c);
    }
}
torch::Tensor decode_attn_gqa(torch::Tensor q, torch::Tensor K, torch::Tensor V, int64_t W){
    // q:[B,Hq,D] K,V:[B,Hkv,N,D]
    const int B=q.size(0),Hq=q.size(1),D=q.size(2);
    const int Hkv=K.size(1),N=K.size(2); const int G=Hq/Hkv;
    auto o=torch::empty({B,Hq,D},q.options()); const float scale=1.0f/sqrtf((float)D);
    dim3 grid(B*Hq),block(32*W);
    #define L(Dv,Wv) decode_attn_gqa_kernel<Dv,Wv><<<grid,block>>>((const __half*)q.data_ptr<at::Half>(),(const __half*)K.data_ptr<at::Half>(),(const __half*)V.data_ptr<at::Half>(),(__half*)o.data_ptr<at::Half>(),N,Hq,Hkv,G,scale)
    TORCH_CHECK(D==64||D==128,"D");
    if(D==64){switch(W){case 2:L(64,2);break;case 4:L(64,4);break;case 8:L(64,8);break;default:TORCH_CHECK(false,"W");}}
    else{switch(W){case 2:L(128,2);break;case 4:L(128,4);break;case 8:L(128,8);break;default:TORCH_CHECK(false,"W");}}
    #undef L
    TORCH_CHECK(cudaGetLastError()==cudaSuccess,"launch"); return o;
}
PYBIND11_MODULE(TORCH_EXTENSION_NAME,m){m.def("decode_attn_gqa",&decode_attn_gqa,"GQA-aware fp16");}
