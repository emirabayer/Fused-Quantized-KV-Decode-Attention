#include <torch/extension.h>
#include <c10/cuda/CUDAException.h>
#include <cuda_fp16.h>
#include <cuda_runtime.h>
// reuse the proven partial kernel from the prealloc module by recompiling it here too
template <int D, int W>
__global__ void gqa_sk_partial_pa2(
    const __half* __restrict__ q, const __half* __restrict__ K, const __half* __restrict__ V,
    float* __restrict__ m_part, float* __restrict__ l_part, float* __restrict__ o_part,
    int N, int S, int G, float scale){
    const int h=blockIdx.x, split=blockIdx.y, kvh=h/G;
    const int lane=threadIdx.x&31, warp=threadIdx.x>>5, ELEMS=D/32;
    const int chunk=(N+S-1)/S, lo=split*chunk, hi=min(lo+chunk,N);
    const __half* qh=q+(size_t)h*D; const __half* Kh=K+(size_t)kvh*N*D; const __half* Vh=V+(size_t)kvh*N*D;
    float q_reg[ELEMS];
    #pragma unroll
    for(int e=0;e<ELEMS;++e) q_reg[e]=__half2float(qh[lane+e*32]);
    float m_run=-INFINITY,l_run=0.0f,o_run[ELEMS];
    #pragma unroll
    for(int e=0;e<ELEMS;++e) o_run[e]=0.0f;
    for(int i=lo+warp;i<hi;i+=W){
        const __half* Ki=Kh+(size_t)i*D; float partial=0.0f;
        #pragma unroll
        for(int e=0;e<ELEMS;++e) partial+=q_reg[e]*__half2float(Ki[lane+e*32]);
        #pragma unroll
        for(int off=16;off>0;off>>=1) partial+=__shfl_down_sync(0xffffffff,partial,off);
        float s=__shfl_sync(0xffffffff,partial,0)*scale;
        float m_new=fmaxf(m_run,s),beta=__expf(m_run-m_new),p=__expf(s-m_new);
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
        for(int w=0;w<W;++w){ float mw=sm[w],mn=fmaxf(m_c,mw);
            float a=(mn==-INFINITY)?0.0f:__expf(m_c-mn);
            float b=(mn==-INFINITY)?0.0f:__expf(mw-mn); l_c=a*l_c+b*sl[w];
            #pragma unroll
            for(int e=0;e<ELEMS;++e) o_c[e]=a*o_c[e]+b*so[w][lane+e*32]; m_c=mn; }
        const size_t base=((size_t)h*S+split);
        if(lane==0){m_part[base]=m_c;l_part[base]=l_c;}
        #pragma unroll
        for(int e=0;e<ELEMS;++e) o_part[base*D+lane+e*32]=o_c[e];
    }
}
// PARALLEL combine: grid=Hq, block=CW warps. Each warp reduces a strided subset of the S
// partials into a local (m,l,o); then warp 0 merges the CW locals. Lane owns ELEMS of o.
template <int D, int CW>
__global__ void gqa_sk_combine_par(
    const float* __restrict__ m_part,const float* __restrict__ l_part,
    const float* __restrict__ o_part,__half* __restrict__ o,int S){
    const int h=blockIdx.x, lane=threadIdx.x&31, warp=threadIdx.x>>5, ELEMS=D/32;
    float m_c=-INFINITY,l_c=0.0f,o_c[ELEMS];
    #pragma unroll
    for(int e=0;e<ELEMS;++e) o_c[e]=0.0f;
    // each warp handles splits warp, warp+CW, warp+2CW, ...
    for(int s=warp;s<S;s+=CW){
        const size_t base=((size_t)h*S+s);
        float m_s=m_part[base],l_s=l_part[base],mn=fmaxf(m_c,m_s);
        float a=(mn==-INFINITY)?0.0f:__expf(m_c-mn);
        float b=(mn==-INFINITY)?0.0f:__expf(m_s-mn); l_c=a*l_c+b*l_s;
        #pragma unroll
        for(int e=0;e<ELEMS;++e) o_c[e]=a*o_c[e]+b*o_part[base*D+lane+e*32]; m_c=mn;
    }
    // merge CW warp-locals via shared mem
    __shared__ float sm[CW],sl[CW],so[CW][D];
    if(lane==0){sm[warp]=m_c;sl[warp]=l_c;}
    #pragma unroll
    for(int e=0;e<ELEMS;++e) so[warp][lane+e*32]=o_c[e];
    __syncthreads();
    if(warp==0){
        float mm=-INFINITY,ll=0.0f,oo[ELEMS];
        #pragma unroll
        for(int e=0;e<ELEMS;++e) oo[e]=0.0f;
        #pragma unroll
        for(int w=0;w<CW;++w){ float mw=sm[w],mn=fmaxf(mm,mw);
            float a=(mn==-INFINITY)?0.0f:__expf(mm-mn);
            float b=(mn==-INFINITY)?0.0f:__expf(mw-mn); ll=a*ll+b*sl[w];
            #pragma unroll
            for(int e=0;e<ELEMS;++e) oo[e]=a*oo[e]+b*so[w][lane+e*32]; mm=mn; }
        __half* oh=o+(size_t)h*D;
        #pragma unroll
        for(int e=0;e<ELEMS;++e) oh[lane+e*32]=__float2half(oo[e]/ll);
    }
}
void gqa_sk_fast(torch::Tensor q,torch::Tensor K,torch::Tensor V,
    torch::Tensor m_part,torch::Tensor l_part,torch::Tensor o_part,torch::Tensor o,
    int64_t S,int64_t W,int64_t CW){
    const int Hq=q.size(0),D=q.size(1),Hkv=K.size(0),N=K.size(1),G=Hq/Hkv;
    const float scale=1.0f/sqrtf((float)D);
    dim3 pg(Hq,S),pb(32*W);
    #define LP(Dv,Wv) gqa_sk_partial_pa2<Dv,Wv><<<pg,pb>>>((const __half*)q.data_ptr<at::Half>(),(const __half*)K.data_ptr<at::Half>(),(const __half*)V.data_ptr<at::Half>(),m_part.data_ptr<float>(),l_part.data_ptr<float>(),o_part.data_ptr<float>(),N,(int)S,G,scale)
    if(D==64){switch(W){case 2:LP(64,2);break;case 4:LP(64,4);break;case 8:LP(64,8);break;default:TORCH_CHECK(false,"W");}}
    else{switch(W){case 2:LP(128,2);break;case 4:LP(128,4);break;case 8:LP(128,8);break;default:TORCH_CHECK(false,"W");}}
    #undef LP
    TORCH_CHECK(cudaGetLastError()==cudaSuccess,"partial");
    dim3 cg(Hq),cb(32*CW);
    #define LC(Dv,CWv) gqa_sk_combine_par<Dv,CWv><<<cg,cb>>>(m_part.data_ptr<float>(),l_part.data_ptr<float>(),o_part.data_ptr<float>(),(__half*)o.data_ptr<at::Half>(),(int)S)
    if(D==64){switch(CW){case 4:LC(64,4);break;case 8:LC(64,8);break;default:TORCH_CHECK(false,"CW");}}
    else{switch(CW){case 4:LC(128,4);break;case 8:LC(128,8);break;default:TORCH_CHECK(false,"CW");}}
    #undef LC
    TORCH_CHECK(cudaGetLastError()==cudaSuccess,"combine");
}
PYBIND11_MODULE(TORCH_EXTENSION_NAME,m){m.def("gqa_sk_fast",&gqa_sk_fast,"GQA split-K, parallel combine");}
