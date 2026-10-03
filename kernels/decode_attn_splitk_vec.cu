#include <torch/extension.h>
#include <c10/cuda/CUDAException.h>
#include <cuda_fp16.h>
#include <cuda_runtime.h>

// Vectorized split-K. Each lane owns VPT contiguous elements, loaded as one vector.
// D=64 -> VPT=2 (32-bit load); D=128 -> VPT=4 (64-bit load). All 32 lanes active.
// Lane t owns elements [t*VPT, t*VPT+VPT).  (contiguous, not strided)
template <int D, int W, int VPT>
__global__ void splitk_partial_vec_kernel(
    const __half* __restrict__ q, const __half* __restrict__ K, const __half* __restrict__ V,
    float* __restrict__ m_part, float* __restrict__ l_part, float* __restrict__ o_part,
    int N, int S, float scale)
{
    const int head=blockIdx.x, split=blockIdx.y;
    const int lane=threadIdx.x&31, warp=threadIdx.x>>5;
    const int chunk=(N+S-1)/S, lo=split*chunk, hi=min(lo+chunk,N);
    const __half* qh=q+(size_t)head*D;
    const __half* Kh=K+(size_t)head*N*D;
    const __half* Vh=V+(size_t)head*N*D;

    // load this lane's VPT contiguous q elements (vectorized) into registers
    float q_reg[VPT];
    const int off0 = lane*VPT;
    #pragma unroll
    for(int j=0;j<VPT;++j) q_reg[j]=__half2float(qh[off0+j]);

    float m_run=-INFINITY, l_run=0.0f, o_run[VPT];
    #pragma unroll
    for(int j=0;j<VPT;++j) o_run[j]=0.0f;

    for(int i=lo+warp; i<hi; i+=W){
        const __half* Ki=Kh+(size_t)i*D;
        // ---- vectorized load of VPT contiguous K elements for this lane ----
        float kk[VPT];
        if(VPT==2){
            float v = *reinterpret_cast<const float*>(Ki+off0);   // 32-bit = 2 halfs
            __half2 h = *reinterpret_cast<__half2*>(&v);
            kk[0]=__half2float(__low2half(h)); kk[1]=__half2float(__high2half(h));
        } else { // VPT==4, 64-bit load
            float2 v = *reinterpret_cast<const float2*>(Ki+off0);
            __half2 h0=*reinterpret_cast<__half2*>(&v.x), h1=*reinterpret_cast<__half2*>(&v.y);
            kk[0]=__half2float(__low2half(h0)); kk[1]=__half2float(__high2half(h0));
            kk[2]=__half2float(__low2half(h1)); kk[3]=__half2float(__high2half(h1));
        }
        float partial=0.0f;
        #pragma unroll
        for(int j=0;j<VPT;++j) partial+=q_reg[j]*kk[j];
        // full warp reduction (all 32 lanes contribute their VPT-element partials)
        #pragma unroll
        for(int o=16;o>0;o>>=1) partial+=__shfl_down_sync(0xffffffff,partial,o);
        float s=__shfl_sync(0xffffffff,partial,0)*scale;

        float m_new=fmaxf(m_run,s), beta=__expf(m_run-m_new), p=__expf(s-m_new);
        l_run=beta*l_run+p;
        const __half* Vi=Vh+(size_t)i*D;
        float vv[VPT];
        if(VPT==2){
            float v=*reinterpret_cast<const float*>(Vi+off0); __half2 h=*reinterpret_cast<__half2*>(&v);
            vv[0]=__half2float(__low2half(h)); vv[1]=__half2float(__high2half(h));
        } else {
            float2 v=*reinterpret_cast<const float2*>(Vi+off0);
            __half2 h0=*reinterpret_cast<__half2*>(&v.x), h1=*reinterpret_cast<__half2*>(&v.y);
            vv[0]=__half2float(__low2half(h0)); vv[1]=__half2float(__high2half(h0));
            vv[2]=__half2float(__low2half(h1)); vv[3]=__half2float(__high2half(h1));
        }
        #pragma unroll
        for(int j=0;j<VPT;++j) o_run[j]=beta*o_run[j]+p*vv[j];
        m_run=m_new;
    }

    // intra-block combine over W warps (guarded -inf, same as fixed scalar version)
    __shared__ float sm[W],sl[W],so[W][D];
    if(lane==0){sm[warp]=m_run;sl[warp]=l_run;}
    #pragma unroll
    for(int j=0;j<VPT;++j) so[warp][off0+j]=o_run[j];
    __syncthreads();
    if(warp==0){
        float m_c=-INFINITY,l_c=0.0f,o_c[VPT];
        #pragma unroll
        for(int j=0;j<VPT;++j) o_c[j]=0.0f;
        #pragma unroll
        for(int w=0;w<W;++w){ float mw=sm[w],mn=fmaxf(m_c,mw);
            float a=(mn==-INFINITY)?0.0f:__expf(m_c-mn);
            float b=(mn==-INFINITY)?0.0f:__expf(mw-mn); l_c=a*l_c+b*sl[w];
            #pragma unroll
            for(int j=0;j<VPT;++j) o_c[j]=a*o_c[j]+b*so[w][off0+j]; m_c=mn; }
        const size_t base=((size_t)head*S+split);
        if(lane==0){ m_part[base]=m_c; l_part[base]=l_c; }
        #pragma unroll
        for(int j=0;j<VPT;++j) o_part[base*D+off0+j]=o_c[j];
    }
}

template <int D>
__global__ void splitk_combine_kernel2(
    const float* __restrict__ m_part, const float* __restrict__ l_part,
    const float* __restrict__ o_part, __half* __restrict__ o, int S)
{
    const int head=blockIdx.x, lane=threadIdx.x&31, ELEMS=D/32;
    float m_c=-INFINITY,l_c=0.0f,o_c[ELEMS];
    #pragma unroll
    for(int e=0;e<ELEMS;++e) o_c[e]=0.0f;
    for(int s=0;s<S;++s){
        const size_t base=((size_t)head*S+s);
        float m_s=m_part[base], l_s=l_part[base];
        float mn=fmaxf(m_c,m_s);
        float a=(mn==-INFINITY)?0.0f:__expf(m_c-mn);
        float b=(mn==-INFINITY)?0.0f:__expf(m_s-mn);
        l_c=a*l_c+b*l_s;
        #pragma unroll
        for(int e=0;e<ELEMS;++e) o_c[e]=a*o_c[e]+b*o_part[base*D+lane+e*32];
        m_c=mn;
    }
    __half* oh=o+(size_t)head*D;
    #pragma unroll
    for(int e=0;e<ELEMS;++e) oh[lane+e*32]=__float2half(o_c[e]/l_c);
}

torch::Tensor decode_attn_splitk_vec(torch::Tensor q, torch::Tensor K, torch::Tensor V, int64_t S, int64_t W){
    const int H=K.size(0),N=K.size(1),D=K.size(2);
    TORCH_CHECK(D==64||D==128,"D");
    auto fopt=torch::TensorOptions().dtype(torch::kFloat32).device(q.device());
    auto m_part=torch::empty({H,S},fopt), l_part=torch::empty({H,S},fopt), o_part=torch::empty({H,S,D},fopt);
    auto o=torch::empty({H,D},q.options()); const float scale=1.0f/sqrtf((float)D);
    dim3 pgrid(H,S), pblock(32*W);
    #define LP(Dv,Wv,Vv) splitk_partial_vec_kernel<Dv,Wv,Vv><<<pgrid,pblock>>>( \
        (const __half*)q.data_ptr<at::Half>(),(const __half*)K.data_ptr<at::Half>(),(const __half*)V.data_ptr<at::Half>(), \
        m_part.data_ptr<float>(),l_part.data_ptr<float>(),o_part.data_ptr<float>(),N,(int)S,scale)
    if(D==64){switch(W){case 2:LP(64,2,2);break;case 4:LP(64,4,2);break;case 8:LP(64,8,2);break;default:TORCH_CHECK(false,"W");}}
    else{switch(W){case 2:LP(128,2,4);break;case 4:LP(128,4,4);break;case 8:LP(128,8,4);break;default:TORCH_CHECK(false,"W");}}
    #undef LP
    TORCH_CHECK(cudaGetLastError()==cudaSuccess,"partial");
    dim3 cgrid(H),cblock(32);
    if(D==64) splitk_combine_kernel2<64><<<cgrid,cblock>>>(m_part.data_ptr<float>(),l_part.data_ptr<float>(),o_part.data_ptr<float>(),(__half*)o.data_ptr<at::Half>(),(int)S);
    else      splitk_combine_kernel2<128><<<cgrid,cblock>>>(m_part.data_ptr<float>(),l_part.data_ptr<float>(),o_part.data_ptr<float>(),(__half*)o.data_ptr<at::Half>(),(int)S);
    TORCH_CHECK(cudaGetLastError()==cudaSuccess,"combine");
    return o;
}
PYBIND11_MODULE(TORCH_EXTENSION_NAME,m){m.def("decode_attn_splitk_vec",&decode_attn_splitk_vec,"vectorized split-K");}
