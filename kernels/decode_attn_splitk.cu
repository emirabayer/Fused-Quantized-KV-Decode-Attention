#include <torch/extension.h>
#include <c10/cuda/CUDAException.h>
#include <cuda_fp16.h>
#include <cuda_runtime.h>

// ---- PARTIAL kernel: grid=(H, S), block = W warps. Block (h,s) does online-softmax
//      over its contiguous N-chunk and writes (m,l,o) partials to fp32 scratch. ----
template <int D, int W>
__global__ void splitk_partial_kernel(
    const __half* __restrict__ q,   // [H, D]
    const __half* __restrict__ K,   // [H, N, D]
    const __half* __restrict__ V,   // [H, N, D]
    float* __restrict__ m_part,     // [H, S]
    float* __restrict__ l_part,     // [H, S]
    float* __restrict__ o_part,     // [H, S, D]
    int N, int S, float scale)
{
    const int head = blockIdx.x;
    const int split = blockIdx.y;
    const int lane = threadIdx.x & 31, warp = threadIdx.x >> 5, ELEMS = D/32;

    const int chunk = (N + S - 1) / S;
    const int lo = split * chunk;
    const int hi = min(lo + chunk, N);

    const __half* qh = q + (size_t)head*D;
    const __half* Kh = K + (size_t)head*N*D;
    const __half* Vh = V + (size_t)head*N*D;

    float q_reg[ELEMS];
    #pragma unroll
    for(int e=0;e<ELEMS;++e) q_reg[e]=__half2float(qh[lane+e*32]);
    float m_run=-INFINITY, l_run=0.0f, o_run[ELEMS];
    #pragma unroll
    for(int e=0;e<ELEMS;++e) o_run[e]=0.0f;

    // stream this split's chunk, strided by warp within the chunk
    for(int i=lo+warp; i<hi; i+=W){
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
    // combine the W warps' partials within this block (shared-mem, like before)
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
        // write UNNORMALIZED partial (m, l, o) for this (head, split)
        const size_t base = ((size_t)head*S + split);
        if(lane==0){ m_part[base]=m_c; l_part[base]=l_c; }
        #pragma unroll
        for(int e=0;e<ELEMS;++e) o_part[base*D + lane + e*32] = o_c[e];
    }
}

// ---- COMBINE kernel: grid=H, one warp. Merge S partials -> final output. ----
template <int D>
__global__ void splitk_combine_kernel(
    const float* __restrict__ m_part,  // [H,S]
    const float* __restrict__ l_part,  // [H,S]
    const float* __restrict__ o_part,  // [H,S,D]
    __half* __restrict__ o,            // [H,D]
    int S)
{
    const int head=blockIdx.x, lane=threadIdx.x&31, ELEMS=D/32;
    float m_c=-INFINITY,l_c=0.0f,o_c[ELEMS];
    #pragma unroll
    for(int e=0;e<ELEMS;++e) o_c[e]=0.0f;
    for(int s=0;s<S;++s){
        const size_t base=((size_t)head*S + s);
        float m_s=m_part[base], l_s=l_part[base];
        float mn=fmaxf(m_c,m_s);
        float a=(mn==-INFINITY)?0.0f:__expf(m_c-mn);
        float b=(mn==-INFINITY)?0.0f:__expf(m_s-mn);
        l_c=a*l_c+b*l_s;
        #pragma unroll
        for(int e=0;e<ELEMS;++e) o_c[e]=a*o_c[e]+b*o_part[base*D + lane + e*32];
        m_c=mn;
    }
    __half* oh=o+(size_t)head*D;
    #pragma unroll
    for(int e=0;e<ELEMS;++e) oh[lane+e*32]=__float2half(o_c[e]/l_c);
}

torch::Tensor decode_attn_splitk(torch::Tensor q, torch::Tensor K, torch::Tensor V, int64_t S, int64_t W){
    const int H=K.size(0), N=K.size(1), D=K.size(2);
    TORCH_CHECK(D==64||D==128,"D");
    auto fopt = torch::TensorOptions().dtype(torch::kFloat32).device(q.device());
    auto m_part = torch::empty({H,S}, fopt);
    auto l_part = torch::empty({H,S}, fopt);
    auto o_part = torch::empty({H,S,D}, fopt);
    auto o = torch::empty({H,D}, q.options());
    const float scale = 1.0f/sqrtf((float)D);

    dim3 pgrid(H, S), pblock(32*W);
    #define LP(Dv,Wv) splitk_partial_kernel<Dv,Wv><<<pgrid,pblock>>>( \
        (const __half*)q.data_ptr<at::Half>(),(const __half*)K.data_ptr<at::Half>(),(const __half*)V.data_ptr<at::Half>(), \
        m_part.data_ptr<float>(),l_part.data_ptr<float>(),o_part.data_ptr<float>(),N,(int)S,scale)
    if(D==64){switch(W){case 2:LP(64,2);break;case 4:LP(64,4);break;case 8:LP(64,8);break;default:TORCH_CHECK(false,"W");}}
    else{switch(W){case 2:LP(128,2);break;case 4:LP(128,4);break;case 8:LP(128,8);break;default:TORCH_CHECK(false,"W");}}
    #undef LP
    TORCH_CHECK(cudaGetLastError()==cudaSuccess,"partial launch");

    dim3 cgrid(H), cblock(32);
    if(D==64) splitk_combine_kernel<64><<<cgrid,cblock>>>(m_part.data_ptr<float>(),l_part.data_ptr<float>(),o_part.data_ptr<float>(),(__half*)o.data_ptr<at::Half>(),(int)S);
    else      splitk_combine_kernel<128><<<cgrid,cblock>>>(m_part.data_ptr<float>(),l_part.data_ptr<float>(),o_part.data_ptr<float>(),(__half*)o.data_ptr<at::Half>(),(int)S);
    TORCH_CHECK(cudaGetLastError()==cudaSuccess,"combine launch");
    return o;
}
PYBIND11_MODULE(TORCH_EXTENSION_NAME,m){m.def("decode_attn_splitk",&decode_attn_splitk,"split-K decode attention");}
