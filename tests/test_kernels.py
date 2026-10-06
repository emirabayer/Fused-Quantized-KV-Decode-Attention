import pytest, torch, math
import torch.nn.functional as F
from torch.utils.cpp_extension import load
pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA only")

@pytest.fixture(scope="module")
def mods():
    fp16=load(name="t_fp16",sources=["kernels/decode_attn_fp16_mw.cu"],extra_cuda_cflags=["-O3"],verbose=False)
    int8=load(name="t_int8",sources=["kernels/decode_attn_int8_mw.cu"],extra_cuda_cflags=["-O3"],verbose=False)
    int4=load(name="t_int4",sources=["kernels/decode_attn_int4.cu"],extra_cuda_cflags=["-O3"],verbose=False)
    gqa =load(name="t_gqa", sources=["kernels/decode_attn_gqa.cu"], extra_cuda_cflags=["-O3"],verbose=False)
    return fp16,int8,int4,gqa

def _sdpa(q,K,V):
    B,D=q.shape;N=K.shape[1]
    return F.scaled_dot_product_attention(q.view(B,1,1,D),K.view(B,1,N,D),V.view(B,1,N,D)).view(B,D)

@pytest.mark.parametrize("B,N,W",[(1,1024,8),(8,2048,4),(256,4096,8)])
def test_fp16(mods,B,N,W):
    fp16,_,_,_=mods; dev="cuda"; torch.manual_seed(0)
    q=torch.randn(B,64,dtype=torch.float16,device=dev)
    K=torch.randn(B,N,64,dtype=torch.float16,device=dev)
    V=torch.randn(B,N,64,dtype=torch.float16,device=dev)
    assert torch.allclose(fp16.decode_attn_fp16_mw(q,K,V,W),_sdpa(q,K,V),atol=2e-2)

@pytest.mark.parametrize("B,N,W",[(1,1024,8),(8,2048,4)])
def test_int8(mods,B,N,W):
    _,int8,_,_=mods; dev="cuda"; torch.manual_seed(0)
    q=torch.randn(B,64,dtype=torch.float16,device=dev)
    K=torch.randn(B,N,64,dtype=torch.float16,device=dev)
    V=torch.randn(B,N,64,dtype=torch.float16,device=dev)
    def qpt(X):
        a=X.abs().amax(2,keepdim=True);s=(a/127.0).clamp(min=1e-8)
        return torch.clamp(torch.round(X/s),-127,127).to(torch.int8).contiguous(),s.squeeze(2).half().contiguous()
    Kq,Ks=qpt(K);Vq,Vs=qpt(V)
    out=int8.decode_attn_int8_mw(q,Kq,Vq,Ks,Vs,W)
    Kd=(Kq.float()*Ks.float()[:,:,None]).half();Vd=(Vq.float()*Vs.float()[:,:,None]).half()
    ref=F.scaled_dot_product_attention(q.view(B,1,1,64),Kd.view(B,1,N,64),Vd.view(B,1,N,64)).view(B,64)
    assert torch.allclose(out,ref,atol=3e-2)

@pytest.mark.parametrize("B,N,W",[(1,1024,8),(16,4096,8)])
def test_gqa(mods,B,N,W):
    _,_,_,gqa=mods; dev="cuda"; torch.manual_seed(0)
    Hq,Hkv,D=32,8,64;G=Hq//Hkv
    q=torch.randn(B,Hq,D,dtype=torch.float16,device=dev)
    K=torch.randn(B,Hkv,N,D,dtype=torch.float16,device=dev)
    V=torch.randn(B,Hkv,N,D,dtype=torch.float16,device=dev)
    out=gqa.decode_attn_gqa(q,K,V,W)
    Ke=K.repeat_interleave(G,1);Ve=V.repeat_interleave(G,1)
    ref=F.scaled_dot_product_attention(q.view(B,Hq,1,D),Ke,Ve).view(B,Hq,D)
    assert torch.allclose(out,ref,atol=2e-2)

@pytest.fixture(scope="module")
def gskmod():
    return load(name="t_gsk",sources=["kernels/decode_attn_gqa_sk_fast.cu"],extra_cuda_cflags=["-O3"],verbose=False)

# GQA-aware split-K reads compact Hkv-head KV. N<S, N=1, and N not dividing S exercise the empty-split
# guards in both combine stages. Hq=32, Hkv=8 (G=4).
@pytest.mark.parametrize("N,D,S,W,CW",[(1024,64,8,8,4),(4096,128,16,8,4),(16384,64,64,8,8),(3,128,8,4,4),(1,64,4,2,4),(100,64,7,4,4)])
def test_gqa_sk_fast(gskmod,N,D,S,W,CW):
    dev="cuda"; torch.manual_seed(0); Hq,Hkv=32,8; G=Hq//Hkv
    q=torch.randn(Hq,D,dtype=torch.float16,device=dev).contiguous()
    K=torch.randn(Hkv,N,D,dtype=torch.float16,device=dev).contiguous()
    V=torch.randn(Hkv,N,D,dtype=torch.float16,device=dev).contiguous()
    mp=torch.empty(Hq,S,dtype=torch.float32,device=dev);lp=torch.empty(Hq,S,dtype=torch.float32,device=dev)
    op=torch.empty(Hq,S,D,dtype=torch.float32,device=dev);out=torch.empty(Hq,D,dtype=torch.float16,device=dev)
    gskmod.gqa_sk_fast(q,K,V,mp,lp,op,out,S,W,CW)
    kE=K.repeat_interleave(G,0);vE=V.repeat_interleave(G,0)
    ref=F.scaled_dot_product_attention(q.view(Hq,1,1,D),kE.view(Hq,1,N,D),vE.view(Hq,1,N,D)).view(Hq,D)
    assert torch.allclose(out,ref,atol=2e-2)
