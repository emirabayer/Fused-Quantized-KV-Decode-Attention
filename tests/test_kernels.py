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
def skmods():
    sk =load(name="t_sk", sources=["kernels/decode_attn_splitk.cu"],    extra_cuda_cflags=["-O3"],verbose=False)
    skv=load(name="t_skv",sources=["kernels/decode_attn_splitk_vec.cu"],extra_cuda_cflags=["-O3"],verbose=False)
    return sk,skv

def _sdpa_mh(q,K,V):
    H,D=q.shape;N=K.shape[1]
    return F.scaled_dot_product_attention(q.view(H,1,1,D),K.view(H,1,N,D),V.view(H,1,N,D)).view(H,D)

# S not dividing N and N<S exercise the empty-split guards in the combine step
@pytest.mark.parametrize("H,N,D,S,W",[(32,1024,64,8,8),(32,4096,64,16,8),(32,4096,128,16,8),(8,100,64,7,4),(8,1,128,4,2)])
def test_splitk(skmods,H,N,D,S,W):
    sk,_=skmods; dev="cuda"; torch.manual_seed(0)
    q=torch.randn(H,D,dtype=torch.float16,device=dev).contiguous()
    K=torch.randn(H,N,D,dtype=torch.float16,device=dev).contiguous()
    V=torch.randn(H,N,D,dtype=torch.float16,device=dev).contiguous()
    assert torch.allclose(sk.decode_attn_splitk(q,K,V,S,W),_sdpa_mh(q,K,V),atol=2e-2)

@pytest.mark.parametrize("H,N,D,S,W",[(32,1024,64,8,8),(32,4096,128,16,8),(8,1,64,4,2)])
def test_splitk_vec(skmods,H,N,D,S,W):
    _,skv=skmods; dev="cuda"; torch.manual_seed(0)
    q=torch.randn(H,D,dtype=torch.float16,device=dev).contiguous()
    K=torch.randn(H,N,D,dtype=torch.float16,device=dev).contiguous()
    V=torch.randn(H,N,D,dtype=torch.float16,device=dev).contiguous()
    assert torch.allclose(skv.decode_attn_splitk_vec(q,K,V,S,W),_sdpa_mh(q,K,V),atol=2e-2)
