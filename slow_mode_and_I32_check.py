import numpy as np
from numpy.polynomial.legendre import leggauss
from ratio_quadrature import partial_N

# ---------- Task 1: R(k_z) resolved ----------
print("Task 1: R(k_z) = <A3D>_bg / <A2D>_bg  (a=1/2)")
kzs = [0.001,0.005,0.02,0.05,0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9,0.97]
Rs = []
for z in kzs:
    N2,N3 = partial_N(z,0.5,nt=80,nph=64,nkb=90)
    Rs.append(N3/N2)
    print(f"  k_z={z:5.3f}  R={N3/N2:6.3f}")
import json; json.dump(dict(kz=kzs,R=Rs), open("/tmp/Rprofile.json","w"))

# ---------- Task 2: is I32 = <A3D^2/A2D>_bg finite? Test the collinear edge (v->+-1/2). ----------
# I32 density (subst vars) = 4 (u^2-v^2) A3D^2 W / (cos(phi) sinh(t)); the 1/cos(phi) at the
# collinear edge v=1/2 is the suspect. Cut phi at arcsin(1-2*10^-d) and watch I32 vs d.
def _gl(n,a,b):
    x,w=leggauss(n); return 0.5*(b-a)*x+0.5*(b+a), 0.5*(b-a)*w

def I32_reg(z,a,d,nt=80,nph=90,nkb=80,Tmax=26.0):
    rhoA2=1-z*z; eps=10.0**(-d)
    tlo=np.arccosh(1+2*eps); phc=np.arcsin(1-2*eps)
    t,wt=_gl(nt,tlo,Tmax); ph,wph=_gl(nph,-phc,phc)
    u=0.5*np.cosh(t); v=0.5*np.sin(ph); u2=u*u; v2=v*v
    U2V2=u2[:,None]-v2[None,:]; UPV=u[:,None]+v[None,:]; UMV=u[:,None]-v[None,:]
    SH=np.sinh(t)[:,None]+0*ph[None,:]; CO=np.cos(ph)[None,:]+0*u[:,None]
    P2=2*u2[:,None]+2*v2[None,:]-1; wtph=wt[:,None]*wph[None,:]
    tot=0.0
    for seg in("M","R","L"):
        if seg=="M":
            if z<=2*eps: continue
            xi,wk=_gl(nkb,-np.pi/2,np.pi/2); c=np.cos(xi)
            kb=-z/2+(z/2)*np.sin(xi); SJ=wk*(z/2)**(1-2*a)*c**(1-2*a); brf=1.0
        else:
            r,wr=_gl(nkb,0.0,1.0); s=r/(1-r); dsdr=1/(1-r)**2
            SJ=wr*2*s**(1-2*a)*dsdr
            if seg=="R": kb=s*s; brf=np.abs(z+kb)**(-a)
            else: kb=-z-s*s; brf=np.abs(z+s*s)**(-a)
        kb=kb[None,None,:]; kg=z+kb; upv=UPV[:,:,None]; umv=UMV[:,:,None]
        beta2=rhoA2*upv**2+kb**2; gam2=rhoA2*umv**2+kg**2
        W=beta2**(-(3-a)/2)*gam2**(-(3-a)/2)*brf
        X=rhoA2*P2[:,:,None]-z*z+kb**2+kg**2
        A3D=0.25*np.sqrt(np.clip(4*beta2*gam2-X**2,0,None))
        dens=4*U2V2[:,:,None]*A3D**2*W/(CO[:,:,None]*SH[:,:,None])
        tot+=np.sum(wtph[:,:,None]*SJ[None,None,:]*dens)
    return tot

print("\nTask 2: I32 = <A3D^2/A2D>_bg at k_z=0.3, vs collinear-edge cutoff d (v<1/2-10^-d):")
prev=None
for d in [2,3,4,5,6,7]:
    v=I32_reg(0.3,0.5,d)
    dv='' if prev is None else f'  d-increment={v-prev:+.3f}'
    print(f"  d={d}:  I32={v:9.4f}{dv}"); prev=v

# check A3D, A2D as v->1/2 at fixed u,kb,z to see if A3D stays nonzero
print("\n  A3D and A2D approaching the collinear edge v->1/2 (u=1.0, k_bz=0.3, k_az=0.3):")
z=0.3; rhoA2=1-z*z; u=1.0; kb=0.3; kg=z+kb
for v in [0.4,0.49,0.499,0.4999]:
    a2=(rhoA2/4)*np.sqrt((1-4*v*v)*(4*u*u-1))
    b2=rhoA2*(u+v)**2+kb*kb; g2=rhoA2*(u-v)**2+kg*kg
    X=rhoA2*(2*u*u+2*v*v-1)-z*z+kb*kb+kg*kg
    a3=0.25*np.sqrt(max(4*b2*g2-X*X,0))
    print(f"    v={v:6.4f}:  A2D={a2:.5f}  A3D={a3:.5f}  A3D/A2D={a3/a2:8.3f}")
