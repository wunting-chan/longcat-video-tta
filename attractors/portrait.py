import glob, json, numpy as np, matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from analyze import style, COLOR, LABEL, FPS, PRE
style()
ms=['real','notta','rolling_notta','sf_always_search','sf_pseudo']
def load(m):
    files=sorted(glob.glob(f'obs/{m}/*.npz')); return np.stack([np.load(f)['low'][:537] for f in files]), [str(x) for x in np.load(files[0])['low_names']]
D={m:load(m) for m in ms}
names=D['notta'][1]
si,ki,fi=names.index('saturation'),names.index('sharpness'),names.index('flow_mag')
k=np.ones(FPS)/FPS
sm=lambda v: np.convolve(np.nan_to_num(v),k,'valid')
fig,axs=plt.subplots(2,3,figsize=(6.3,3.7))
ax=list(axs.ravel())
for a,m in zip(ax[:5],ms):
    L=D[m][0]
    for c in range(len(L)):
        x=sm(L[c,PRE:,si]); y=np.log10(sm(L[c,PRE:,ki])+1e-6)
        a.plot(x[::8],y[::8],color=COLOR[m],lw=0.25,alpha=0.18)
    xs=np.array([sm(L[c,PRE:,si]) for c in range(len(L))]); ys=np.log10(np.array([sm(L[c,PRE:,ki]) for c in range(len(L))])+1e-6)
    a.scatter(xs[:,0],ys[:,0],s=2,color='#999999',lw=0,label='t = 0.5 s',zorder=3)
    a.scatter(xs[:,-1],ys[:,-1],s=2.5,color=COLOR[m],lw=0,label='t = 31 s',zorder=4)
    a.plot(np.median(xs,0)[::8],np.median(ys,0)[::8],color='black',lw=0.9,zorder=5)
    a.set_title(LABEL[m],pad=3); a.set_xlabel('Saturation'); a.set_xlim(0,0.95); a.set_ylim(-3.2,-0.6)
ax[0].set_ylabel('log$_{10}$ sharpness'); ax[3].set_ylabel('log$_{10}$ sharpness'); ax[0].legend(loc='lower right',fontsize=4.4,markerscale=2)
a=ax[5]
for m in ms:
    L=D[m][0]; ts=np.arange(1,31)
    iqr=[]
    for s in ts:
        t=PRE+s*FPS; pc=np.nanmean(L[:,t-8:t+8,si],1); iqr.append((np.percentile(pc,75)-np.percentile(pc,25))/np.median(pc))
    a.plot(ts,iqr,color=COLOR[m],lw=0.8,label=LABEL[m])
a.set_title('Ensemble spread of saturation',pad=3); a.set_xlabel('Time after opening (s)'); a.set_ylabel('IQR / median across 128 clips'); a.legend(fontsize=4.2,loc='upper right')
fig.tight_layout(w_pad=0.6,h_pad=1.0)
fig.savefig('results/phase_portrait.png',dpi=230,bbox_inches='tight'); fig.savefig('results/phase_portrait.pdf',bbox_inches='tight')
