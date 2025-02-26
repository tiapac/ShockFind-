
import find_shocks_utils as fs
import matplotlib.pyplot as plt
import time
import numpy as np
import sys 
t0=time.time() 
#/home/mattia/codes/myramses/robe
#s=fs.shocks(outpath="/home/mattia/codes/ramses/simulatins/equilibrium", outnumb=3, mhd=True, lvl=7, bounds=None)
s=fs.shocks(outpath="/home/mattia/codes/ramses/simulatins/SS_SHOCK", outnumb=1, mhd=True, lvl=7, bounds=None)
#s=fs.shocks(outpath="/home/mattia/codes/CR/cr_particles_multip_MESS", outnumb=285, mhd=True, lvl=7, bounds=None)

#s=fs.shocks(outpath="/home/mattia/codes/ramses/simulatins/equilibrium/", outnumb=6, mhd=True, lvl=7, bounds=None)
for ax in "xyz":
    for f in [
        
        #in ["velocity_x_gradient_x","velocity_x_gradient_y","velocity_x_gradient_z"
        
        #]:#,
        "velocity_x","velocity_y","velocity_z"]:
        ff=s.ds.Splot(f, axis=ax, log=False)
        ff.save()
s.extra_params(gamma=5./3.)

ss=s.run(load_data = False,  load_data_cube=True, parallel=False,
         save_data = True,  save_data_cube=True,
          quiet=False,
          ca=0.05,#!*1.29603957e+07,
          cs=0.05,#*9.53444838e+05,
          #rhomean=-24,
          #Tlim=1.0e1,
          #ncpus=8,
          
          )




t=time.time()
print("time spent for the full analysis %.3f s"%(t-t0))
fig0=plt.figure(0)
##ax0=fig0.subplots()
s.plot3D(alpha=0.05, fig=fig0)
#plt.show()

#quit()
#st=s.plot2D("velocity_magnitude",log=False, axis="x")
#st.save()
fig1=plt.figure(1)
s.plot2DD("velocity_magnitude",log=True, lim=5, axis="z", fig=fig1, scale=3.086e18)
ss,header, extra, shockslist=ss[1]
#print("header", header)
condF   = np.logical_and(ss[6]==12, ss[15]==0  )
condS   = np.logical_and(ss[6]==34, ss[15]==0  )
condboh = ss[6]==0 
#plt.show()
ind=1
fig=plt.figure(2)
ax=fig.subplots()
bbins=50
alpha=0.5
#alpha1=1
ax.hist(ss[ind][condF],  log=False, color="midnightblue", bins=bbins, alpha=alpha, label="Fast n = %5d"%len(ss[ind][condF]))
ax.hist(ss[ind][condS],  log=False, color="red",bins=bbins, alpha=alpha,           label="Slow n = %5d"%len(ss[ind][condS]))
ax.hist(ss[ind][condboh],log=False, color="green", bins=bbins, alpha=alpha,        label="???  n = %5d"%len(ss[ind][condboh]))
ax.legend()
##ax.set_xlabel(r"$v_{sh}\,\,[km\,s^{-1}]$")
##ax.set_ylabel(r"$N_\{rm cells}$")
##ax.set_xlim(ss[ind].min(), ss[ind].max())
plt.show()
#