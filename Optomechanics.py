#!/usr/bin/env python



from qutip import *
from scipy import *

 

def correlation_ss_gtt(H, tlist, c_ops, a_op, b_op, c_op, d_op, rho0=None):
    """
    Calculate the correlation function <A(0)B(tau)C(tau)D(0)>

    (ss_gtt = steadystate general two-time)
    
    See, Gardiner, Quantum Noise, Section 5.2.1

    .. note::
        Experimental. 
    """
    if rho0 == None:
        rho0 = steadystate(H, c_ops)

    return mesolve(H, d_op * rho0 * a_op, tlist, c_ops, [b_op * c_op]).expect[0]

# Parameters
g0 = 1e-3
kappaInt = 500#1e6
GammaM = 0.5
kappaExt = 500#1e6
kappa = kappaInt + kappaExt
OmegaM = 2*pi*6e9
nB = 1 # Bath temperature
eta = 0 # whether to simulate counter rotating terms (1: yes, 0: no)
xzpf = 1e-15
#hbar = 1.05457173e-34
hbar = 1
alphap = 1*1000
alpham = 40*1000
Gp = g0*alphap
Gm = g0*alpham


N = 5 # dimensionality of optical mode Hilbert space
M = 5 # dimensionality of mechanical mode Hilbert space
a=destroy(N)  # Define optical mode annhilation operator
ad=create(N)  # Idem for creation operator
b=destroy(M)  # Define mechanical mode annhilation operator
bd=create(M)  # Idem for creation operator


# Define optomechanical hamiltonian
#Hom=tensor((a.dag())*a,qeye(4))+tensor(qeye(4),(b.dag())*b)
#Hlsb=g0*tensor((a.dag())*a,(b+b.dag()))

if eta == 1:  # include off-resonant terms
    Hom = -hbar*(Gp+eta*Gm*exp(2j*OmegaM*t))*tensor(a.dag(),b.dag()) \
		-hbar*(Gm+eta*Gp*exp(-2j*OmegaM*t))*tensor(a.dag(),b) \
		-hbar*(Gp.conjugate()+eta*Gm.conjugate()*exp(-2j*OmegaM*t))*tensor(a,b) \
		-hbar*(Gm.conjugate()+eta*Gp.conjugate()*exp(2j*OmegaM*t))*tensor(a,b.dag())
else:
    Hom = -hbar*Gp*tensor(a.dag(),b.dag()) \
		-hbar*Gm*tensor(a.dag(),b) \
		-hbar*Gp.conjugate()*tensor(a,b) \
		-hbar*Gm.conjugate()*tensor(a,b.dag())


# collapse operators appearing in Linblad master equation to simulate optical and mechanical
# coupling to the environment. Optics couples to a 0 temperature bath so it only
# introduces a decaying term.
# Mechanics couples to a bath at finite temperature so it introduces two terms,
# one corresponding to the phonon decay and on to phonon injection from the bath
# We start with an empty list of collapse operators and add them one by one
# for each decay channel
c_ops = []

# Optical decay to a bath at nPh = 0
OpticalDecRate = kappa
if OpticalDecRate > 0.0:
    c_ops.append(sqrt(OpticalDecRate) * tensor(a,qeye(M)))

# Mechanical decay happens at a rate GammaM*(1+nB) and introduces a dissipating term 
# which destroys mechanical phonons (b) 
MechDecayRate = GammaM * (1 + nB)
if MechDecayRate > 0.0:
    c_ops.append(sqrt(MechDecayRate) * tensor(qeye(N),b)) #decay operators

# Mechanical heating happens a rate GammaM*nB and introduces a term that 
# creates phonons (b.dag())
MechHeatingRate = GammaM * nB
if MechHeatingRate > 0.0:
    c_ops.append(sqrt(MechHeatingRate) * tensor(qeye(N),b.dag())) #excitation operators

#c_ops = [sqrt(kappa) * tensor(a,qeye(M)), sqrt(nB*GammaM) * tensor(qeye(N),b.dag()) , sqrt((nB+1)*GammaM) * tensor(qeye(N),b)]

# Initial state (optics in Fock state 1 and mechanics in ground state)
psi0 = tensor(ket2dm(basis(N,0)),ket2dm(basis(M,0)))

# list of times for which the solver should store the state vector
tlist = linspace(0., 1, 100)

def Ht(t,tred,twait,tblue):
    tperiod = tred + tblue
    tp = t - int(t/tperiod)
    if tp < tred:
        

# Solve the master eq
result = mesolve(Hom, psi0, tlist, c_ops, [])

# Calculate expectation value
photN=expect(tensor(a.dag()*a,qeye(M)), result.states)
photM=expect(tensor(qeye(N),b.dag()*b), result.states)

# Calculate correlation function
tlistg2 = linspace(0., 0.01, 100)
G2 = correlation_ss_gtt(Hom, tlistg2, c_ops, tensor(a,qeye(M)), tensor(a,qeye(M)), tensor(a.dag(),qeye(M)), tensor(a.dag(),qeye(M)), psi0)

import matplotlib.pyplot as plt
#plt.rc('text', usetex=True)
#plt.rc('font', family='serif')
plt.clf()
plt.show()
plt.subplot(2,1,1)
plt.cla()
ax1 = plt.gca()
plt.plot(tlist,photN)
ax1.set_ylabel(r"Average photon number", fontsize=16);
ax2 = ax1.twinx()
ax2.plot(tlist,photM)
plt.xlabel(r'$t$', fontsize=20)
ax2.set_ylabel(r"Average phonon number", fontsize=16);
plt.legend(('Optics', 'Mechanics'))

plt.subplot(2,1,2)
plt.cla()
plt.plot(tlistg2,G2)
plt.xlabel('tau')
plt.ylabel('G^(2)(tau)')

