#!/usr/bin/env python



from qutip import *
from scipy import *


def correlation_me_4op_2t_notStat(H, rho0, tlist, taulist, c_ops,
                           a_op, b_op, c_op, d_op, reverse=False,
                           options=Odeoptions()):
    """
    Calculate the four-operator two-time correlation function on the form
    <A(t)B(t+tau)C(t+tau)D(t)>.

    See, Gardiner, Quantum Noise, Section 5.2.1
    """

    if debug:
        print(inspect.stack()[0][3])

    if rho0 is None:
        rho0 = steadystate(H, c_ops)
    elif rho0 and isket(rho0):
        rho0 = ket2dm(rho0)

    C_mat = np.zeros([np.size(tlist), np.size(taulist)], dtype=complex)

    rho_t = mesolve(H, rho0, tlist, c_ops, [], options=opts).states
    print(expect(tensor(qeye(N),b.dag()*b),rho_t[1]))
    
    for t_idx, rho in enumerate(rho_t):
        t0 = tlist[t_idx]
        C_mat[t_idx, :] = mesolve(H, d_op * rho * a_op, taulist,
                                  c_ops, [b_op * c_op], args = {'t0': t0}, 
                                  options=opts).expect[0]

    return C_mat

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

def correlation_g2tau(H, tlist, c_ops, a_op, b_op, c_op, d_op, rho0=None):
    """
    Calculate the correlation function <A(t)B(t+tau)C(t+tau)D(t)>

    (ss_gtt = steadystate general two-time)
    
    See, Gardiner, Quantum Noise, Section 5.2.1

    .. note::
        Experimental. 
    """
    G2 = zeros(tlist.shape)
    if rho0 == None:
        rho0 = steadystate(H, c_ops)
    #for t in tlist:
    t = 0.01
    tlist1 = linspace(0., t, 2)
    result = mesolve(H, rho0, tlist1, c_ops, [])
    rhoInit = result.states[-1]
    G2 = G2 + mesolve(H, d_op * rhoInit * a_op, tlist, c_ops, [b_op * c_op]).expect[0]
    return G2


# Parameters (in MHz, us)
g0 = 2*pi*1.1
kappaInt = 2*pi*0#1e6
GammaM = 2*pi*7.5e-3
kappaExt = 2*pi*120#1e6
kappa = kappaInt + kappaExt
OmegaM = 2*pi*5.1e3


nB = 6.4 # Bath temperature
nI = 0.01 # Initial mode temperature
eta = 0 # whether to simulate counter rotating terms (1: yes, 0: no)
xzpf = 1e-15
#hbar = 1.05457173e-34
hbar = 1

alphaBlue = .1
alphaRed = 100
Gp = g0*alphaBlue
Gm = g0*alphaRed

# Pulse timing
tred = 10
tblue = 0.1
twait = 1
CoincidenceWindow = 1e-3
tperiod = tred + tblue + twait

N = 3 # dimensionality of optical mode Hilbert space
M = 3 # dimensionality of mechanical mode Hilbert space
a=destroy(N)  # Define optical mode annhilation operator
ad=create(N)  # Idem for creation operator
b=destroy(M)  # Define mechanical mode annhilation operator
bd=create(M)  # Idem for creation operator

# Some solver options
opts        = Odeoptions()
#opts.atol   = 1e-8
#opts.rtol   = 1e-8
#opts.nsteps = 1000 
opts.max_step = 0.001



# Time dependent hamiltonian coeffiecients for the pulsed case

# Blue laser coefficient (downconversion term)
def HtcoeffBlue(t, args):    
    tp = t - tperiod*floor(t/tperiod)
    if 0 < tp < tblue:
        return 1
    else:
        return 0

# Red laser coefficient
def HtcoeffRed(t, args):
    tp = t - tperiod*floor(t/tperiod)
    if  tblue + twait < tp < tred + twait + tblue:        
        return 1
    else:
        return 0
        
# Red laser coefficient for conditional g2
def HtcoeffG2Red(t, args):
    #tp = t - tperiod*floor(t/tperiod)
    if len(args) > 0:
        t0 = args['t0']
    else:
        t0 = 0
    if  twait < t0 + t < tred + twait:        
        return 1
    else:
        return 0
        
    # Define optomechanical hamiltonian
    #Hom=tensor((a.dag())*a,qeye(4))+tensor(qeye(4),(b.dag())*b)
    #Hlsb=g0*tensor((a.dag())*a,(b+b.dag()))
    

if eta == 1:  # include off-resonant terms
    HomBlue = -hbar*Gp*tensor(a.dag(),b.dag()) \
              -hbar*Gp*exp(-2j*OmegaM*t)*tensor(a.dag(),b) \
              -hbar*(Gp.conjugate())*tensor(a,b) \
              -hbar*(Gp.conjugate()*exp(2j*OmegaM*t))*tensor(a,b.dag())
              
    HomRed =  -hbar*(Gm*exp(2j*OmegaM*t))*tensor(a.dag(),b.dag()) \
              -hbar*Gm*tensor(a.dag(),b) \
		  -hbar*(Gm.conjugate()*exp(-2j*OmegaM*t))*tensor(a,b) \
		  -hbar*(Gm.conjugate())*tensor(a,b.dag())
    
else:
    HomBlue = -hbar*Gp*tensor(a.dag(),b.dag()) \
             -hbar*Gp.conjugate()*tensor(a,b) 
    HomRed  = -hbar*Gm*tensor(a.dag(),b) \
		-hbar*Gm.conjugate()*tensor(a,b.dag())
  
#    Hom = -hbar*Gp*tensor(a.dag(),b.dag()) \
#		-hbar*Gm*tensor(a.dag(),b) \
#		-hbar*Gp.conjugate()*tensor(a,b) \
#		-hbar*Gm.conjugate()*tensor(a,b.dag())
  
Hom=[0*tensor(a,b),[HomBlue,HtcoeffBlue], [HomRed,HtcoeffRed]]
HomG2 = [0*tensor(a,b),[HomRed,HtcoeffG2Red]]
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
psi0 = tensor(ket2dm(basis(N,0)),thermal_dm(M,nI))

# list of times for which the solver should store the state vector
tlist = linspace(0., tperiod, 50)

# Solve the master eq
result1 = mesolve(Hom, psi0, tlist, c_ops, [])

# Density matrix at the end of the blue pulse
psiF = result1.states[-1]

# Asume Single photon detection at time t1

# Calculate expectation value
photN=expect(tensor(a.dag()*a,qeye(M)), result1.states)
photM=expect(tensor(qeye(N),b.dag()*b), result1.states)

# Time evolution conditioned on detection of first photon and assuming only red laser on
# Initial condition for projected density matrix
psi0P = tensor(ket2dm(basis(N,0)),ket2dm(basis(M,1)))
tlist_cond = linspace(0., tred, 50)
# Solve the master eq
result2 = mesolve(HomG2, psi0P, tlist_cond, c_ops, [])
# Calculate photon and phonon number
photN_cond=expect(tensor(a.dag()*a,qeye(M)), result2.states)
photM_cond=expect(tensor(qeye(N),b.dag()*b), result2.states)

# Calculate correlation function
tlistg2 = linspace(twait, twait+CoincidenceWindow, 2)
taulistg2 = linspace(0, tred-twait-2*CoincidenceWindow, 100)

# Initial condition for projected density matrix
psi0P = tensor(ket2dm(basis(N,0)),ket2dm(basis(M,1)))

#G2 = correlation_ss_gtt(HomRed, tlistg2, c_ops, tensor(a.dag(),qeye(M)), tensor(a.dag(),qeye(M)), tensor(a,qeye(M)), tensor(a,qeye(M)), psi0P)
#G2 = correlation_g2tau(HomRed, tlistg2, c_ops, tensor(a.dag(),qeye(M)), tensor(a.dag(),qeye(M)), tensor(a,qeye(M)), tensor(a,qeye(M)), psi0P)
G2mat = correlation_me_4op_2t_notStat(HomG2,psi0P,tlistg2,taulistg2,c_ops,tensor(a.dag(),qeye(M)), tensor(a.dag(),qeye(M)), tensor(a,qeye(M)), tensor(a,qeye(M)))
#G2 = G2mat.sum(axis=0)/(photN_cond*photN_cond[0])
#psiSS=steadystate(HomRed,c_ops)
#finalN = expect(tensor(a.dag()*a,qeye(M)),psiSS)
#G2 = G2mat.sum(axis=0)
G2 = G2mat[1]
G2 = G2/G2[-1]

#%% Plot stuff
import matplotlib.pyplot as plt
plt.rc('text', usetex=False)
plt.rc('font', family='serif')
plt.rc('font', size='16')
plt.clf()
plt.show()
plt.subplot(3,1,1)
plt.subplots_adjust(hspace=.5)
plt.cla()
plt.title('Dynamics of full cycle')
ax1 = plt.gca()
pl1=ax1.plot(tlist,photN)
ax1.ticklabel_format(style='sci', axis='y', scilimits=(0,0))
plt.rc('text', usetex=True)
ax1.set_ylabel(r"$\left<a^\dag(t) a(t)\right>$", fontsize=16,color='b');
plt.rc('text', usetex=False)
ax2 = ax1.twinx()
pl2=ax2.plot(tlist,photM,'g')
ax2.ticklabel_format(style='sci', axis='y', scilimits=(0,0))
pl = pl1+pl2
plt.rc('text', usetex=True)
plt.xlabel(r'$t (\mu s)$')

ax2.set_ylabel(r"$\left<b^\dag(t) b(t)\right>$", fontsize=16,color='g');
plt.rc('text', usetex=False)
plt.axvspan(0, tblue, facecolor='b', alpha=0.3)
plt.axvspan(tblue+twait, tperiod, facecolor='r', alpha=0.3)


ax1.legend(pl,('Optics', 'Mechanics'), loc=0)

plt.subplot(3,1,2)
ax1 = plt.gca()
plt.title('Dynamics during swapping pulse')
pl1=ax1.plot(tlist_cond,photN_cond)
plt.rc('text', usetex=True)
ax1.set_ylabel(r"$\left<a^\dag(t) a(t)\right>$", fontsize=16,color='b');
plt.rc('text', usetex=False)
ax2 = ax1.twinx()
pl2=ax2.plot(tlist_cond,photM_cond,'g')
pl = pl1+pl2
plt.xlabel(r't', fontsize=20)
plt.rc('text', usetex=True)
ax2.set_xlabel(r'$t (\mu s)$')
ax2.set_ylabel(r"$\left<b^\dag(t)b(t)\right>$", fontsize=16,color='g');
plt.rc('text', usetex=False)
ax1.legend(pl,('Optics', 'Mechanics'), loc=0)

plt.subplot(3,1,3)
plt.cla()

plt.plot(taulistg2,G2)


