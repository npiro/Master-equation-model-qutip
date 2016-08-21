from qutip import *
# Define atomic states. Use ordering from paper
ustate = basis(3,0)
excited = basis(3,1)
ground = basis(3,2)

# Set where to truncate Fock state for cavity
N = 2

# Create the atomic operators needed for the Hamiltonian
sigma_ge = tensor(qeye(N), ground * excited.dag()) # |g><e|
sigma_ue = tensor(qeye(N), ustate * excited.dag()) # |u><e|

# Create the photon operator
a = tensor(destroy(N), qeye(3))
ada = tensor(num(N), qeye(3))

# Define collapse operators
c_op_list = []
# Cavity decay rate
kappa = 1.5
c_op_list.append(sqrt(kappa) * a)

# Atomic decay rate
gamma = 6 #decay rate
# Use Rb branching ratio of 5/9 e->u, 4/9 e->g
c_op_list.append(sqrt(5*gamma/9) * sigma_ue)
c_op_list.append(sqrt(4*gamma/9) * sigma_ge)

# Define time vector
t = linspace(-15,15,100)

# Define initial state
psi0 = tensor(basis(N,0), ustate)

# Define states onto which to project
state_GG = tensor(basis(N,1), ground)
sigma_GG = state_GG * state_GG.dag()
state_UU = tensor(basis(N,0), ustate)
sigma_UU = state_UU * state_UU.dag()

# Set up the time varying Hamiltonian
g = 5 #coupling strength
H0 = -g * (sigma_ge.dag() * a + a.dag() * sigma_ge) #time-INDEPENDENT term
H1 = (sigma_ue.dag() + sigma_ue) #time-DEPENDENT term

def H1_coeff(t, args):
        return 9 * exp(-(t/5.)**2)
        
H=[H0,[H1,H1_coeff]]
output = mesolve(H, psi0, t, c_op_list,[ada, sigma_UU, sigma_GG])