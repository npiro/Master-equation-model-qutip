# -*- coding: utf-8 -*-
"""
Created on Sun Nov 10 15:57:40 2013

@author: piromast
"""
from qutip import *

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