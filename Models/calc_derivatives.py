import sympy
from sympy.matrices import Matrix

def difftotal(expr, diffby, diffmap):
    """Take the total derivative with respect to a variable.

    Example:

        theta, t, theta_dot = symbols("theta t theta_dot")
        difftotal(cos(theta), t, {theta: theta_dot})

    returns

        -theta_dot*sin(theta)
    """
    # Replace all symbols in the diffmap by a functional form
    # fnexpr = expr.subs({s for s in diffmap})
    fnexpr = expr
    # Do the differentiation
    diffexpr = sympy.diff(fnexpr, diffby)
    # Replace the Derivatives with the variables in diffmap
    derivmap = {}
    for v,dv in diffmap.items():
        derivmap[sympy.Derivative(v(diffby), diffby)] = dv(diffby)
        
    finaldiff = diffexpr.subs(derivmap)
    # Replace the functional forms with their original form
    return finaldiff

# I understand that this is possible: 
x = sympy.Symbol('x')  
y = sympy.Symbol('y')
psi = sympy.Symbol('psi')
v = sympy.Symbol('v')
K = sympy.Symbol('K')
s = sympy.Symbol('s')

xN = sympy.Symbol('xN')
yN = sympy.Symbol('yN')
psiN = sympy.Symbol('psiN')
vN = sympy.Symbol('vN')
KN = sympy.Symbol('KN')
sN = sympy.Symbol('sN')

xS = sympy.Function('xS')
yS = sympy.Function('yS')
psiS = sympy.Function('psiS')
s = sympy.Symbol('s')
sL = sympy.Symbol('sL')

R = sympy.Symbol('R')

e_y = -(x-xS(s))*sympy.sin(psiS(s)) + (y-yS(s))*sympy.cos(psiS(s))
e_psi = psi-psiS(s)
d = s-sL

e_y_N = -(xN-xS(sN))*sympy.sin(psiS(sN)) + (yN-yS(sN))*sympy.cos(psiS(sN))
e_psi_N = psiN-psiS(sN)
d_N = sN-sL

# l = 
# lf = 

X = Matrix([x, y, psi, v])
XN = Matrix([xN, yN, psiN])
Y = Matrix([v, e_y, e_psi])
U = Matrix([v, K])
YN = Matrix([vN, e_y_N, e_psi_N])

Y_sym = sympy.MatrixSymbol('Y', 4, 1)
U_sym = sympy.MatrixSymbol('U', 2, 1)
YN_sym = sympy.MatrixSymbol('YN', 4, 1)

Q = sympy.MatrixSymbol('Q', 4, 4)
R = sympy.MatrixSymbol('R', 2, 2)
S = sympy.MatrixSymbol('S', 4, 4)
w_aL = sympy.Symbol('w_aL')

l = (Y_sym.transpose()*Q*Y_sym + U_sym.transpose()*R*U_sym)/2
lF = (YN_sym.transpose()*S*YN_sym)

print(e_y)
print(e_psi)
print(d)

print(l)
print(lF)

J = l+lF

print("J:")
print(J)

dY_xx = Y.diff(X).diff(x)
dJ_y = J.diff(Y_sym) 
dY_x = Y.diff(X)
dJ_xy = J.diff(X).diff(Y)

dY_X = Y.diff(X)
dY_XX = dY_X.diff(X)

print("Derivatives:")
print(dY_xx)
print(dJ_y)
print(dY_x)
print(dJ_xy)

import pdb; pdb.set_trace()

# Q_lqr = 
# R_lqr =
# N_lqr =
# F_lqr = 