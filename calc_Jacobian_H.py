import sympy
from sympy.matrices import Matrix

# I understand that this is possible: 
x = sympy.Symbol('x')  
y = sympy.Symbol('y')
psi = sympy.Symbol('psi')
v = sympy.Symbol('v')
delta = sympy.Symbol('delta')

xc = sympy.Symbol('xc')
yc = sympy.Symbol('yc')
psic = sympy.Symbol('psic')

R = sympy.Symbol('R')

x_sigma = xc + (x-xc)*R/sympy.sqrt((x-xc)**2 + (y-yc)**2)
y_sigma = yc + (y-yc)*R/sympy.sqrt((x-xc)**2 + (y-yc)**2)
psi_sigma = sympy.atan2((x-xc), (y-yc))

e_y = -(x-x_sigma)*sympy.sin(psi_sigma) + (y-y_sigma)*sympy.cos(psi_sigma)
e_psi = psi-psi_sigma

s = 5*(sympy.pi/2-psi_sigma)

jac = sympy.simplify(Matrix([[v], [e_y], [0]]).jacobian([x, y, psi, v, delta]))
print(jac)



# y_sigma = sympy.Symbol('y_sigma')
# psi_sigma = sympy.Symbol('psi_sigma')

# ey = -(x-x_sigma)*sin(psi_sigma)

# expr_1 = x1**2+x2
# poly_1 = sympy.poly(expr_1, x1, x2)

# print Matrix([[poly_1.diff(x1)],[poly_1.diff(x2)]])