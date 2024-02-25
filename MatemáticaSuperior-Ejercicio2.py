import numpy as np
import matplotlib.pyplot as plt
import sympy as sp

# Elegimos crear un método para ecuaciones no lineales de 3 incógnitas basado en el método de la secante.

a=np.arange(-1,7,0.1)

#PRUEBAS INICIALES SOBRE EL FUNCIONAMIENTO DE LA SECANTE

def secanteUnaIncognita(x):
    r=[]
    for i in range(len(x)-1):
        r.append((x[i+1]-x[i]))
    return r

def secanteEnUnPunto(exp,punto,h):
    sec = []
    sec.append((exp.subs({x: punto[0]+h, y: punto[1], z: punto[2]}) -
                exp.subs({x: punto[0], y: punto[1], z: punto[2]}))/h)
    sec.append((exp.subs({x: punto[0], y: punto[1]+h, z: punto[2]}) -
                exp.subs({x: punto[0], y: punto[1], z: punto[2]}))/h)
    sec.append((exp.subs({x: punto[0], y: punto[1], z: punto[2]+h}) -
                exp.subs({x: punto[0], y: punto[1], z: punto[2]}))/h)
    return sec

def recta_tangente(x,p):
    c=[]
    for j in range(len(x)-1):
        c.append((secante(x)[p])*(j-p)+x[p])
    print(secante(x)[p])
    return c

# COMIENZO DEL EJERCICIO

def secante(exp,h):
    sec = []
    if x in exp.free_symbols:
        sec.append((exp.subs(x, x+h) -
                    exp)/h)
    else:
        sec.append(0)
    if y in exp.free_symbols:
        sec.append((exp.subs(y, y+h) -
                    exp/h))
    else:
        sec.append(0)
    if z in exp.free_symbols:
        sec.append((exp.subs(z, z+h) -
                    exp)/h)
    else:
        sec.append(0)

    s = sp.sympify(sec)
    return s

b=np.arange(-1,6.9,0.1)

x, y, z = sp.symbols('x y z')
f1 = x**2+2*y**2+sp.exp(x+y)+x*z-6.1718 #input("Ingresa la funcion 1: \n")
f2 =10*y+y*z #input("Ingresa la funcion 2: \n") x**2+y**2+z**2
f3 = sp.sin(x*z)+y**2+x-1.141 #input("Ingresa la funcion 3: \n") sp.exp(x)

#EJEMPLOS PARA PRUEBA

#x+y+z-3
#x**2+y**2+z**2-5
#sp.exp(x)+x*y+x*z-1
#Probar con 10 iteraciones y error 0.0000001
#Probar 25 iteraciones y error 0.0000000000000001
#Xinicial = sp.Matrix([0.1, 1.2, 2.5])

#x**2+x*z-9
#z**2+x*z-16
#y**2-x*z
#Xinicial = sp.Matrix([2,2,2])

#x**2+2*y**2+sp.exp(x+y)+x*z-6.1718
#10*y+y*z
#sp.sin(x*z)+y**2+x-1.141
#Xinicial = sp.Matrix([1.5, 0.5, 0.3])

#sp.exp(-x**2-y**2)-2*y
#x+2*y+z-8
#x**3+y-z
#Xinicial = sp.Matrix([0.5,0.5,0.5])

def Jacobiana(f1,f2,f3):
    j = sp.Matrix([[sp.diff(f1, x), sp.diff(f1, y), sp.diff(f1, z)],
                     [sp.diff(f2, x), sp.diff(f2, y), sp.diff(f2, z)],
                     [sp.diff(f3, x), sp.diff(f3, y), sp.diff(f3, z)]])
    return j

def MatrizSecante(f1,f2,f3):

    sec1 = secante(f1, 1)
    sec2 = secante(f2, 1)
    sec3 = secante(f3, 1)

    s = sp.Matrix([[sec1[0], sec1[1], sec1[2]],
                   [sec2[0], sec2[1], sec2[2]],
                   [sec3[0], sec3[1], sec3[2]]])
    return s

# print(Jacobiana(f1,f2,f3)," ",Jacobiana(f1,f2,f3).shape)

imax = int(input("Especifique la cantidad de iteraciones que se haran: \n"))
tol = float(input("Especifique la tolerancia que se tendra: \n"))
#ingreso = input("Especifique el punto inicial del algoritmo: \n")
#ingreso2 = sp.sympify(ingreso)
Xinicial = sp.Matrix([1.5, 0.5, 0.3])

#NOTA: el primer valor que probe era [0,0,0], pero la inversa tiene elementos fraccionarios y se anulaba el denominador.

A0 = sp.Matrix([f1, f2, f3])

J = Jacobiana(f1, f2, f3)

X0 = []

ejeX=[]
ejeY=[]

for i in range(imax):
    if i == 0:
        X0 = Xinicial

    #J.subs(x, X0[0])
    #J.subs(y, X0[1])
    #J.subs(z, X0[2])

    #A = A0

    #A.subs(x, X0[0])
    #A.subs(y, X0[1])
    #A.subs(z, X0[2])

    #print(J.subs({x: X0[0], y: X0[1], z: X0[2]}), '\n', A0.subs({x: X0[0], y: X0[1], z: X0[2]}))


    H0 = sp.linsolve((J.subs({x: X0[0], y: X0[1], z: X0[2]}), A0.subs({x: X0[0], y: X0[1], z: X0[2]})))

    H = sp.Matrix([H0.args[0]]).transpose()

    #print(H)

    Xn = X0 - H

    dX = Xn - X0

    ejeX.append(i)
    ejeY.append(dX.norm())

    print("El valor de X", i+1, " es: ", Xn)

    if i == (imax-1):
        print('Se han realizado todas las iteraciones.\n')

    if dX.norm() < tol:
        print("Convergió.\nLa diferencia entre las dos últimas iteraciones es de: ", dX.norm())
        break
    else:
        X0 = Xn


plt.plot(ejeX, ejeY)
plt.xlabel('iteraciones')
plt.ylabel('distancia euclidiana')
plt.title('dX - n')
plt.show()

S0 = []
i = 0
MS = MatrizSecante(f1, f2, f3)

ejeXsec=[]
ejeYsec=[]

for i in range(imax):
    if i == 0:
        S0 = Xinicial

    H0 = sp.linsolve((MS.subs({x: S0[0], y: S0[1], z: S0[2]}), A0.subs({x: S0[0], y: S0[1], z: S0[2]})))

    H = sp.Matrix([H0.args[0]]).transpose()

    #print(H)

    Sn = S0 - H

    dS = Sn - X0

    ejeXsec.append(i)
    ejeYsec.append(dS.norm())

    print("El valor de X", i+1, " es: ", Sn)

    if i == (imax-1):
        print('Se han realizado todas las iteraciones.\n')

    if dS.norm() < tol:
        print("Convergió.\nLa diferencia entre las dos últimas iteraciones es de: ", dS.norm())
        break
    else:
        S0 = Sn

plt.plot(ejeXsec, ejeYsec)
plt.xlabel('iteraciones')
plt.ylabel('distancia euclidiana')
plt.title('dS - n')
plt.show()



