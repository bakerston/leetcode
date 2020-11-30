#print((statistics.multimode(a)))
#print(list(map(adic.get,a)))
import numpy as np
import matplotlib.pyplot as plt
#initial 
Nx,Nt,L,T,a=20,10,20,10,0.3
I=np.zeros(Nx+1)
I[11]=10
ans=[]
x = np.linspace(0, L, Nx+1)    
dx = x[1] - x[0]
t = np.linspace(0, T, Nt+1)    
dt = t[1] - t[0]
F = a*dt/dx**2
u   = np.zeros(Nx+1)           
u_1 = np.zeros(Nx+1)          
for i in range(0, Nx+1):
    u_1[i] = I[i]

for n in range(0, Nt):
    for i in range(1, Nx):
        u[i] = u_1[i] + F*(u_1[i-1] - 2*u_1[i] + u_1[i+1])
    u[0] = 0;  u[Nx] = 0
    u_1[:]= u
    print(u_1)
#plot
plt.title("1-D diffusion")
x_list = [x for x in range(Nx+1)]

a=[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 3, 4, 3, 0, 0, 0, 0, 0, 0, 0, 0]
b=[0,  0,  0,  0,  0,  0,  0,  0,  0,  0.9, 2.4, 3.4, 2.4, 0.9, 0,  0,  0,  0,
 0,  0,  0 ]
c=[0,0,0,   0,   0,   0,   0,   0,   0.27, 1.08, 2.25, 2.8,  2.25, 1.08,
 0.27, 0,   0,   0,   0,   0,   0  ]
d=[0, 0,0,0,0,0,    0,  0.081, 0.432, 1.188, 2.064, 2.47,
 2.064, 1.188, 0.432, 0.081,0,0,0,0,0]
f=[0.00000000e+00, 5.90490000e-05, 7.87320000e-04, 5.31441000e-03,
 2.38820400e-02, 7.96396050e-02, 2.08622304e-01, 4.44174840e-01,
 7.85466720e-01, 1.17003933e+00, 1.48124136e+00, 1.60154604e+00,
 1.48124136e+00, 1.17003933e+00, 7.85466720e-01, 4.44174840e-01,
 2.08622304e-01, 7.96396050e-02, 2.38820400e-02, 5.25536100e-03,
 0.00000000e+00]
plt.plot(x_list,a, label='1s')
plt.plot(x_list,b,  label='2s')
plt.plot(x_list,c,  label='3s')
plt.plot(x_list,d,  label='4s')
plt.plot(x_list,f,  label='10s')
plt.legend() # 显示图例

plt.xlabel('x')
plt.ylabel('C')
plt.show()
"""
x_list = [x for x in range(Nx+1)]
y_list = [x for x in u_1]
plt.figure('Line fig')
ax = plt.gca()
ax.set_xlabel('x')
ax.set_ylabel('y')
ax.plot(x_list, y_list, color='r', linewidth=1, alpha=0.6)
plt.show()
"""    