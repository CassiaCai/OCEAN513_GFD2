#!/usr/bin/env python
# coding: utf-8

# # Numerical Model for Quasi-Geostrophic Equations
# ## Pseudo-spectral code to solve the QG equations
# 
# QG model is a great tool for exploring mesoscale ocean dynamics, including eddies and Rossby waves. 
# 
# Solutions to the QG equations exhibit highly nonlinear and chaotic behavior. As such, apart from a few idealized cases, it isn’t possible to solve QG equations analytically and we must resort to numerical integration. 

# ### QG equations in spectral space (pseudo-spectral method)
# #### QG equations for a single layer using the beta plane approximation
# 
# ![CassiaCai_HW5P1.jpg](attachment:de3ff8b6-7046-407b-a01f-10d5a3ea12cf.jpg)
# ![CassiaCai_HW5P12.jpg](attachment:074ab6d8-4840-416d-9bec-c56f7379559a.jpg)

# In[ ]:


import numpy as np
import matplotlib.pyplot as plt


# In[ ]:


# (a) Define numerical constants and the domain dimensions (using a square box of
# 1000 km by 1000 km). Use the number of grid boxes that are powers of 2 to
# speed up the calculation of the Fourier transform.
L = 1000e3
W = 1000e3
nx = 128
ny = 128

dx = L / nx
dy = W / ny

# x_range = (np.arange(1 / 2, nx + 1) / nx) * L - L / 2
# y_range = (np.arange(1 / 2, ny + 1) / ny) * W - W / 2
# x, y = np.meshgrid(x_range, y_range)

[x,y]=np.meshgrid(np.arange(1/2,nx,1)/nx*L-L/2,np.arange(1/2,ny,1)/ny*W-W/2);


# In[ ]:


# (b) The Fast Fourier Transform function outputs the transformed 2D array with the zonal
# and meridional wavevectors k, l defined as:
k0x=2*np.pi/L 
k0y=2*np.pi/W

k_range = np.concatenate((np.arange(0, nx // 2 + 1), np.arange(-nx // 2 + 1, 0))) * k0x
l_range = np.concatenate((np.arange(0, ny // 2 + 1), np.arange(-ny // 2 + 1, 0))) * k0y

[k, l] = np.meshgrid(k_range, l_range)
# we define the magnitude of the wave number as K and plot it as a 2D image with k,l being the axis
K = np.sqrt(k**2 + l**2)

plt.figure(figsize=(7,5))
im = plt.scatter(k,l,c=K)
cbar = plt.colorbar(im)
plt.xlabel('k', fontsize=15)
plt.ylabel('l', fontsize=15)
plt.xticks(fontsize=10,rotation=90)
plt.yticks(fontsize=10)
cbar.set_label('magnitude of wave number K', rotation=270, labelpad=15)
plt.show()


# In[ ]:


# (c) To avoid the formation of numerical instabilities in the spectral simulation,
# we  filter out scales of motion that approach the grid box size. In the
# finite-difference method, this is done by introducing the dissipation in the form
# of viscous terms that often leads to overly blurry solutions. 
# In spectral code, the dissipation of small-scale features is done in spectral space
# using predefined "filters" that are multiplying the spectral variables after each
# time step. These filters are simply matrixes with coefficients varying between one
# resolved scales (no damping) and zero at non-resolved scales. 

cphi = 0.69 * np.pi
wvx = np.sqrt(K**2) * dx

z = -18 * (wvx - cphi)**7
z_clipped = np.clip(z, -700, 700)  # Clip the values to avoid overflow

filtr = np.exp(z_clipped) * (wvx > cphi) + (wvx <= cphi)
filtr[np.isnan(filtr)] = 1

plt.figure(figsize=(7,5))
im = plt.scatter(k,l,c=filtr)
cbar = plt.colorbar(im)
plt.xlabel('k', fontsize=15)
plt.ylabel('l', fontsize=15)
plt.xticks(fontsize=10,rotation=90)
plt.yticks(fontsize=10)
cbar.set_label('filter value', rotation=270, labelpad=15)
plt.show()


# In[ ]:


# (d) To explore the Rossby wave propagation, define the initial conditions
# for potential vorticity with an example wave package as:
f0 = 1e-4
Ld = 40e3
q = 0.2 * f0 * np.exp(-(x / (L / 5))**2) * np.sin(12 * np.pi * x / L)

# PV inversion to find u and v
q_hat = np.fft.fft2(q)
psi_hat = -q_hat / ((k**2 + l**2)+(1/Ld**2))
u = np.fft.ifft2(-1j*l*psi_hat) 
v = np.fft.ifft2(1j*k*psi_hat)

step=8

plt.figure(figsize=(7,5))
im = plt.scatter(x,y,c=q)
cbar = plt.colorbar(im)
plt.xlabel('x', fontsize=15)
plt.ylabel('y', fontsize=15)
plt.xticks(fontsize=10,rotation=90)
plt.yticks(fontsize=10)
cbar.set_label('potential vorticity', rotation=270, labelpad=15)
plt.quiver(x[::step,::step], y[::step,::step], u[::step,::step], v[::step,::step],width=0.002,headwidth=10)
plt.show()


# In[ ]:


# (e) Implement the time-stepping loop to obtain the evolution of the PV 
# in spectral space. use the time step dt = 4 hrs
dt = 4*60*60 # hours
b = 1.6e-11


# In[ ]:


def compute_d_dtq_hat(q):
    q_hat = np.fft.fft2(q)
    psi_hat = -q_hat / ((k**2 + l**2)+(1/Ld**2))
    u = np.fft.ifft2(-1j*l*psi_hat)
    v = np.fft.ifft2(1j*k*psi_hat)
    du_qhat = np.fft.fft2(u*q)
    dv_qhat = np.fft.fft2(v*q)
    v_hat = np.fft.fft2(v)
    d_dtq_hat = -(1j*k*du_qhat + 1j*l*dv_qhat + b*1j*k*psi_hat)
    return d_dtq_hat


# In[ ]:


q_store = np.zeros((43830,128,128))

for timestep in range(43830):
    present_dqdt=compute_d_dtq_hat(q)

    if timestep==0:
        past_dqdt=present_dqdt
    q_hat =(q_hat + 1.5*dt*present_dqdt - 0.5*dt*past_dqdt)*filtr
    q = np.fft.ifft2(q_hat)
    # plt.scatter(x,y,c=q)
    # plt.show()
    q_store[timestep,:,:] = q
    past_dqdt=present_dqdt


# In[ ]:


# Run simulation for some time (about a few years of model time) 
# and plot the PV snapshots at some regular intervals (e.g. every 20 time steps) 
# to see the propagation of the idealized Rossby wave package. 

for i in np.arange(0,40000,4000):
    plt.scatter(x,y, c=q_store[i,:,:])
    plt.show()


# In[ ]:


# Based on the simulation results, what is the direction of the phase and group 
# speeds for short and long Rossby waves that have the initial conditions as shown below?
# Initial conditions for the short Rossby waves:
q = 0.2 * f0 * np.exp(-(x / (L / 5))**2) * np.sin(12 * np.pi * x / L)
# # Initial conditions for the long Rossby waves:
# q = 0.2 * f0 * np.exp(-(x / (L / 5))**2) * np.sin(4 * np.pi * x / L)
q_store = np.zeros((40000,128,128))

for timestep in range(40000):
    present_dqdt=compute_d_dtq_hat(q)

    if timestep==0:
        past_dqdt=present_dqdt
    q_hat =(q_hat + 1.5*dt*present_dqdt - 0.5*dt*past_dqdt)*filtr
    q = np.fft.ifft2(q_hat)
    # plt.scatter(x,y,c=q)
    # plt.show()
    q_store[timestep,:,:] = q
    past_dqdt=present_dqdt
    


# In[ ]:


for i in np.arange(0,40000,4000):
    plt.scatter(x,y, c=q_store[i,:,:])
    plt.show()


# #### Observations: 
# 
# The direction of the phase and group speeds for short Rossby waves with this initial conditions is to the right. 
# Initial conditions for the short Rossby waves:
# q=0.2*f0*exp(-(x/(L/5)).^2).*sin(12*pi*x/L); 
# 
# The direction of the phase and group speeds for long Rossby waves with this initial conditions is to the right. 
# Initial conditions for the long Rossby waves:
# q=0.2*f0*exp(-(x/(L/5)).^2).*sin(4*pi*x/L);  
