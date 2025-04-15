# Lid-Driven Cavity Flow Simulation

This project numerically solves the **2D lid-driven cavity flow** using the **finite difference method**, with **RK3/RK4 (Runge-Kutta 3rd/4th order)** for time integration and a **projection method** to enforce incompressibility through the pressure Poisson equation.

## Overview

The lid-driven cavity problem is a classic benchmark problem in computational fluid dynamics (CFD). It involves a square cavity where the top lid moves with a constant velocity, while all other walls remain stationary. The goal is to solve the incompressible Navier-Stokes equations to obtain the velocity and pressure fields.

### Key Features

- Finite difference discretization
- RK3/RK4 for time integration of intermediate velocities
- Projection method for incompressible flow
- Numba-accelerated performance
- NPZ binary file format for efficient data saving
- Visualization using Matplotlib
- Animated GIF creation of the velocity field

## Mathematical Formulation

### Equations Solved

- **Momentum Equation (Intermediate velocity)**
  
  $$\frac{\partial \vec{u}}{\partial t} + (\vec{u} \cdot \nabla)\vec{u} = -\nabla p + \nu \nabla^2 \vec{u}$$

- **Continuity Equation (Incompressibility Constraint)**

  $$\nabla \cdot \vec{u} = 0$$

- **Projection Method**
  
  Intermediate velocity is projected onto a divergence-free space by solving the **Pressure Poisson Equation**:

  $$\nabla^2 p = \frac{\rho}{\Delta t} \nabla \cdot \vec{u}^*$$

### Boundary Conditions

- Top wall: $(u = 1)$, $( v = 0)$ (moving lid)
- Other walls: $(u = 0)$, $(v = 0)$

## References

- Jofre, L., Abdellatif, A., & Oyarzun, G. (2023). *RHEA: an open-source Reproducible Hybrid-architecture flow solver Engineered for Academia*. "Journal of Open Source Software", 13 Gener 2023, vol. 8(81), núm. 4637, p. 1-6.
- Moin, P. *Fundamentals of Engineering Numerical Analysis*. "Cambridge University Press", 2010.
- Pope, S. B. *Turbulent Flows*. "Cambridge University Press", 2000.
- Vermeire, B. C., Pereira, C. A., & Karbasian, H. *Computational Fluid Dynamics: An Open-Source Approach*. "Concordia University", 2020.
- Hager, G., & Wellein, G. *Introduction to High Performance Computing for Scientists and Engineers*. "CRC Press", 2011.
- Manneville, P. *Instabilities, Chaos and Turbulence: An Introduction to Nonlinear Dynamics and Complex Systems*. "Imperial College Press", 2004.