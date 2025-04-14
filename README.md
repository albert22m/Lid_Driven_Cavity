# Lid-Driven Cavity Flow Simulation

This project numerically solves the **2D lid-driven cavity flow** using the **finite difference method**, with **RK4 (Runge-Kutta 4th order)** for time integration and a **projection method** to enforce incompressibility through the pressure Poisson equation.

## Overview

The lid-driven cavity problem is a classic benchmark problem in computational fluid dynamics (CFD). It involves a square cavity where the top lid moves with a constant velocity, while all other walls remain stationary. The goal is to solve the incompressible Navier-Stokes equations to obtain the velocity and pressure fields.

### Key Features

- Finite difference discretization
- RK4 for time integration of intermediate velocities
- Projection method for incompressible flow
- Numba-accelerated performance (`@njit`)
- CSV export of simulation frames
- Visualization using Matplotlib
- Animated GIF creation of the velocity field

## Mathematical Formulation

### Equations Solved

- **Momentum Equations (Intermediate velocity)**
  
  \[
  \frac{\partial \vec{u}}{\partial t} + (\vec{u} \cdot \nabla)\vec{u} = -\nabla p + \nu \nabla^2 \vec{u}
  \]

- **Incompressibility Constraint**

  \[
  \nabla \cdot \vec{u} = 0
  \]

- **Projection Method**
  
  Intermediate velocity is projected onto a divergence-free space by solving the **Pressure Poisson Equation**:

  \[
  \nabla^2 p = \frac{\rho}{\Delta t} \nabla \cdot \vec{u}^*
  \]

### Boundary Conditions

- Top wall: \( u = 1 \), \( v = 0 \) (moving lid)
- Other walls: \( u = 0 \), \( v = 0 \)

## How to Run

### Requirements

- Python 3.x
- Numpy
- Matplotlib
- Numba
- ImageIO

You can install the dependencies using:

```bash
pip install numpy matplotlib numba imageio
