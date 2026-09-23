# Beam Machine Learning Project

## Overview & goals

A PyTorch neural network that predicts cantilever beam stress and deflection using physical and geometric paramters.

Proof of concept for future plans of using neural networks as a surrogate for Finite Element Analysis.

## Physics

### Maximum Deflection

For a cantilever beam with a point load applied at the free end:

\[
\delta_{\max} = \frac{FL^3}{3EI}
\]

where the area moment of inertia for a rectangular cross-section is:

\[
I = \frac{bh^3}{12}
\]

### Maximum Bending Stress

The maximum bending stress occurs at the outer surface of the beam:

\[
\sigma_{\max} = \frac{Mc}{I}
\]

where:

\[
M = FL
\]

and:

\[
c = \frac{h}{2}
\]

Therefore:

\[
\sigma_{\max} = \frac{FL(h/2)}{I}
\]

## Machine Learning Model

This is a PyTorch neural network that uses 5 input features, 64 hidden units, and 2 output features.

Model_0 uses 2 linear layers, while Model_1 features an additional 2 nonlinear ReLU layers.

The model fits using the MSELoss() loss function (mean squared eror), and uses Adam for the optimizer.

The model trains for 400 epochs on CUDA, following the typical training loop.

## Data

The dataset (10,000) is created using  a Pandas DataFrame. The table includes:

Input:

- Material (aluminum or steel)
- Length (m)
- Width (m)
- Height (m)
- Force (N)
- Young's Modulus (GPa)

Output:

- Stress (MPa)
- Deflection (mm)

The data is fed into the model using Pandas DataLoaders. Before input, the data is standardized through the method:

\[
X_{\text{standardized}} = \frac{X - \mu_X}{\sigma_X}
\]

\[
y_{\text{standardized}} = \frac{y - \mu_y}{\sigma_y}
\]

The mean and standard deviation of input (X) and output (y) are saved as paramaters as a seperate PyTorch model.

The data is split into 80% for training and 20% for testing.

## Results

### Deflection

\[
R^2 > 0.99
\]

Average error: 0.5320 mm

### Stress

\[
R^2 > 0.99
\]

Average error: 0.6819 MPa

### Example Prediction

#### Actual

Deflection: 7.3124 mm
Stress: 31.6345 MPa

#### Predicted

Deflection: 7.2804 mm
Stress: 31.1238 MPa

## Limitations

- Data is generated from analytical beam equations.
- Model is only accurate within training domain.

## Future Work

- Generate FEM-based training data
- Support more complex geometries
- Predict additional quantities
- Investigate neural-network surrogate models for FEA