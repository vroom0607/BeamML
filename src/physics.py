#max deflection and stress of cantilever beam

def max_deflection(force, length, base, height, E):
    """
    force: load at end of cantilever beam
    length: length of beam
    base: width of beam/cross-sectional area
    height: height of beam/cross-sectional area
    E: young's modulus of elasticity, treated as constant based on beam material
    """

    inertia = (base * height ** 3) / 12 #moment of inertia for cross-section
    return (force * length ** 3) / (3 * E * inertia)

def max_stress(force, length, base, height):
    """
    force: load at end of cantilever beam
    length: length of beam
    base: width of beam/cross-sectional area
    height: height of beam/cross-sectional area
    """

    y = height / 2
    inertia = (base * height ** 3) / 12 #moment of inertia for cross-section
    return (y * force * length) / inertia