# Select potentials to be passed to the Gross-Pitaevskii calculation

function harmodic(x; a, v0=1.0)
    # Harmonic oscillator potential
    -v0 * (x - a/2)^2
end

function quartic(x; a, v0=1.0)
    # Quartic potential
    -v0 * (x - a/2)^4
end

function absfunc(x; a, v0=1.0)
    # Absolute value potential
    -v0 * abs(x - a/2)
end

function optical_lattice(x; a, v0=1.0, r=2)
    if r % 2 != 0
        error("The exponent r must be an even integer.")
    end
    # Optical lattice potential
    -v0 * sin(π * x / a)^r
end