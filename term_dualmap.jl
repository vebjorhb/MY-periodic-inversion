using DFTK
using LinearAlgebra

"""
DualMap: Implements

1/ε ∫ [ρ(x) - ρref(x)] * [ρ(y) - ρref(y)] G(x-y) dx dy

where G(x, y) = 1 / |x-y| (the Coulomb kernel)
and the integral running both times over the unit cell.
The duality mapping corresponds to the canoncial map J: H^{-1}_{hom,per} → H^1_{hom,per}, 
between the homogenous periodic Sobolev spaces H^{-1}_{hom,per} and H^1_{hom,per}.

Note that the floating-point type of ρref sets the precision at which
the dual map term is computed.
"""
struct DualMap
    ρref_generator  # Reference density (on the same discretisation as basis)
    ε               # Regularisation parameter
    Tpotential      # Precision for computing the potential
    zero_DC
end
DualMap(ρref_generator, ε; Tpotential=Double64, zero_DC=true) = DualMap(ρref_generator, ε, Tpotential, zero_DC)
(term::DualMap)(basis) = TermDualMap(basis, term.ρref_generator(basis),
                                     term.ε, term.Tpotential; term.zero_DC)
Base.show(io::IO, term::DualMap) = print(io, "DualMap(ε=$(term.ε))")

struct TermDualMap{Tref,Tarr,Tbfft} <: DFTK.TermNonlinear
    ρref_tot_fourier::Array{Complex{Tref}, 3}
    ε::Tref
    yukawa_coeffs::Array{Tref, 3}
    ρref::Tarr
    opBFFT::Tbfft
end
function TermDualMap(basis::PlaneWaveBasis{T}, ρref::AbstractArray, ε, Tpotential; zero_DC=true) where {T}
    @assert size(ρref) == (basis.fft_size..., basis.model.n_spin_components)
    Tref = something(Tpotential, T)
    if T == Tref
        opFFT  = basis.fft_grid.opFFT
        opBFFT = basis.fft_grid.opBFFT
    else
        opFFT, opBFFT = let
            dummy = similar(G_vectors(basis), complex(Tref), basis.fft_size)
            (ipFFT, opFFT, ipBFFT, opBFFT) = DFTK.build_fft_plans!(dummy)
            opFFT, opBFFT
        end
    end
    ρref_tot_fourier = Tref(basis.fft_grid.fft_normalization) .* (opFFT * complex.(Tref.(total_density(ρref))))
    yukawa_coeffs = 1 ./Tref(ε) ./ (DFTK.norm2.(G_vectors_cart(basis)))  
    yukawa_coeffs[1] = 0.0
    TermDualMap(ρref_tot_fourier, Tref(ε), yukawa_coeffs, ρref, opBFFT)
end

DFTK.@timing "ene_ops: dualmap" function DFTK.ene_ops(term::TermDualMap{Tref}, basis::PlaneWaveBasis{T},
                                                      ψ, occupation; ρ, kwargs...) where {T,Tref}
    cTref = complex(Tref)
    ρtot = total_density(ρ)
    ρtot_fourier  = cTref.(fft(basis, ρtot))
    ρdiff_fourier = ρtot_fourier - term.ρref_tot_fourier
    pot_fourier   = term.yukawa_coeffs .* ρdiff_fourier

    E        = T(real(dot(pot_fourier, ρdiff_fourier))/2)
    pot_real = T.(real(basis.fft_grid.ifft_normalization * (term.opBFFT * pot_fourier)))

    ops = [DFTK.RealSpaceMultiplication(basis, kpt, pot_real) for kpt in basis.kpoints]
    (; E, ops)
end

function DFTK.apply_kernel(term::TermDualMap{Tref}, basis::DFTK.PlaneWaveBasis,
                          δρ::AbstractArray{Tδ}; kwargs...) where {Tref, Tδ}    
    δρtot = total_density(δρ)                        # 3D real array    
    cTref = Complex{Tref}    
    δρtot_F = cTref.(DFTK.fft(basis, δρtot))              # cast for opBFFT plan    
    δV_F    = term.yukawa_coeffs .* δρtot_F    
    δV_R3   = real(basis.fft_grid.ifft_normalization * (term.opBFFT * δV_F))  # 3D    
    
    # Return δV with same shape as δρ (broadcast to all spin components)    
    δV = zero(δρ)    
    δV .= Tδ.(δV_R3)                                      # broadcasts along spin dim    
end

function apply_J(basis, ρ, ε)
    # Apply dual map J without rescaling by mean
    term = TermDualMap(basis, zero(ρ), ε, Float64, zero_DC=false)
    ψ = nothing
    occupation = nothing
    (; ops) = DFTK.ene_ops(term, basis, ψ, occupation; ρ)

    @assert basis.model.n_spin_components == 1
    ops[1].potential
end
