using DFTK
using LinearAlgebra
"""
ExternalHartree: Implements

External Hartree energy is
            E(ρ) = ⟨v_H[ρref], ρ⟩ - ½ ⟨v_H[ρref], ρref⟩
where v_H[ρref](r) = ∑_G ((4π * scaling_factor) / |G|^2 ) * ρref_G e^{iG⋅r} for all G≠0.
The last term in is inclued such that E(ρref) = E_H(ρref) as implemented in DFTK.

The corresponding potential operator is
            V_H[ρref] = ∑_G ((4π * scaling_factor) / |G|^2 ) * ρref_G |G⟩⟨G| for all G≠0.
Note that the corresponding kernel operation is zero since the potential is independent of ρ.

Note that the floating-point type of ρref sets the precision at which
the external Hartree term is computed.
"""

struct ExternalHartree
    ρref_generator  
    Tpotential      # Precision for computing the potential
    scaling_factor
end
ExternalHartree(ρref_generator; Tpotential=Double64, scaling_factor=1.0) = ExternalHartree(ρref_generator, Tpotential, scaling_factor)
(term::ExternalHartree)(basis) = TermExternalHartree(basis, term.ρref_generator(basis),
                                    term.Tpotential; term.scaling_factor)

struct TermExternalHartree{Tref,Tarr,Tbfft} <: DFTK.TermNonlinear
    ρref_tot_fourier::Array{Complex{Tref}, 3}
    yukawa_coeffs::Array{Tref, 3}
    ρref::Tarr
    scaling_factor::Tref
    opBFFT::Tbfft
end
function TermExternalHartree(basis::PlaneWaveBasis{T}, ρref::AbstractArray, Tpotential; scaling_factor=1.0) where {T}
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
    yukawa_coeffs = Tref(4π * scaling_factor) ./ (DFTK.norm2.(G_vectors_cart(basis)))  
    yukawa_coeffs[1] = 0.0
    TermExternalHartree(ρref_tot_fourier, yukawa_coeffs, ρref, Tref(scaling_factor), opBFFT)
end

DFTK.@timing "ene_ops: external_Hartree" function DFTK.ene_ops(term::TermExternalHartree{Tref}, basis::PlaneWaveBasis{T},
                                                      ψ, occupation; ρ, kwargs...) where {T,Tref}
    cTref = complex(Tref)
    ρtot = total_density(ρ)
    ρtot_fourier  = cTref.(fft(basis, ρtot))
    pot_fourier   = term.yukawa_coeffs .* term.ρref_tot_fourier

    E = T(real(dot(pot_fourier, ρtot_fourier)) 
        - real(dot(term.ρref_tot_fourier, term.ρref_tot_fourier))/ 2)
        
    pot_real = T.(real(basis.fft_grid.ifft_normalization * (term.opBFFT * pot_fourier)))

    ops = [DFTK.RealSpaceMultiplication(basis, kpt, pot_real) for kpt in basis.kpoints]
    (; E, ops)
end

DFTK.apply_kernel(term::TermExternalHartree, basis::PlaneWaveBasis, δρ; kwargs...) = zero(δρ)
