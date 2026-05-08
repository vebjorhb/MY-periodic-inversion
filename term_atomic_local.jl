using DFTK: TermLocalPotential, compute_local_potential, forces_local, derivative_wrt_αs, multiply_ψ_by_blochwave
"""
    Scaling of the atomic local potential:
        v_loc^λ (r) = λ v_loc (r)
"""
struct TermAtomicLocal{AT} <: TermLocalPotential
    potential_values::AT
    scaling_factor::Real
end

struct AtomicLocal
    scaling_factor::Real
end
AtomicLocal(; scaling_factor=1) = AtomicLocal(scaling_factor)

function (atomic::AtomicLocal)(basis::PlaneWaveBasis{T}) where {T}
    TermAtomicLocal(atomic.scaling_factor .* compute_local_potential(basis), atomic.scaling_factor)
end

function compute_forces(term::TermAtomicLocal, basis::PlaneWaveBasis{T}, ψ, occupation;
                        ρ, kwargs...) where {T}
    S = promote_type(T, real(eltype(ψ[1])))
    term.scaling_factor .* forces_local(S, basis, ρ, zero(Vec3{T}))
end

@views function compute_dynmat(term::TermAtomicLocal, basis::PlaneWaveBasis{T}, ψ, occupation;
                               ρ, δρs, q=zero(Vec3{T}), kwargs...) where {T}
    S = complex(T)
    model = basis.model
    positions = model.positions
    n_atoms = length(positions)
    n_dim = model.n_dim

    dynmat_δH = zeros(S, 3, n_atoms, 3, n_atoms)
    for s = 1:n_atoms, α = 1:n_dim
        dynmat_δH[:, :, α, s] .-= stack(forces_local(S, basis, δρs[α, s], q))
    end
    dynmat_δH .*= term.scaling_factor

    dynmat_δ²H = zeros(S, 3, n_atoms, 3, n_atoms)
    ρ_fourier = fft(basis, total_density(ρ))
    δ²V_fourier = similar(ρ_fourier)
    for s = 1:n_atoms, α = 1:n_dim, β = 1:n_dim
        δ²V = derivative_wrt_αs(basis.model.positions, β, s) do positions_βs
            derivative_wrt_αs(positions_βs, α, s) do positions_βsαs
                compute_local_potential(basis; positions=positions_βsαs)
            end
        end
        dynmat_δ²H[β, s, α, s] += sum(conj(ρ_fourier) .* fft!(δ²V_fourier, basis, δ²V))
    end
    dynmat_δ²H .*= term.scaling_factor

    dynmat_δH + dynmat_δ²H
end

function compute_δHψ_αs(term::TermAtomicLocal, basis::PlaneWaveBasis, ψ, α, s, q)
    δV_αs = similar(ψ[1], basis.fft_size..., basis.model.n_spin_components)
    δV_αs .= derivative_wrt_αs(basis.model.positions, α, s) do positions_αs
        compute_local_potential(basis; q, positions=positions_αs)
    end
    term.scaling_factor .* multiply_ψ_by_blochwave(basis, ψ, δV_αs, q)
end
