using DFTK: Mixing
using AtomsBase
using DFTK
using DoubleFloats
using GenericLinearAlgebra
using MPI
using Statistics
using LineSearches

include("term_dualmap.jl")
include("term_external_hartree.jl")
include("term_atomic_local.jl")

struct InversionVxc end   # Invert with -½Δ + Vext + Vₕ
function basis_inversion(::InversionVxc, system::AbstractSystem, ρref_generator, ε::Number, T;
                         kwargs_model, kwargs_basis)
    model = model_atomic(system;
                         model_name="InversionVxc",
                         extra_terms=[Hartree(), DualMap(ρref_generator, ε; Tpotential=T)],
                         kwargs_model...)
    PlaneWaveBasis(model; kwargs_basis...)
end

function extract_reference_potential(::InversionVxc, ham_reference::Hamiltonian, ρref)
    @assert ham_reference.basis.model.n_spin_components == 1
    i_term = only(i for (i, t) in enumerate(ham_reference.basis.model.term_types) if t isa Xc)
    pot = ham_reference[1].operators[i_term].potential
    reshape(pot, ham_reference.basis.fft_size..., 1)
end

struct InversionVxcMod 
    α::Float64  # Modulator for the Hartree term
    β::Float64  # Modulator for the Hartree potential from ρref
    γ::Float64  # Modulator for the atomic local potential
end

# Invert with -½Δ + α v_H[ρ] + β v_H[ρref] + γ v_ext_local 
function basis_inversion(inv::InversionVxcMod,
                         system::AbstractSystem,
                         ρref_generator, ε::Number, T;
                         kwargs_model, kwargs_basis)
    @info "Guiding functional is T(ρ) + $(inv.α) v_H[ρ] + $(inv.β) v_H[ρref] + $(inv.γ) v_ext_loc"
    terms = [   
        Kinetic(; blowup=BlowupIdentity()),                     # Same as in model_atomic
        Ewald(), AtomicNonlocal(), PspCorrection(),             # Standard terms
        AtomicLocal(; scaling_factor=inv.γ),                    # Moulated local atomic potential
        Hartree(inv.α),                                         # Modulated Hartree term
        ExternalHartree(ρref_generator; scaling_factor=inv.β),  # Modulated external Hartree
        DualMap(ρref_generator, ε; Tpotential=T)                # Dual map term
    ]
    model = Model(system; model_name="InversionVxcMod",
                        terms, kwargs_model...)
    
    PlaneWaveBasis(model; kwargs_basis...)
end

function extract_reference_potential(inv::InversionVxcMod, ham_reference::Hamiltonian, ρref)
    @assert ham_reference.basis.model.n_spin_components == 1
    xc_term             = only(i for (i, t) in enumerate(ham_reference.basis.model.term_types) if t isa Xc)
    hartree_term        = only(i for (i, t) in enumerate(ham_reference.basis.model.term_types) if t isa Hartree)
    atomic_loc_term     = only(i for (i, t) in enumerate(ham_reference.basis.model.term_types) if t isa DFTK.AtomicLocal)

    xc_pot              = ham_reference[1].operators[xc_term].potential         # reference xc potential
    hartree_pot         = ham_reference[1].operators[hartree_term].potential    # reference hartree potential
    atomic_local_pot    = ham_reference[1].operators[atomic_loc_term].potential # reference atomic local potential

    """
    To compensate for the modulators in the guiding functional:
        vref = v_xc + (1 - α - β) v_H + (1 - γ) v_ext_local

    If α = β = γ = 0:       vref = v_xc + v_H + v_ext_local (i.e. the KS potential)
            -½Δ + v_KS
    If α = β = 0, γ = 1:    vref = v_Hxc (i.e. Hartree exchange-correlation potential)
            -½Δ + v_Hxc
    If α = 1, β = 0, γ = 1: vref = v_xc
            -½Δ + v_Hxc + v_ext
    If α = 0, β = 1, γ = 1: vref = v_xc
            -½Δ + v_Hxc + v_ext 
    If α = 1 - 1 / n_electrons, β = 0, γ = 1:
                The Fermi-Amaldi case
    """
    pot = (xc_pot 
        + (1 - inv.α - inv.β) .* hartree_pot 
        + (1 - inv.γ) .* atomic_local_pot) 
    reshape(pot, ham_reference.basis.fft_size..., 1)
end

struct InversionVks end   # Invert with -½Δ + nonlocal V
function basis_inversion(::InversionVks, system::AbstractSystem, ρref_generator, ε::Number, T;
                         kwargs_model, kwargs_basis)
    # model_atomic without AtomicLocal()
    terms = [Kinetic(), AtomicNonlocal(), Ewald(), PspCorrection(),
             DualMap(ρref_generator, ε; Tpotential=T)]
    model = Model(system; model_name="InversionVks",
                          terms,
                          kwargs_model...)
    PlaneWaveBasis(model; kwargs_basis...)
end
function extract_reference_potential(::InversionVks, ham_reference::Hamiltonian, ρref)
    # The basis_inversion call above assumes no temperature
    @assert iszero(ham_reference.basis.model.temperature)
    DFTK.total_local_potential(ham_reference)
end

struct InversionVfa end   # Invert with -½Δ + Vext + (1 - 1/N) Vₕ
function basis_inversion(::InversionVfa, system::AbstractSystem, ρref_generator, ε::Number, T;
                         kwargs_model, kwargs_basis)
    n_electrons = DFTK.n_electrons_from_atoms(DFTK.parse_system(system).atoms)
    scaling_factor = (1 - 1 / n_electrons)
    model = model_atomic(system;
                         model_name="InversionVfa",
                         extra_terms=[Hartree(; scaling_factor),
                                      DualMap(ρref_generator, ε; Tpotential=T)],
                         kwargs_model...)
    PlaneWaveBasis(model; kwargs_basis...)
end
function extract_reference_potential(::InversionVfa, ham_reference::Hamiltonian, ρref)
    @assert ham_reference.basis.model.n_spin_components == 1
    i_xc    = only(i for (i, t) in enumerate(ham_reference.basis.model.term_types) if t isa Xc)
    i_har   = only(i for (i, t) in enumerate(ham_reference.basis.model.term_types) if t isa Hartree)
    pot_xc  = ham_reference[1].operators[i_xc].potential
    pot_har = ham_reference[1].operators[i_har].potential

    n_electrons = ham_reference.basis.model.n_electrons
    pot = pot_xc + 1/n_electrons * pot_har
    reshape(pot, ham_reference.basis.fft_size..., 1)
end

struct InversionVpbe end  # Invert with -½Δ + Vext + Vpbe (i.e. the X and the C from PBE)
function basis_inversion(::InversionVpbe, system::AbstractSystem, ρref_generator, ε::Number, T;
                         kwargs_model, kwargs_basis)
    model = model_atomic(system;
                         model_name="InversionVpbe",
                         extra_terms=[Xc([:gga_x_pbe, :gga_c_pbe]), DualMap(ρref_generator, ε; Tpotential=T)],
                         kwargs_model...)
    PlaneWaveBasis(model; kwargs_basis...)
end
function extract_reference_potential(::InversionVpbe, ham_reference::Hamiltonian, ρref)
    @assert ham_reference.basis.model.n_spin_components == 1
    i_term = only(i for (i, t) in enumerate(ham_reference.basis.model.term_types) if t isa Hartree)
    pot = ham_reference[1].operators[i_term].potential
    reshape(pot, ham_reference.basis.fft_size..., 1)
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


"""
This function diagonalises the KS Hamiltonian defined in basis at the density ρref
and returns the result.
"""
function rediagonalise_hamiltonian(basis::PlaneWaveBasis{T}, ρref, basis_ref::PlaneWaveBasis{T}) where {T}
    ρref_basis = interpolate_density(ρref, basis_ref, basis)  # ρref in the basis
    ψ = nothing
    occupation = nothing
    _, ham  = energy_hamiltonian(basis, ψ, occupation; ρ=T.(ρref_basis))
    res = DFTK.next_density(ham; ψ, occupation)
    merge(res, (; ρ=res.ρout))
end

function ρref_by_interpolation(T, ρref, basis_ref::PlaneWaveBasis)
    function callback(basis_inversion::PlaneWaveBasis)
        DFTK.interpolate_density(ρref, basis_ref, basis_inversion)
    end
end
function ρref_by_interpolation(ρref::AbstractArray{T}, basis_ref::PlaneWaveBasis) where {T}
    ρref_by_interpolation(T, ρref, basis_ref)
end


function fix_scfres_for_save(scfres; nb::Union{Int,Nothing}=nothing)
    """
    Normalize eigenvalues and occupations in an scfres NamedTuple so that
    - `eigenvalues :: Vector{Vector{Float64}}`
    - `occupation :: Vector{Vector{Float64}}`
    - All inner vectors have equal length `nb`

    If you only have occupied bands, it pads eigenvalues with NaN and occupations with 0.0.
    """
    # Determine nb if not provided:
    nb′ = something(nb, try # Try to infer from ψ if present and rectangular:
                            minimum(size.(scfres.ψ, 2))  # conservative choice if ψ rectangular
                        catch
                            maximum(get.(Ref(length), scfres.eigenvalues, 0))
                        end)

    # Coerce eigenvalues
    eigs_block = Vector{Vector{Float64}}(undef, length(scfres.eigenvalues))
    for k in eachindex(scfres.eigenvalues)
        v = Float64.(scfres.eigenvalues[k])
        if length(v) < nb′
            v = vcat(v, fill(NaN, nb′ - length(v)))
        elseif length(v) > nb′
            v = v[1:nb′]
        end
        eigs_block[k] = v
    end

    # Coerce occupations
    occ_block = Vector{Vector{Float64}}(undef, length(scfres.occupation))
    for k in eachindex(scfres.occupation)
        v = Float64.(scfres.occupation[k])
        if length(v) < nb′
            v = vcat(v, zeros(nb′ - length(v)))
        elseif length(v) > nb′
            v = v[1:nb′]
        end
        occ_block[k] = v
    end

    # Return a new NamedTuple with fixed fields
    return merge(scfres, (; eigenvalues = eigs_block, occupation = occ_block))
end

function DmConvergenceInversion(ρref, ε, δ, ρtol)
    n_satisfied = 0
    n_unstable  = 0
    function callback(info)
        # Rationale is that the change is smaller than our error to the
        # ground state density so no point continuing the iterations ...
        #
        # Also we use the duality mapping to ensure we have a convergence
        # to δ in the potential.
        hm1_norm_ρdiff = norm_hm1(info.ham.basis, info.ρout - ρref)
        hm1_norm_Δρ    = norm_hm1(info.ham.basis, info.ρout - info.ρin)
        l2_norm_Δρ     = norm_l2(info.ham.basis,  info.ρout - info.ρin)
        is_satisfied   = hm1_norm_Δρ < hm1_norm_ρdiff && hm1_norm_Δρ < ε * δ
        is_unstable    = l2_norm_Δρ  < ρtol / ε

        is_satisfied  && (n_satisfied += 1)
        !is_satisfied && (n_satisfied  = 0)
        is_unstable   && (n_unstable  += 1)
        !is_unstable  && (n_unstable   = 0)

        n_satisfied = MPI.bcast(n_satisfied, 0, MPI.COMM_WORLD)
        n_unstable  = MPI.bcast(n_unstable,  0, MPI.COMM_WORLD)
        if mpi_master() && n_unstable > 1
            println("    ", "Stopping to avoid numerical instabilities.")
        end
        return n_satisfied > 1 || n_unstable > 1
    end
end
function KSinversionCallback(ρref, vref, ε)
    function callback(info)
        if info.stage == :iterate
            hm1_norm_ρdiff = norm_hm1(info.ham.basis, info.ρout - ρref)
            hm1_norm_Δρ    = norm_hm1(info.ham.basis, info.ρout - info.ρin)

            if !isnothing(vref)
                error_V = extract_inverted_potential(info.ham) - vref
                error_V_h1 = norm_h1(info.ham.basis, error_V .- mean(error_V))
            end

            if mpi_master()
                print("    ", "‖ρε - ρref‖ = $hm1_norm_ρdiff ‖Δρ‖ = $hm1_norm_Δρ")
                !isnothing(vref) && print("  ‖vε - vref‖ = $error_V_h1")
                
                if hasproperty(info, :optim_state)
                    print("  α = $(round(info.optim_state.alpha; sigdigits=3))")
                end
                println("")
            end
        end
        info
    end
end

function kohn_sham_inversion(system::AbstractSystem, ρref_generator;
                             method=InversionVxc(),
                             kwargs_model, kwargs_basis,
                             δ=1e-2, εs=exp10.(0:-0.25:-3),
                             ψ=nothing, verbose=false, ρtol=0,
                             retry::Int64 = 0,  # 0: retry without preconditioner 
                                                # 1: retry without preconditioner, 
                                                #       then if needed with Newton 
                                                # 2: retry with Newton
                                                # other values: no retries
                             verbose_unperturbed_data=nothing,
                             maxiter=300, vref=nothing, use_scf=false,
                             Tinversion=Double64,
                             kwargs...)
    # TODO Notes:
    #      - Using Double64 was nice for testing, but experiments show
    #        that running in Float64 does not lead to too many additional issues
    #        (i.e. it leads to issues, but at around the same ε values as the
    #        limit of numerical accuracy in ρref is anyway reached)
    
    hm1_norm_ρdiff = 0
    scfres = nothing
    ρs = []
    vs = []
    energies = []
    for ε in εs
        if mpi_master()
            println()
            println("# --- ε=$ε")
            println()
    	end
        scfres = nothing
        GC.gc() 
        basis  = basis_inversion(method, system, ρref_generator, ε, Tinversion;
                                kwargs_model, kwargs_basis)
        T = eltype(basis)
        ρref = T.(only(t for t in basis.terms if t isa TermDualMap).ρref)
        callback = (verbose ? KSinversionCallback(ρref, vref, ε) ∘ DFTK.ScfDefaultCallback()
                            : DFTK.ScfDefaultCallback())

        # Strip extra bands if there are any
        n_bands = let
            filled_occ = DFTK.filled_occupation(basis.model)
            div(basis.model.n_electrons, basis.model.n_spin_components * filled_occ, RoundUp)
        end
        if !isnothing(ψ) && size(ψ[1], 2) > n_bands
            ψ = [@view ψk[:, 1:n_bands] for ψk in ψ]
        end

        extra_args = (; )
        if use_scf          # If the SCF is never used, the remove it to clean up
            damping    = min(0.8, 0.8^-log10(ε / hm1_norm_ρdiff))
            algorithm  = self_consistent_field
            extra_args = (; diagtolalg=AdaptiveDiagtol(; diagtol_first=min(δ, ε)),
                            mixing=InversionMixing(; ε), damping)
        else
            algorithm  = direct_minimization
            extra_args = (; linesearch=LineSearches.MoreThuente(),
                            g_abstol = 1e-8)
                            # x_abstol = 1e-8,
                            # x_reltol = 1e-6,
                            # f_abstol = 1e-12,
                            # f_reltol = 1e-8)
        end
        scfres = algorithm(basis; ψ, callback, maxiter, kwargs..., extra_args...,
                           is_converged=DmConvergenceInversion(ρref, ε, δ, ρtol)) 
        if !use_scf && !scfres.converged && retry == 0
            println("Trying again without preconditioner ...")
            scfres = direct_minimization(basis; scfres.ψ, callback, maxiter, kwargs..., extra_args...,
                                         is_converged=DmConvergenceInversion(ρref, ε, δ, ρtol),
                                         prec_type=PreconditionerNone)
        elseif !use_scf && !scfres.converged && retry == 1
            println("Trying again without preconditioner ...")
            scfres_temp = direct_minimization(basis; scfres.ψ, callback, maxiter, kwargs..., extra_args...,
                                         is_converged=DmConvergenceInversion(ρref, ε, δ, ρtol),
                                         prec_type=PreconditionerNone)
            if scfres_temp.converged
                scfres = scfres_temp
            elseif norm_hm1(basis, scfres_temp.ρ - ρref) < norm_hm1(basis, scfres.ρ - ρref)
                println("Trying again with Newton's method ...")
                # Using the last ψ of failed DM as starting guess to given Newton a better chance
                scfres = newton(basis, scfres.ψ; tol=1e-8, maxiter=50,
                                is_converged=DmConvergenceInversion(ρref, ε, δ, ρtol),
                                callback=callback)
                # Newton method returns diffrently than SCF/DM, so we need to fix it for saving
                scfres = fix_scfres_for_save(scfres)
            else
                println("Trying again with Newton's method ...")
                # Using the last ψ of failed DM/SCF as starting guess to given Newton a better chance
                scfres = newton(basis, scfres.ψ; tol=1e-8, maxiter=50,
                                is_converged=DmConvergenceInversion(ρref, ε, δ, ρtol),
                                callback=callback)
                # Newton method returns diffrently than SCF/DM, so we need to fix it for saving
                scfres = fix_scfres_for_save(scfres)  
            end
        elseif !scfres.converged  && retry == 2
            println("Trying again with Newton's method ...")
            # Using the last ψ of failed DM as starting guess to given Newton a better chance
            scfres = newton(basis, scfres.ψ; tol=1e-8, maxiter=50,
                            is_converged=DmConvergenceInversion(ρref, ε, δ, ρtol),
                            callback=callback)
            scfres = fix_scfres_for_save(scfres)  
        end

        ψ = copy(scfres.ψ)
        push!(vs, copy(extract_inverted_potential(scfres.ham)))
        push!(ρs, copy(scfres.ρ))
        push!(energies, DFTK.todict(scfres.energies))

        error_V_h1  = nothing
        hm1_norm_ρdiff = norm_hm1(basis, scfres.ρ - ρref)
        if !isnothing(vref)
            error_V = vs[end] - vref
            error_V_h1 = norm_h1(basis, error_V .- mean(error_V))
        end
        if !isnothing(verbose_unperturbed_data)
            εmatch, i_unpert = findmin(εi -> abs(εi - ε), verbose_unperturbed_data.εs)
            @assert εmatch < 1e-10
            unper_ρ = verbose_unperturbed_data.ρs[i_unpert]
            Δρ     = ρref - verbose_unperturbed_data.ρref
            Δρ_hm1 = norm_hm1(scfres.ham.basis, Δρ)
            Qε = norm_hm1(scfres.ham.basis, unper_ρ .- scfres.ρ) / Δρ_hm1
        end

        if mpi_master()
            println()
            print("#-- ‖ρε - ρref‖ = $hm1_norm_ρdiff vs. $(δ*ε) = δ*ε")
            !isnothing(vref) && print(" ‖vε - vref‖ = $error_V_h1")
            !isnothing(verbose_unperturbed_data) && print("  Qε=$Qε")
            println()
            println()
        end
    end

    (; scfres, energies, εs, vs, ρs)
end

function norm_sobolev(basis::PlaneWaveBasis, x, s::Real)
    if ndims(x) > 3
        x = total_density(x)
    end
    x_fourier = fft(basis, x)
    norm_weights = (1 .+ DFTK.norm2.(G_vectors_cart(basis))) .^ s
    sqrt(sum(abs, x_fourier .* x_fourier .* norm_weights))
end
norm_h1(basis, x)    = norm_sobolev(basis, x, 1)
norm_hm1(basis, x)   = norm_sobolev(basis, x, -1)
norm_l2(basis, x)    = norm(x) * sqrt(basis.dvol)
norm_linf(basis, x)  = maximum(abs, x)

function extract_inverted_potential(ham::Hamiltonian)
    model = ham.basis.model
    @assert model.n_spin_components == 1
    if startswith(model.model_name, "Inversion")
        i_dual = only(i for (i, t) in enumerate(model.term_types) if t isa DualMap)
        return ham[1].operators[i_dual].potential
    else
        error("No inversion hamiltonian !")
    end
end

@kwdef struct InversionMixing <: Mixing
    kTF::Real = 0.8
    ε::Real   = 1.0
end
function DFTK.mix_density(mixing::InversionMixing, basis::PlaneWaveBasis, δF;
                          kwargs...)
    @assert basis.model.n_spin_components == 1
    T   = eltype(δF)
    G²  = DFTK.norm2.(G_vectors_cart(basis))
    kTF = T.(mixing.kTF)
    ε   = T.(mixing.ε)

    # Extra term from Kohn-Sham inversion
    inversion_term = kTF.^2 ./ ε * G² ./ (1 .+ G²)

    δFtot_fourier = fft(basis, total_density(δF))
    δρtot_fourier = δFtot_fourier .* G² ./ (kTF.^2 .+ G² .+ inversion_term)
    DFTK.enforce_real!(δρtot_fourier, basis)
    δρtot = irfft(basis, δρtot_fourier)

    # Copy DC component, otherwise it never gets updated
    δρtot .+= mean(total_density(δF)) .- mean(δρtot)
    ρ_from_total_and_spin(δρtot, nothing)
end
