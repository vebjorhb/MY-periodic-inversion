using DFTK
using DoubleFloats
using LinearAlgebra
using LineSearches
using AtomsIO
using JLD2
using MPI
using Suppressor

include("term_dualmap.jl")
include("GP_proximal_point_algorithm.jl")

function load_reference(prefix)
    # Load the reference calculation from a JLD2 file
    jldfile = prefix * ".jld2"
    @assert isfile(jldfile) "Reference file $jldfile does not exist."
    @info "Loading reference calculation from $jldfile"
    
    # Load the JLD2 file and return the result
    scfres = load_scfres(file; skip_hamiltonian=true)
    (;basi)
end


function KSinversionCallback(ρref, ε; vref=nothing)
        function callback(info)
            if info.stage == :iterate
                hm1_norm_ρdiff = norm_hm1(info.ham.basis, info.ρout - ρref)
                hm1_norm_Δρ    = norm_hm1(info.ham.basis, info.ρout - info.ρin)
            end
            if mpi_master()
                if info.stage == :finalize
                    hm1_norm_ρdiff = norm_hm1(info.ham.basis, info.ρ - ρref)
                end
            end
            info
        end
    end

function DmConvergenceInversion(ρref, ε, δ; ρtol=0)
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



function kohn_sham_inversion(ρref;  εs=10.0 .^ ( 0:-1:-9), λs = 1e-2,
                                    C=1, α=2, a=10, Ecut=500, n_electrons=1, 
                                    pot::Function = x -> (x - a/2)^2,
                                    verbose=false, vref=nothing,
                                    δ=1e-2, ρtol=0, 
                                    tol=1e-12, ftol=1e-8,
                                    maxiter=10_000,
                                    method=:direct_minimization)
    # We assume that the reference density ρref is computed using the same Ecut as passed here. 
    # Set up the model and basis
    if mpi_master()
        println()
        DFTK.versioninfo()
        println()
    end
    DFTK.reset_timer!(DFTK.timer)

    lattice = a .* [[1 0 0.]; [0 0 0]; [0 0 0]]
    if vref === nothing
        terms = [Kinetic(),
                 ExternalFromReal(r -> pot(r[1])),
                 LocalNonlinearityDC(ρ -> C * ρ^α),
                ] 
        model = Model(lattice; n_electrons, terms, spin_polarization=:spinless)
        refbasis = PlaneWaveBasis(model, Ecut=Ecut, kgrid=(1, 1, 1))
        vref = extract_GPE_potential(refbasis, ρref)
    end

    scfres         = nothing
    ψguess         = nothing
    ρs             = []
    vs             = []
    energies       = []
    hm1_norm_ρdiff = []
    ρ_hm1_norm_diff = 0
    converged      = 0
    info           = []
   
    @info "Running Kohn-Sham inversion with Ecut=$Ecut, n_electrons=$n_electrons, C=$C, α=$α, a=$a"
    for ε in εs
        @info "Running Kohn-Sham inversion for ε=$(round(ε, sigdigits=3))"
        terms = [Kinetic(),
                 ExternalFromReal(r -> pot(r[1])),
                 DualMap(basis -> ρref, ε; Tpotential=Float64),
                ]
        model = Model(lattice; n_electrons, terms, spin_polarization=:spinless)
        basis = PlaneWaveBasis(model, Ecut=Ecut, kgrid=(1, 1, 1))
        scfres = nothing
        callback = KSinversionCallback(ρref, ε; vref)
        extra_args = (; )
        if method == :direct_minimization
            @suppress_err (
            scfres = direct_minimization(basis;
                                            callback, 
                                            maxiter,
                                            x_reltol=ftol,
                                            f_reltol=ftol,
                                            linesearch= LineSearches.BackTracking(),
                                            alphaguess=InitialStatic(;alpha=1),
                                            ψ=ψguess,
                                            ));
            push!(info, (;  f_calls     = scfres.optim_res.f_calls,
                            g_calls     = scfres.optim_res.g_calls,
                            converged   = scfres.converged,
                            runtime_ns  = scfres.runtime_ns))
        elseif method == :self_consistent_field
            damping = min(0.8, ε / 2.0)
            println("Using damping factor: $damping")
            scfres = self_consistent_field(basis;   
                                        callback,
                                        tol,
                                        maxiter=1_000,
                                        diagtolalg = AdaptiveDiagtol(; diagtol_first=min(δ, ε)), 
                                        mixing = SimpleMixing(),
                                        ψ = ψguess,
                                        is_converged=DmConvergenceInversion(ρref, ε, δ; ρtol),
                                        solver=DFTK.scf_damping_solver(; damping)
                                        );
        elseif method == :newton
            if ψguess == nothing
                @suppress_err (
                scfres = direct_minimization(basis;
                                            callback, 
                                            maxiter,
                                            x_reltol=ftol,
                                            f_reltol=ftol,
                                            linesearch= LineSearches.BackTracking(),
                                            alphaguess=InitialStatic(;alpha=1),
                                            is_converged=DmConvergenceInversion(ρref, ε, δ; ρtol)
                                            ));
                ψguess = scfres.ψ
            end
            scfres = newton(basis, ψguess;
                                tol,
                                is_converged=DmConvergenceInversion(ρref, ε, δ; ρtol));
            push!(info, (;  converged   = scfres.converged,
                            runtime_ns  = scfres.runtime_ns))
        elseif method == :proximal_point_algorithm
            scfres = proximal_point_algorithm(basis, ε, λs;
                                            ρgs=ρref,
                                            ρ0 = (scfres == nothing ? nothing : ρlast),
                                            μ0 = (scfres == nothing ? 0.0     : scfres.μ0),
                                            tol, ftol, δ, ρtol,
                                            maxiter)
            ρlast = (scfres.converged ? scfres.ρ : nothing)
            push!(info, (;  f_calls     = scfres.info.f_calls,
                            n_iter      = scfres.info.n_iter,
                            converged   = scfres.info.converged,
                            runtime_ns  = scfres.runtime_ns))
        elseif method == :pde_solver
            scfres = proximal_point(basis, ε;
                                    ρgs=ρref,
                                    ρ0 = (scfres == nothing ? nothing : scfres.ρ),
                                    μ0 = (scfres == nothing ? 0.0     : scfres.μ0),
                                    ftol,
                                    maxiter)
            push!(info, (;  f_calls     = scfres.f_calls,
                            converged   = scfres.converged,
                            runtime_ns  = scfres.runtime_ns))

        else
            error("Unknown method: $method")
        end
       
        ψguess = scfres.ψ 
        push!(vs, apply_J(basis, scfres.ρ - ρref, ε))
        push!(ρs, scfres.ρ)
        ρ_hm1_norm_diff = norm_hm1(basis, scfres.ρ - ρref)
        push!(hm1_norm_ρdiff, ρ_hm1_norm_diff)
        converged += (scfres.converged ? 1 : 0)
    end
    @info "Kohn-Sham inversion completed with $(round(converged/length(εs) * 100, sigdigits=3)) % reaching convergence cirterion."    
    
    mpi_master() && println(DFTK.timer)
    (; scfres, energies, εs, vs, ρs, ρref, hm1_norm_ρdiff, basis=scfres.basis, vref,info)
end
