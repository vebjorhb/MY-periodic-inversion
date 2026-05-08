using NLsolve

include("GP_reference.jl")
include("GP_potentials.jl")
include("term_dualmap.jl")

# Define additional dual map to include second regularisation term in Hamiltonian
struct DualMapλ
    ρref_generator  # Reference density (on the same discretisation as basis)
    ε               # Regularisation parameter
    Tpotential      # Precision for computing the potential
    zero_DC
end
DualMapλ(ρref_generator, ε; Tpotential=Float64, zero_DC=true) = DualMapλ(ρref_generator, ε, nothing, zero_DC)
(term::DualMapλ)(basis) = TermDualMap(basis, term.ρref_generator(basis),
                                     term.ε, term.Tpotential; term.zero_DC)
Base.show(io::IO, term::DualMapλ) = print(io, "DualMapλ(ε=$(term.ε))")

# Helper funvtions: 
#       discrete periodic derivatives
function derivative(f, N::Int, dx::Float64)
    [( -f[mod1(i+2,N)] + 8f[mod1(i+1,N)] - 8f[mod1(i-1,N)] + f[mod1(i-2,N)] ) / (12dx) for i in 1:N]
end
function second_derivative(f, N::Int, dx::Float64)
    [( -f[mod1(i+2,N)] + 16f[mod1(i+1,N)] - 30f[i] + 16f[mod1(i-1,N)] - f[mod1(i-2,N)] ) / (12dx^2) for i in 1:N]
end


function proximal_point(basis, ε; ρgs, ρ0=nothing, μ0=0.0, ftol=1e-8, maxiter=1000)
    # Compute proximal point by solving the PDE
    ## -1/4*(∂²ρ/∂x²)/ρ + 1/8*(∂ρ/∂x)^2/ρ^2 + V_ext + 1//ε J(ρ - ρ_gs)

    start_ns = time_ns()
    vext = extract_external_potential(basis)
    N = length(ρgs)

    x = r_vectors_cart(basis)
    dx = (x[2] - x[1])[1]

    _ρgs = copy(ρgs)[:, 1, 1]  # make sure it's a 1D array

    if ρ0 === nothing
        # if no initial guess, use ρgs
        ρ0 = copy(ρgs)[:, 1, 1]
    else
        # initial guess
        ρ0 = copy(ρ0)[:, 1, 1]
    end

    rhs_integral = sum(_ρgs)*dx     # ∫ ρgs dx
    function f!(F, u)
        ρ = @view u[1:N]
        μ = u[N+1]
        dρdx   = derivative(ρ, N, dx)
        d2ρdx2 = second_derivative(ρ, N, dx)

        # Avoid division by very small ρ
        ρsafe = max.(ρ, ftol/2)

        term1 = @. -1/4 * (d2ρdx2 / ρsafe)
        term2 = @.  1/8 * abs(dρdx / ρsafe)^2
        term3 = vext
        term4 = apply_J(basis, reshape(ρ .- _ρgs, size(ρgs)), ε)[:, 1, 1]

        @. F[1:N] = term1 + term2 + term3 + term4 + μ
        F[N+1] = sum(ρ)*dx - rhs_integral

        F
    end

    # solve non-linear PDE 
    sol = nlsolve(  (F,x) -> f!(F,x),
                    [ρ0; μ0];
                    method=:newton,
                    linesearch=LineSearches.BackTracking(),
                    ftol)
    println("Converged: $(sol.f_converged) μ = $(sol.zero[N+1])")

    res = (; basis, ε,
            ρ=reshape(sol.zero[1:N], size(ρgs)),
            μ0=sol.zero[N+1],
            converged=sol.f_converged,
            method=sol.method,
            ψ=nothing,          # add dummy fields to be consistent with SCFResult
            runtime_ns=time_ns() - start_ns,
            f_calls=sol.f_calls)
    res
end

function proximal_point(basis, ρk, ε, λ; ρgs,
                        ρ0=nothing, μ0=0.0,
                        ftol=1e-8, maxiter=1000)
    # Compute proximal point by solving the PDE
    ## -1/4*(∂²ρ/∂x²)/ρ + 1/8*(∂ρ/∂x)^2/ρ^2 + V_ext + 1//ε J(ρ - ρ_gs)

    vext = extract_external_potential(basis)
    N = length(ρgs)

    x = r_vectors_cart(basis)
    dx = (x[2] - x[1])[1]

    _ρgs = copy(ρgs)[:, 1, 1]  # make sure it's a 1D array
    _ρk = copy(ρk)[:, 1, 1]

    if ρ0 === nothing
        # if no initial guess, use ρgs
        ρ0 = copy(ρgs)[:, 1, 1] 
    else
        # initial guess
        ρ0 = copy(ρ0)[:, 1, 1]
    end

    rhs_integral = sum(_ρgs)*dx     # ∫ ρgs dx
    function f!(F, u)
        ρ = @view u[1:N]
        μ = u[N+1]
        dρdx   = derivative(ρ, N, dx)
        d2ρdx2 = second_derivative(ρ, N, dx)

        # Avoid division by very small ρ
        ρsafe = max.(ρ, ftol/2)

        term1 = @. -1/4 * (d2ρdx2 / ρsafe)
        term2 = @.  1/8 * abs(dρdx / ρsafe)^2
        term3 = vext
        term4 = apply_J(basis, reshape(ρ .- _ρgs, size(ρgs)), ε)[:, 1, 1]
        term5 = apply_J(basis, reshape(ρ .- _ρk, size(ρgs)), λ)[:, 1, 1]
        @. F[1:N] = term1 + term2 + term3 + term4 + term5 + μ
        F[N+1] = sum(ρ)*dx - rhs_integral

        F
    end

    sol = nlsolve(  (F,x) -> f!(F,x),
                    [ρ0; μ0];
                    method=:newton,
                    linesearch=LineSearches.BackTracking(),
                    ftol)
    if sol.f_converged == false
        @warn "Proximal point calculation did not converge"
    end

    res = (; basis, ε, λ,
            ρ=reshape(sol.zero[1:N], size(ρgs)),
            μ0=sol.zero[N+1],
            converged=sol.f_converged,
            method=sol.method,
            ψ=nothing,          # add dummy fields to be consistent with SCFResult
            f_calls=sol.f_calls)
    res
end



function proximal_direct_minimization(basis, ρk, ε, λ; ρgs, ρ0=nothing, ftol=1e-8, maxiter=1000, δ=1e-2, ρtol=0,)
    i_term = only(i for (i, t) in enumerate(basis.model.term_types) if t isa DFTK.ExternalFromReal)

    terms = [   Kinetic(),
                basis.model.term_types[i_term],
                DualMap(basis -> ρgs, ε),
                DualMapλ(basis -> ρk, λ),
                ]
    itermodel = Model(basis.model.lattice; basis.model.n_electrons, terms, spin_polarization=:spinless)
    iterbasis = PlaneWaveBasis(itermodel, Ecut=basis.Ecut, kgrid=(1,1,1))
    iterscfres = nothing
    @suppress_err (
    iterscfres = direct_minimization(iterbasis;
                                callback=identity,
                                maxiter,
                                x_reltol=ftol,
                                f_reltol=ftol,
                                linesearch=LineSearches.BackTracking(),
                                alphaguess=InitialStatic(;alpha=1),
                                is_converged=DmConvergenceInversion(ρgs, ε, δ; ρtol),
                                ψ=nothing))
    iterscfres
end

function proximal_point_algorithm(basis, ε, λs;
                                    ρgs, ρ0=nothing, μ0=0.0,
                                    tol=1e-8, ftol=1e-12,
                                    δ=1e-2, ρtol=0,
                                    maxiter=1000, seed=nothing,
                                    method=:pde_solver)
    # Proximal Point Algorithm to solve the Gross-Pitaevskii equation for the inverse problem
    start_ns = time_ns()
    if ftol > tol
        @warn "ftol > tol, last Δρ will always return 0.0"
    end
    out = nothing

    ρk = []
    ρlast = nothing
    ρnorm = 0.0

    converged = false
    converged_iter = Bool[]
    n_iter = 0
    history_Δρ = Float64[]
    f_calls = 0
    history_f_calls = Int[]

    if length(λs) == 1
        λs = fill(λs, maxiter)
    end

    for λ in λs
        if length(ρk) == 0 && ρ0 == nothing
            ρlast = copy(ρgs)
        elseif length(ρk) == 0
            ρlast = copy(ρ0)
        else
            ρlast = copy(ρk[end])
        end

        if method==:pde_solver
            out = proximal_point(basis, ρlast, ε, λ; ρgs, ρ0, ftol, maxiter)
            μ0 = out.μ0
            f_calls += out.f_calls
            push!(history_f_calls, out.f_calls)
        elseif method==:direct_minimization
            out = proximal_direct_minimization(basis, ρlast, ε, λ; ρgs, ρ0, ftol, maxiter, δ, ρtol)
        else
            println("Unknown method $(method) passed to proximal point algorithm")
            println("   avaliable options are :pde_solver, :direct_minimization")
        end

        # print the current state to the terminal window
        ρnorm = norm_hm1(basis, out.ρ .- ρlast)
        print("\r  Converged: $(out.converged), μ = $(round(μ0, sigdigits=4)), ‖ρk^ε - ρkm1^ε‖_{H^{-1}} = $(round(ρnorm, sigdigits=4))   ")
        flush(stdout)
        sleep(0.05)

        append!(ρk, [out.ρ])
        n_iter += 1
        push!(converged_iter, out.converged)
        push!(history_Δρ, ρnorm)


        if out.converged == false
            println("\n  Proximal point iteration did not converge for λ=$λ")
            converged = false
            break
        elseif 0 < ρnorm < tol
            println("\n  Proximal point iteration converged in $(length(ρk)) iterations")
            converged = true
            break
        elseif ρnorm == 0 && !any(λs .< λ) && length(ρk) > 1
            println("\n  Proximal point iteration converged in $(length(ρk)) iterations")
            converged = true
            break
        end

        ρ0 = ρk[end]
    end

    if length(λs) == maxiter && ρnorm >= tol
        println("\n  Proximal point iteration did not converge in $maxiter iterations")
        converged = false
    elseif length(λs) < maxiter
        println("\n  Provided list of λs exhausted.")
        println("  Proximal point iteration stopped after $(length(ρk)) iterations")
        coverged = false
    end
    println("‖ρ^ε - ρgs‖_{H^{-1}} = $(norm_hm1(basis, ρk[end] - ρgs))")
    println(" ")
    info = (;   converged_iter, converged, n_iter, history_Δρ,
                f_calls, history_f_calls)
    res = (; basis, converged, info, λs, ε,
            ρ=ρk[end], ρs = ρk,
            ψ=nothing,          # add dummy fields to be consistent with SCF result
            stage=:finalize, tol, maxiter,
            seed, runtime_ns=time_ns() - start_ns,
            algorithm="PPA")
    res
end
