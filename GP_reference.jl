using DFTK
using JLD2
using LinearAlgebra
using LineSearches
using Suppressor

struct LocalNonlinearityDC
	f
end
struct TermLocalNonlinearityDC <: DFTK.TermNonlinear
end
(L::LocalNonlinearityDC)(::DFTK.AbstractBasis) = TermLocalNonlinearityDC()

function DFTK.ene_ops(term::TermLocalNonlinearityDC, basis::PlaneWaveBasis{T}, ψ, occupation;
					ρ, kwargs...) where {T}
	E = sum(ρ_i -> ρ_i^2, ρ) * basis.dvol
	potential = 2*ρ
	ops = [DFTK.RealSpaceMultiplication(basis, kpt, potential[:, :, :, kpt.spin])
			for kpt in basis.kpoints]
	(; E, ops)
end

function extract_GPE_potential(basis, ρ)
	# we assume the basis is the basis of the reference calculation
	i_term = only(i for (i, t) in enumerate(basis.model.term_types) if t isa LocalNonlinearityDC)
	(; ops) = DFTK.ene_ops(basis.terms[i_term], basis, nothing, nothing; ρ)
	ops[1].potential
end

function extract_external_potential(basis)
	# we assume the basis is the basis of the reference calculation
	i_term = only(i for (i, t) in enumerate(basis.model.term_types) if t isa DFTK.ExternalFromReal)
	(; ops) = DFTK.ene_ops(basis.terms[i_term], basis, nothing, nothing; )
	ops[1].potential
end

function norm_sobolev(basis::PlaneWaveBasis, x, s::Real)
	## The homogenous periodic Sobolev norm for s=±1
    if ndims(x) > 3
        x = total_density(x)
    end
    x_fourier = fft(basis, x)
    norm_weights = (DFTK.norm2.(G_vectors_cart(basis))) .^ s
	norm_weights[1] = 0
    sqrt(sum(abs, x_fourier .* x_fourier .* norm_weights))
end
norm_h1(basis, x)    = norm_sobolev(basis, x, 1)
norm_hm1(basis, x)   = norm_sobolev(basis, x, -1)
norm_l2(basis, x)    = norm(x) * sqrt(basis.dvol)
norm_linf(basis, x)  = maximum(abs, x)


function run_reference(prefix; C=1, α=2, a=10, Ecut=500, n_electrons=1, 
							   pot::Function = (x - a/2)^2, tol=1e-12, method=:direct_minimization)
	if mpi_master()
        println()
        DFTK.versioninfo()
        println()
    end
    DFTK.reset_timer!(DFTK.timer)
	
	lattice = a .* [[1 0 0.]; [0 0 0]; [0 0 0]]
	
	terms = [Kinetic(),
	         ExternalFromReal(r -> pot(r[1])),
	         LocalNonlinearityDC(ρ -> C * ρ^α),
			]
	model = Model(lattice; n_electrons, terms, spin_polarization=:spinless);  # spinless electrons
	basis = PlaneWaveBasis(model, Ecut=Ecut, kgrid=(1, 1, 1))
	scfres = nothing
	@info "Running reference calculation with Ecut=$Ecut, n_electrons=$n_electrons, C=$C, α=$α, a=$a"
	extra_args = (; )
	if method == :direct_minimization
		@suppress_err (
		scfres = direct_minimization(basis;
										callback=identity, 
										maxiter=10_000,
										x_reltol=tol,
										f_reltol=tol,
										linesearch= LineSearches.BackTracking()));
	elseif method == :self_consistent_field
		damping    = 0.8
		scfres = self_consistent_field(basis;   
                                            callback=identity,
                                            tol,
                                            maxiter=10_000,
                                            mixing = SimpleMixing(),
                                            damping);
	else
		error("Unknown method: $method")
	end

	println("Reference converged using $(scfres.n_iter) iterations in $(round(scfres.runtime_ns*1e-9,sigdigits=3)) seconds.")
	@info "Density converged to $(scfres.history_Δρ[end])"
	
	mpi_master() && println(DFTK.timer)
	scfres
end

