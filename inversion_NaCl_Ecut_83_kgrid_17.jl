include("inversion.jl")
reference = "reference_NaCl_Ecut_83_kgrid_17.jld2"
prefix, _ = splitext(@__FILE__)
εs = exp10.(0:-0.25:-7.5)
run_exact_inversion(prefix, reference; verbose=true, retry=2,
                    δ=1e-2, method=InversionVxc(), εs, ρtol=5e-13)
