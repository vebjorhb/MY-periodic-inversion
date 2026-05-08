include("reference.jl")
using LazyArtifacts

Ecut      = 45
kspacing  = 0.12u"1/Å"
system    = load_system("Si.extxyz")
psps      = Dict(:Si => artifact"pd_nc_sr_pbe_standard_0.4.1_upf/Si.upf")
prefix, _ = splitext(@__FILE__)
run_reference(system, prefix; kspacing, Ecut,kwargs_model = (; pseudopotentials = psps))
