include("reference.jl")
using LazyArtifacts

Ecut      = 83
kspacing  = 0.12u"1/Å"
system    = load_system("NaCl.extxyz")
psps      = Dict(   :Na => artifact"pd_nc_sr_pbe_standard_0.4.1_upf/Na.upf", 
                    :Cl => artifact"pd_nc_sr_pbe_standard_0.4.1_upf/Cl.upf")
prefix, _ = splitext(@__FILE__)
run_reference(system, prefix; kspacing, Ecut,kwargs_model = (; pseudopotentials = psps))
