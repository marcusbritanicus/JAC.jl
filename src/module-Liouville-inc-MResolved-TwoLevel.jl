# module-Liouville-inc-MResolved-MultiLevel.jl
# M-resolved multi-level multi-photon Liouville evolution with optional
# photoelectron angular distribution (PAD).

using ..Basics, ..Defaults, ..Pulse, ..PhotoExcitation, ..PhotoIonization
using ..SelfConsistent, ..Radial, ..Nuclear, ..ManyElectron, ..AngularMomentum
using LinearAlgebra, DelimitedFiles, Printf, WignerSymbols

export MResolvedMultiLevelScheme, performMResolvedMultiLevel, levelLabel

# ----------------------------------------------------------------------
# MResolvedMultiLevelScheme
# ----------------------------------------------------------------------
struct MResolvedMultiLevelScheme <: AbstractLiouvilleScheme
    levelSelection   ::LevelSelection
    levelNotations   ::Vector{String}
    gammaIon         ::Float64
    detunings        ::Vector{Float64}
    includeCoherences::Bool
    autoDetune       ::Bool
end

function MResolvedMultiLevelScheme(indices::Vector{Int}, labels::Vector{String},
                                    gammaIon::Float64;
                                    detunings::Vector{Float64}=Float64[],
                                    includeCoherences::Bool=true,
                                    autoDetune::Bool=true)
    n_real_levels = length(indices) - 1
    n_transitions = n_real_levels - 1
    if !isempty(detunings) && length(detunings) != n_transitions
        @warn "Number of detunings ($(length(detunings))) does not match number of transitions ($n_transitions). Using autoDetune."
        detunings = Float64[]
    end

    return MResolvedMultiLevelScheme(
        LevelSelection(false, indices, LevelSymmetry[]),
        labels, gammaIon, detunings, includeCoherences, autoDetune
    )
end

function MResolvedMultiLevelScheme()
    return MResolvedMultiLevelScheme(
        LevelSelection(false, Int[], LevelSymmetry[]),
        String[], 0.0, Float64[], true, true
    )
end

function Base.show(io::IO, scheme::MResolvedMultiLevelScheme)
    println(io, "MResolvedMultiLevelScheme:")
    println(io, "  levelSelection:    $(scheme.levelSelection)")
    println(io, "  levelNotations:    $(scheme.levelNotations)")
    println(io, "  gammaIon:          $(scheme.gammaIon) a.u.")
    println(io, "  includeCoherences: $(scheme.includeCoherences)")
    println(io, "  autoDetune:        $(scheme.autoDetune)")
    if !isempty(scheme.detunings)
        println(io, "  detunings:         $(scheme.detunings)")
    else
        println(io, "  detunings:         (auto-calculated)")
    end
    if !isempty(scheme.levelSelection.indices)
        println(io, "  levels:")
        for (i, idx) in enumerate(scheme.levelSelection.indices)
            label = i <= length(scheme.levelNotations) ? scheme.levelNotations[i] : "level$(idx)"
            println(io, "    level $i: index=$(idx), label=\"$(label)\"")
        end
    end
end

# ----------------------------------------------------------------------
# MResolvedMultiLevelSystem   (+ piLines field)
# ----------------------------------------------------------------------
struct MResolvedMultiLevelSystem
    levels         ::Vector{Level}
    levelConfigs   ::Vector{Configuration}
    levelNotations ::Vector{String}
    J_values       ::Vector{AngularJ64}
    n_sublevels    ::Vector{Int}
    totalDim       ::Int

    dipoles        ::Vector{Matrix{ComplexF64}}

    gammaIon       ::Vector{Float64}
    piLines        ::Vector{PhotoIonization.Line}       # ← ADDED

    deltas         ::Vector{Float64}

    pulses         ::Vector{Pulse.AbstractPulse}
    omega_ref      ::Float64

    dt             ::Float64
    t_max          ::Float64

    levelIndices   ::Vector{Dict{AngularM64, Int}}
    indexToLevel   ::Vector{Int}
    indexToM       ::Vector{AngularM64}
    levelStart     ::Vector{Int}
end

# ----------------------------------------------------------------------
# Helpers
# ----------------------------------------------------------------------
function projections(J::AngularJ64)
    j = Float64(J.num) / Float64(J.den)
    m_vals = -j:1.0:j
    return [rationalize(m) for m in m_vals]
end

function getPolarizationComponent(pulse::Pulse.AbstractPulse)
    return 1
end

function levelLabel(sys::MResolvedMultiLevelSystem, level_i::Int)
    if level_i <= length(sys.levelNotations) && !isempty(sys.levelNotations[level_i])
        return sys.levelNotations[level_i]
    elseif level_i <= length(sys.levelConfigs)
        return string(sys.levelConfigs[level_i])
    else
        return "level_$(level_i)"
    end
end

# ----------------------------------------------------------------------
# computeReducedDipole
# ----------------------------------------------------------------------
function computeReducedDipole(initial::Level, final::Level, omega::Float64,
                              grid::Radial.Grid, q::Int, stokes::ExpStokes)
    settings = PhotoExcitation.Settings([Basics.E1], [Basics.UseCoulomb],
                                        false, false, false, false,
                                        Basics.LineSelection(), 0.0, 0.0, 1e6,
                                        stokes)
    channels = PhotoExcitation.determineChannels(final, initial, settings)
    if isempty(channels)
        @warn "No E1 channel found for this transition."
        return 0.0 + 0.0im
    end
    line = PhotoExcitation.Line(initial, final, omega,
                                Basics.EmProperty(0., 0.), Basics.EmProperty(0., 0.),
                                Basics.TensorComp[], true, channels)
    computed = PhotoExcitation.computeAmplitudesProperties(line, grid, settings,
                                                            printout=false)

    f_coul = computed.oscStrength.Coulomb
    if abs(f_coul) < 1e-15
        @warn "Oscillator strength is zero or very small: $f_coul"
        return 0.0 + 0.0im
    end

    Jg = initial.J
    Jg_num = Basics.twice(Jg)
    d_red_sq = 3 * (Jg_num + 1) / (2 * omega) * abs(f_coul)
    d_red = sqrt(d_red_sq)

    amp = 0.0 + 0.0im
    for ch in computed.channels
        if ch.gauge == Basics.Coulomb
            amp = ch.amplitude
            break
        end
    end
    if abs(amp) < 1e-15
        return d_red + 0.0im
    end

    return d_red * (amp / abs(amp))
end

# ----------------------------------------------------------------------
# computeMDipole
# ----------------------------------------------------------------------
function computeMDipole(initial::Level, final::Level, omega::Float64,
                        grid::Radial.Grid, q::Int, stokes::ExpStokes)
    d_red = computeReducedDipole(initial, final, omega, grid, q, stokes)

    Ji = initial.J
    Jf = final.J

    n_i = Int(2 * Ji.num // Ji.den + 1)
    n_f = Int(2 * Jf.num // Jf.den + 1)
    D = zeros(ComplexF64, n_f, n_i)

    Mi_list = projections(Ji)
    Mf_list = projections(Jf)

    for (i, Mi) in enumerate(Mi_list)
        for (j, Mf) in enumerate(Mf_list)
            threej = wigner3j(Jf.num // Jf.den, 1, Ji.num // Ji.den,
                              -Mf, q, Mi)
            if abs(threej) > 1e-15
                phase = (-1.0)^(Float64(Jf.num // Jf.den) - Float64(Mf))
                D[j, i] = phase * d_red * threej
            else
                D[j, i] = 0.0 + 0.0im
            end
        end
    end

    return D
end

# ----------------------------------------------------------------------
# computeMIonizationRates   ← returns (gammaM, collectedLines)
# ----------------------------------------------------------------------
function computeMIonizationRates(excitedLevel::Level,
                                 finalMultiplet::Multiplet,
                                 nm::Nuclear.Model, grid::Radial.Grid,
                                 ionizingPulse::Pulse.AbstractPulse,
                                 piSettings::PhotoIonization.Settings)
    Je = excitedLevel.J
    nE = Int(2 * Je.num // Je.den + 1)
    gammaM = zeros(Float64, nE)

    omega_ion    = ionizingPulse.omega
    omega_ion_eV = Defaults.convertUnits("energy: from atomic", omega_ion)

    stokes = piSettings.stokes
    Me_list = projections(Je)

    collectedLines = PhotoIonization.Line[]
    collectedOnce  = false

    for (idx, M) in enumerate(Me_list)
        mVal = Float64(M)
        settingsM = PhotoIonization.Settings(
            piSettings;
            mValues          = (mVal, piSettings.mValues[2]),
            photonEnergies   = [omega_ion_eV],
            electronEnergies = Float64[],
            stokes           = stokes
        )

        excitedMultiplet = Multiplet("excited", [excitedLevel])

        lines = PhotoIonization.computeLines(
            finalMultiplet, excitedMultiplet, nm, grid, settingsM,
            output=true
        )

        if !collectedOnce
            for line in lines
                if line.initialLevel.index == excitedLevel.index
                    push!(collectedLines, line)
                end
            end
            collectedOnce = true
        end

        total_sigma = 0.0
        for line in lines
            if line.initialLevel.index == excitedLevel.index
                total_sigma += line.crossSection.Coulomb
            end
        end

        intensity   = ionizingPulse.A0^2 / (8 * pi * Defaults.getDefaults("alpha"))
        photon_flux = intensity / omega_ion
        gammaM[idx] = total_sigma * photon_flux
    end

    return (gammaM, collectedLines)          # ← returns a TUPLE
end

# ======================================================================
# PAD HELPERS
# ======================================================================
function angularFunctionK_Mres(L1::Int64, L2::Int64, X::Int64,
                                Ji::AngularJ64, Jf::AngularJ64,
                                kappa1::Int64, J1::AngularJ64,
                                kappa2::Int64, J2::AngularJ64,
                                gamma::Int64,
                                M0a::Float64, M0b::Float64,
                                mAlpha::Union{Nothing, Float64} = nothing)
    s1 = Basics.Subshell(20, kappa1);  j1 = Basics.subshell_j(s1)
    s2 = Basics.Subshell(20, kappa2);  j2 = Basics.subshell_j(s2)

    wb = (2L1+1) * (Basics.twice(j1)+1) * (Basics.twice(J1)+1) *
         (2L2+1) * (Basics.twice(j2)+1) * (Basics.twice(J2)+1)
    wa = (2X+1) * sqrt(wb) *
         AngularMomentum.phaseFactor([Ji, -1, Jf, 1, AngularJ64(1//2)])

    J_val  = Float64(J1);  J_p = Float64(J2);  ℓ = X
    J₀     = Float64(Ji);  Jα  = Float64(Jf)
    jα_val = Float64(j1);  jα_p = Float64(j2)
    j_val  = Float64(L1);  j_p  = Float64(L2)

    Mα_range = mAlpha === nothing ? (-Jα:Jα) : (mAlpha:mAlpha)

    Edmonds_sum = 0.0
    for M in -J_val:J_val
        for M_dprime in -J_p:J_p
            Edmonds_sum1 = 0.0
            for Mα in Mα_range
                for mα in -jα_val:jα_val
                    for mα_prime in -jα_p:jα_p
                        T0 = (-1)^(round(Int, jα_val + jα_p + Jα + mα + mα_prime + Mα))
                        T1 = AngularMomentum.Wigner_3j(J_p,  jα_p, Jα, M_dprime, mα_prime, -Mα)
                        T2 = AngularMomentum.Wigner_3j(jα_val, J_val, Jα, -mα, M, Mα)
                        T3 = AngularMomentum.Wigner_3j(jα_val, jα_p, ℓ, mα, -mα_prime, gamma)
                        Edmonds_sum1 += T1 * T2 * T3 * T0
                    end
                end
            end

            Edmonds_sum2 = 0.0
            for m in -j_val:j_val
                for m_prime in -j_p:j_p
                    T02 = (-1)^(round(Int, j_val + j_p + J₀ + m + m_prime + M0a))
                    T12 = AngularMomentum.Wigner_3j(J_p, j_p, J₀, M_dprime, m_prime, -M0b)
                    T22 = AngularMomentum.Wigner_3j(j_val, J_val, J₀, -m, M, M0a)
                    T32 = AngularMomentum.Wigner_3j(j_val, j_p, ℓ, m, -m_prime, gamma)
                    Edmonds_sum2 += T12 * T22 * T32 * T02
                end
            end

            Edmonds_sum += (2 * ℓ + 1) * Edmonds_sum1 * Edmonds_sum2
        end
    end

    return wa * Edmonds_sum
end

function spinDensityMatrix_liouville(λ1::Int64, λ2::Int64,
                                      stokes::Basics.ExpStokes)
    if      λ1 == 1  && λ2 == 1;   return (1.0 + stokes.P3)/2.0
    elseif  λ1 == 1  && λ2 == -1;  return (stokes.P1 - stokes.P2*im)/2.0
    elseif  λ1 == -1 && λ2 == 1;   return (stokes.P1 + stokes.P2*im)/2.0
    elseif  λ1 == -1 && λ2 == -1;  return (1.0 - stokes.P3)/2.0
    else    error("Invalid helicities: λ1=$λ1, λ2=$λ2")
    end
end

function computeMResolvedPAD(ρ_E::Matrix{ComplexF64},
                              piLines::Vector{PhotoIonization.Line},
                              Je::AngularJ64,
                              θ_vals::Vector{Float64},
                              φ_vals::Vector{Float64};
                              gauge  ::Basics.EmGauge = Basics.Coulomb,
                              mAlpha ::Union{Nothing,Float64} = nothing,
                              stokes ::Basics.ExpStokes = Basics.ExpStokes())
    nθ = length(θ_vals);  nφ = length(φ_vals)
    PAD = zeros(Float64, nθ, nφ)

    Ms = projections(Je)

    println("\n  Computing M-resolved PAD on $(nθ)×$(nφ) grid, gauge = $gauge")

    for line in piLines
        sigmaBar = 0.0
        for cha in line.channels
            cha.gauge == gauge || continue
            sigmaBar += abs(cha.amplitude)^2
        end
        sigmaBar < 1e-15 && continue

        println("    line i=$(line.initialLevel.index) -> f=$(line.finalLevel.index), σ̄ = $(sigmaBar)")

        angCsCoeff = 2.0 * pi^3 * 137.03599 /
                     (line.photonEnergy * (Basics.twice(line.initialLevel.J) + 1))

        for (iθ, θ) in enumerate(θ_vals)
            for (iφ, φ) in enumerate(φ_vals)
                val = 0.0 + 0.0im
                for X in 1:20
                    for cha in line.channels
                        cha.gauge == gauge || continue
                        L1 = cha.multipole.L; J1 = cha.symmetry.J; κ1 = cha.kappa
                        l1 = Basics.subshell_l(Basics.Subshell(101, κ1))
                        j1 = Basics.subshell_j(Basics.Subshell(20, κ1))

                        for chb in line.channels
                            chb.gauge == gauge || continue
                            L2 = chb.multipole.L; J2 = chb.symmetry.J; κ2 = chb.kappa
                            l2 = Basics.subshell_l(Basics.Subshell(101, κ2))
                            j2 = Basics.subshell_j(Basics.Subshell(20, κ2))

                            AngularMomentum.isTriangle(L1, L2, X)             || continue
                            AngularMomentum.isTriangle(J1, J2, AngularJ64(X)) || continue
                            AngularMomentum.isTriangle(j1, j2, AngularJ64(X)) || continue
                            mod(l1 + l2 + X, 2) == 0                           || continue

                            amp_prod = cha.amplitude * conj(chb.amplitude)

                            for λ1 in (-1, 1), λ2 in (-1, 1)
                                γ = λ2 - λ1

                                W = 0.0
                                for μ in (-1//2, 1//2)
                                    W += PhotoIonization.angularFunctionW(
                                        θ, L1, L2, X, λ1, λ2,
                                        κ1, μ, κ2, μ, l1, l2)
                                end
                                W = W * spinDensityMatrix_liouville(λ1, λ2, stokes)
                                W = W * exp(1.0im * (λ1 - λ2) * φ)
                                W = W * (1.0im)^(L2 - L1) * (λ1 * λ2) / 2.0
                                W = W * AngularMomentum.phaseMultipole(1.0im * λ1,  cha.multipole)
                                W = W * AngularMomentum.phaseMultipole(-1.0im * λ2, chb.multipole)

                                ρK = 0.0 + 0.0im
                                for (iM, Ma) in enumerate(Ms)
                                    M0a = Float64(Ma)
                                    for (jM, Mb) in enumerate(Ms)
                                        M0b = Float64(Mb)
                                        ρ_ab = ρ_E[iM, jM]
                                        iszero(ρ_ab) && continue
                                        K = angularFunctionK_Mres(
                                            L1, L2, X,
                                            line.initialLevel.J, line.finalLevel.J,
                                            κ1, J1, κ2, J2,
                                            γ, M0a, M0b, mAlpha)
                                        ρK += ρ_ab * K
                                    end
                                end

                                val += ρK * W * amp_prod
                            end
                        end
                    end
                end
                PAD[iθ, iφ] += real(val) / sigmaBar * angCsCoeff
            end
        end
    end

    return PAD
end

# ======================================================================
# buildMResolvedMultiLevelSystem
#   FIX: piLines initialized before if/else; struct gets piLines argument.
# ======================================================================
function buildMResolvedMultiLevelSystem(scheme::MResolvedMultiLevelScheme,
                                        comp::Computation;
                                        piSettings::Union{Nothing, PhotoIonization.Settings}=nothing)
    println("\n" * "="^60)
    println("BUILDING M-RESOLVED MULTI-LEVEL SYSTEM")
    println("="^60)

    n_configs     = length(comp.refConfigs)
    n_real_levels = n_configs - 1
    n_transitions = n_real_levels - 1

    println("\nNumber of configurations: $n_configs")
    println("Number of real levels:    $n_real_levels")
    println("Number of transitions:    $n_transitions")
    println("Ion config (last):        $(comp.refConfigs[end])")

    if n_real_levels < 2
        error("Need at least 2 real levels (ground + excited). " *
              "Got $n_real_levels real levels and $n_configs total configs.")
    end
    if n_transitions < 1
        error("Need at least 1 dipole transition. Got $n_transitions. " *
              "Provide at least 3 configs (ground + excited + ion).")
    end

    multiplets  = Vector{Multiplet}(undef, n_real_levels)
    levels      = Vector{Level}(undef, n_real_levels)
    J_values    = Vector{AngularJ64}(undef, n_real_levels)
    n_sublevels = Vector{Int}(undef, n_real_levels)

    println("\nRunning SCF for real levels...")
    for i in 1:n_real_levels
        multiplets[i]  = SelfConsistent.performSCF([comp.refConfigs[i]],
                                                    comp.nuclearModel,
                                                    comp.grid,
                                                    comp.asfSettings)
        levels[i]      = multiplets[i].levels[1]
        J_values[i]    = levels[i].J
        n_sublevels[i] = Int(2 * J_values[i].num // J_values[i].den + 1)

        println("  Level $i: $(comp.refConfigs[i])  " *
                "J=$(J_values[i])  n_sub=$(n_sublevels[i])  " *
                "E=$(round(levels[i].energy, digits=6)) a.u.")
    end

    totalDim = sum(n_sublevels)
    println("\nTotal dimension: $totalDim")

    println("\nRunning SCF for the ion (ionization channel)...")
    ionMultiplet = SelfConsistent.performSCF([comp.refConfigs[end]],
                                              comp.nuclearModel,
                                              comp.grid,
                                              comp.asfSettings)
    println("  Ion config: $(comp.refConfigs[end])")
    println("  Ion levels: $(length(ionMultiplet.levels))")

    if length(comp.pulses) < 2
        error("Need at least two pulses: XUV pump and NIR ionizing.")
    end

    p1 = (typeof(comp.pulses[1]) == Pulse.FelPulse) ?
         Pulse.convertPulse(comp.pulses[1]) : comp.pulses[1]
    p2 = (typeof(comp.pulses[2]) == Pulse.FelPulse) ?
         Pulse.convertPulse(comp.pulses[2]) : comp.pulses[2]

    deltas = Vector{Float64}(undef, n_transitions)

    if scheme.autoDetune || isempty(scheme.detunings)
        println("\nAuto-calculating detunings from level energies...")
        omega_pump = p1.omega
        deltas[1]  = levels[2].energy - levels[1].energy - omega_pump

        omega_nir = p2.omega
        for i in 2:n_transitions
            deltas[i] = levels[i+1].energy - levels[i].energy - omega_nir
        end
        println("  Detunings (a.u.): $deltas")
    else
        deltas = scheme.detunings
        println("\nUsing user-specified detunings: $deltas")
    end

    println("\nComputing dipole matrix elements...")
    q_pump = getPolarizationComponent(p1)
    q_nir  = getPolarizationComponent(p2)
    stokes = ExpStokes()

    dipoles = Vector{Matrix{ComplexF64}}(undef, n_transitions)

    for i in 1:n_transitions
        omega_use = (i == 1) ? p1.omega : p2.omega
        q_use     = (i == 1) ? q_pump   : q_nir

        dipoles[i] = computeMDipole(levels[i], levels[i+1], omega_use,
                                    comp.grid, q_use, stokes)

        println("  Dipole $i: $(comp.refConfigs[i]) → $(comp.refConfigs[i+1])  " *
                "size=$(size(dipoles[i]))")
    end

    f = open( "/tmp/dipoles.txt", "w" )
    for i in 1:n_transitions
        println( f, "  Dipole $i: max|D| = $(maximum(abs.(dipoles[i])))")
        for (j, row) in enumerate(eachrow(dipoles[i]))
            println( f, "    row $j: $(row)")
        end
    end
    close( f )

    println("\nBuilding index maps...")
    levelIndices = Vector{Dict{AngularM64, Int}}(undef, n_real_levels)
    indexToLevel = Vector{Int}(undef, totalDim)
    indexToM     = Vector{AngularM64}(undef, totalDim)
    levelStart   = Vector{Int}(undef, n_real_levels)

    global_idx = 1
    for level_i in 1:n_real_levels
        levelStart[level_i]   = global_idx
        levelIndices[level_i] = Dict{AngularM64, Int}()

        M_list = projections(J_values[level_i])
        for M in M_list
            M_ang = AngularM64(M.num, M.den)
            levelIndices[level_i][M_ang] = global_idx
            indexToLevel[global_idx]     = level_i
            indexToM[global_idx]         = M_ang
            global_idx += 1
        end
    end

    levelNotations = copy(scheme.levelNotations)
    while length(levelNotations) < n_real_levels
        push!(levelNotations, string(comp.refConfigs[length(levelNotations)+1]))
    end
    levelNotations = levelNotations[1:n_real_levels]

    println("\nComputing M-resolved ionization rates from highest level...")
    gammaIon = zeros(Float64, n_sublevels[end])
    piLines  = PhotoIonization.Line[]           # ← initialized UNCONDITIONALLY

    if scheme.gammaIon > 0.0
        gammaIon .= scheme.gammaIon
        println("  Using uniform gammaIon = $(scheme.gammaIon) a.u.")
    else
        piSettings_use = if piSettings === nothing
            println("  No piSettings supplied; using internal defaults.")
            PhotoIonization.Settings(
                PhotoIonization.Settings(),
                multipoles = [Basics.E1, Basics.E2],
                gauges     = [Basics.UseCoulomb, Basics.UseBabushkin],
                electronEnergies = [p2.omega + levels[end].energy],
                thetas     = collect(0:π/6:2π),
                phis       = collect(0:π/6:2π),
                mValues    = (0.5, 0.0),
                calcAnisotropy               = false,
                calcPartialCs                = false,
                calcTimeDelay                = false,
                calcNonE1AngleDifferentialCS = false,
                calcTensors                  = false,
                printBefore                  = false,
                stokes                       = ExpStokes(),
                freeElectronShift            = 0.0,
                lValues                      = [0, 1, 2, 3, 4, 5, 6]
            )
        else
            println("  Using user-supplied PhotoIonization.Settings.")
            piSettings
        end

        # ← TUPLE destructure — now correct, since the function returns a tuple
        gammaIon, piLines = computeMIonizationRates(levels[end], ionMultiplet,
                                                    comp.nuclearModel, comp.grid,
                                                    p2, piSettings_use)
        println("  Computed ionization rates: $gammaIon")
        println("  Collected $(length(piLines)) photoionization line(s).")
    end

    dt    = 0.1
    t_max = 0.0

    for p in comp.pulses
        p_conv = (typeof(p) == Pulse.FelPulse) ? Pulse.convertPulse(p) : p
        if typeof(p_conv) == Pulse.GaussianSimplified
            t_max = max(t_max, p_conv.timeDelay + 3 * p_conv.fwhm)
        end
    end

    if t_max == 0.0
        Ω_max = 0.0
        for dmat in dipoles
            if !isempty(dmat)
                Ω_max = max(Ω_max, maximum(abs.(dmat)) * p1.A0)
            end
        end
        if Ω_max > 1e-12
            t_max = 2π / Ω_max * 10
            dt    = 2π / Ω_max / 100
        else
            t_max = 1000.0
            dt    = 0.1
        end
    end

    println("\nTime parameters:")
    println("  dt    = $dt a.u.")
    println("  t_max = $t_max a.u.")

    sys = MResolvedMultiLevelSystem(
        levels,
        comp.refConfigs[1:n_real_levels],
        levelNotations,
        J_values,
        n_sublevels,
        totalDim,
        dipoles,
        gammaIon,
        piLines,                             # ← passed to struct
        deltas,
        comp.pulses,
        p1.omega,
        dt,
        t_max,
        levelIndices,
        indexToLevel,
        indexToM,
        levelStart
    )

    return sys
end

# ----------------------------------------------------------------------
# multiLevelRabi
# ----------------------------------------------------------------------
function multiLevelRabi(t::Float64, sys::MResolvedMultiLevelSystem,
                        transition_idx::Int)
    if isempty(sys.pulses)
        return 0.0 + 0.0im
    end

    pulse_idx = (transition_idx == 1) ? 1 : 2
    if pulse_idx > length(sys.pulses)
        return 0.0 + 0.0im
    end

    pulse = sys.pulses[pulse_idx]
    p = (typeof(pulse) == Pulse.FelPulse) ? Pulse.convertPulse(pulse) : pulse

    if typeof(p) != Pulse.GaussianSimplified
        return 0.0 + 0.0im
    end

    sigma = p.fwhm / (2 * sqrt(2 * log(2)))
    td = p.timeDelay
    env = exp(-(t - td)^2 / (2 * sigma^2))

    E0 = p.A0
    Ω = -E0 / 2 * env

    return Ω
end

# ----------------------------------------------------------------------
# buildMResolvedMultiLevelHamiltonian
# ----------------------------------------------------------------------
function buildMResolvedMultiLevelHamiltonian(t::Float64,
                                             sys::MResolvedMultiLevelSystem)
    N = sys.totalDim
    H = zeros(ComplexF64, N, N)
    n_levels = length(sys.levels)

    for level_i in 1:n_levels
        start = sys.levelStart[level_i]
        n_sub = sys.n_sublevels[level_i]

        if level_i == 1
            # Ground state energy = 0
        else
            Δ = sys.deltas[level_i - 1]
            for j in 1:n_sub
                idx = start + j - 1
                H[idx, idx] = -Δ
            end
        end
    end

    for level_i in 1:(n_levels - 1)
        n_i = sys.n_sublevels[level_i]
        n_j = sys.n_sublevels[level_i + 1]
        start_i = sys.levelStart[level_i]
        start_j = sys.levelStart[level_i + 1]

        D = sys.dipoles[level_i]
        Ω = multiLevelRabi(t, sys, level_i)

        for i in 1:n_i
            for j in 1:n_j
                H[start_i + i - 1, start_j + j - 1] = 0.5 * Ω * D[j, i]
                H[start_j + j - 1, start_i + i - 1] = 0.5 * conj(Ω) * conj(D[j, i])
            end
        end
    end

    return H
end

# ----------------------------------------------------------------------
# mResolvedMultiLevelDerivative
# ----------------------------------------------------------------------
function mResolvedMultiLevelDerivative(ρ::Matrix{ComplexF64}, t::Float64,
                                       sys::MResolvedMultiLevelSystem)
    n_levels = length(sys.levels)
    N = sys.totalDim

    H = buildMResolvedMultiLevelHamiltonian(t, sys)
    dρ = -im * (H * ρ - ρ * H)

    highest_idx = n_levels
    start_highest = sys.levelStart[highest_idx]
    n_highest = sys.n_sublevels[highest_idx]

    for j in 1:n_highest
        idxE = start_highest + j - 1
        gamma_j = sys.gammaIon[j]

        dρ[idxE, idxE] -= gamma_j * ρ[idxE, idxE]

        for k in 1:n_highest
            if j != k
                idxE2 = start_highest + k - 1
                gamma_k = sys.gammaIon[k]
                dρ[idxE, idxE2] -= 0.5 * (gamma_j + gamma_k) * ρ[idxE, idxE2]
                dρ[idxE2, idxE] -= 0.5 * (gamma_j + gamma_k) * ρ[idxE2, idxE]
            end
        end

        for i in 1:N
            if i != idxE
                dρ[idxE, i] -= 0.5 * gamma_j * ρ[idxE, i]
                dρ[i, idxE] -= 0.5 * gamma_j * ρ[i, idxE]
            end
        end
    end

    return dρ
end

# ----------------------------------------------------------------------
# extractMResolvedMultiLevelPopulations
# ----------------------------------------------------------------------
function extractMResolvedMultiLevelPopulations(ρ::Matrix{ComplexF64},
                                                sys::MResolvedMultiLevelSystem)
    n_levels = length(sys.levels)

    popLevelsM = Vector{Vector{Float64}}(undef, n_levels)
    popLevels = Vector{Float64}(undef, n_levels)

    for level_i in 1:n_levels
        start = sys.levelStart[level_i]
        n_sub = sys.n_sublevels[level_i]
        popM = [real(ρ[start + j - 1, start + j - 1]) for j in 1:n_sub]
        popLevelsM[level_i] = popM
        popLevels[level_i] = sum(popM)
    end

    popIon = 1.0 - sum(popLevels)

    cdad = Float64[]
    highest_idx = n_levels
    J_highest = sys.J_values[highest_idx]
    n_highest = sys.n_sublevels[highest_idx]

    if n_highest > 1
        twoJ = div(2 * J_highest.num, J_highest.den)
        for m_num in 2:2:twoJ
            M_val = m_num // 2
            M_plus = AngularM64(M_val)
            M_minus = AngularM64(-M_val)

            if haskey(sys.levelIndices[highest_idx], M_plus) &&
               haskey(sys.levelIndices[highest_idx], M_minus)
                idx_plus = sys.levelIndices[highest_idx][M_plus]
                idx_minus = sys.levelIndices[highest_idx][M_minus]
                p_plus = real(ρ[idx_plus, idx_plus])
                p_minus = real(ρ[idx_minus, idx_minus])
                if p_plus + p_minus > 1e-15
                    push!(cdad, (p_plus - p_minus) / (p_plus + p_minus))
                else
                    push!(cdad, 0.0)
                end
            end
        end
    end

    return (popLevelsM, popLevels, popIon, cdad)
end

# ----------------------------------------------------------------------
# printMResolvedMultiLevelState
# ----------------------------------------------------------------------
function printMResolvedMultiLevelState(ρ::Matrix{ComplexF64},
                                       sys::MResolvedMultiLevelSystem,
                                       label::String="")
    n_levels = length(sys.levels)
    N = sys.totalDim

    println("\n$label")
    println("  Density matrix ($N×$N):")

    for level_i in 1:n_levels
        start = sys.levelStart[level_i]
        n_sub = sys.n_sublevels[level_i]
        lbl = levelLabel(sys, level_i)
        println("  Level $level_i ($lbl):")
        M_list = projections(sys.J_values[level_i])
        for j in 1:n_sub
            M = M_list[j]
            idx = start + j - 1
            println("    M=$M: $(real(ρ[idx, idx]))")
        end
    end
end

# ----------------------------------------------------------------------
# performMResolvedMultiLevel
#   calcPAD toggle + piSettings pass-through + PAD block
# ----------------------------------------------------------------------
function performMResolvedMultiLevel(scheme::MResolvedMultiLevelScheme,
                                    comp::Computation,
                                    output::Bool=true;
                                    calcPAD::Bool=true,
                                    piSettings::Union{Nothing, PhotoIonization.Settings}=nothing)
    println("\n" * "="^60)
    println("M-RESOLVED MULTI-LEVEL TWO-COLOUR IONIZATION")
    println("Sequential multiphoton absorption through Rydberg states")
    println("="^60)

    sys = buildMResolvedMultiLevelSystem(scheme, comp; piSettings=piSettings)

    ρ0 = zeros(ComplexF64, sys.totalDim, sys.totalDim)
    M0 = projections(sys.J_values[1])[1]
    M0_ang = AngularM64(M0.num, M0.den)
    if haskey(sys.levelIndices[1], M0_ang)
        idx0 = sys.levelIndices[1][M0_ang]
        ρ0[idx0, idx0] = 1.0
        println("\nInitial state: ground $(levelLabel(sys, 1)), M=$M0")
    else
        error("Could not find initial M state")
    end

    tspan = (0.0, sys.t_max)
    dt = sys.dt
    println("\nPropagation parameters:")
    println("  t_span = $(tspan[1]) to $(tspan[2]) a.u.")
    println("  dt = $dt a.u.")
    println("  Number of steps = $(Int(ceil((tspan[2]-tspan[1])/dt)))")

    println("\nStarting propagation...")
    times, ρ_history = propagateDensityMatrix(
        (ρ, t) -> mResolvedMultiLevelDerivative(ρ, t, sys),
        ρ0, tspan, dt
    )
    println("Propagation complete. $(length(times)) time steps saved.")

    println("\nExtracting observables...")
    n_steps = length(times)
    n_levels = length(sys.levels)

    popLevelsM = Vector{Vector{Vector{Float64}}}(undef, n_steps)
    popLevels = Vector{Vector{Float64}}(undef, n_steps)
    popIon = Vector{Float64}(undef, n_steps)
    cdad = Vector{Vector{Float64}}(undef, n_steps)

    for i in 1:n_steps
        pLM, pL, pI, c = extractMResolvedMultiLevelPopulations(ρ_history[i], sys)
        popLevelsM[i] = pLM
        popLevels[i] = pL
        popIon[i] = pI
        cdad[i] = c
    end

    println("\nFinal populations (at t = $(times[end]) a.u.):")
    for level_i in 1:n_levels
        lbl = levelLabel(sys, level_i)
        println("  $lbl: $(popLevels[end][level_i])")
        M_list = projections(sys.J_values[level_i])
        for (j, M) in enumerate(M_list)
            println("    M=$M: $(popLevelsM[end][level_i][j])")
        end
    end
    println("  Ionization: $(popIon[end])")

    if !isempty(cdad[end])
        println("  CDAD values:")
        for (i, asym) in enumerate(cdad[end])
            println("    ±$(i): $asym")
        end
    end

    # ---- PAD (switchable) ----
    PAD    = nothing
    θ_vals = Float64[]
    φ_vals = Float64[]

    if !calcPAD
        println("\n[INFO] PAD computation disabled (calcPAD=false).")
    elseif isempty(sys.piLines)
        println("\n[INFO] calcPAD=true but no photoionization lines were collected. " *
                "Skipping PAD.  (Was scheme.gammaIon > 0?)")
    else
        println("\nComputing M-resolved PAD from the highest level...")
        θ_vals = collect(0.0:π/180:π)
        φ_vals = collect(0.0:π/90:2π - π/90)

        highest_idx = n_levels
        start_h    = sys.levelStart[highest_idx]
        n_h        = sys.n_sublevels[highest_idx]
        ρ_E_final  = ρ_history[end][start_h:start_h+n_h-1, start_h:start_h+n_h-1]
        Je_highest = sys.J_values[highest_idx]

        stokes_pad = ExpStokes()
        PAD = computeMResolvedPAD(ρ_E_final, sys.piLines, Je_highest,
                                  θ_vals, φ_vals;
                                  gauge  = Basics.Coulomb,
                                  mAlpha = nothing,
                                  stokes = stokes_pad)
        println("PAD computed. Max value = $(maximum(PAD)), integral ≈ " *
                "$(sum(PAD .* sin.(θ_vals) .* (θ_vals[2]-θ_vals[1]) .* (φ_vals[2]-φ_vals[1])))")

        pad_header = "theta\\phi\t" * join([string(φ) for φ in φ_vals], "\t")
        writedlm("m_resolved_multilevel_PAD.dat",
                 [pad_header; hcat(θ_vals, PAD)], '\t')
        println("PAD saved to m_resolved_multilevel_PAD.dat")
    end

    println("\nSaving results to 'm_resolved_multilevel.dat'...")

    header = "Time"
    for level_i in 1:n_levels
        lbl = levelLabel(sys, level_i)
        header *= "\t$(lbl)_total"
        M_list = projections(sys.J_values[level_i])
        for M in M_list
            header *= "\t$(lbl)_M=$(Float64(M))"
        end
    end
    header *= "\tIonization"
    if !isempty(cdad[1])
        for i in 1:length(cdad[1])
            header *= "\tCDAD_±$(i)"
        end
    end

    n_cdad = isempty(cdad[1]) ? 0 : length(cdad[1])
    n_cols = 1 + n_levels + sum([length(projections(sys.J_values[i])) for i in 1:n_levels]) + 1 + n_cdad
    data = zeros(Float64, n_steps, n_cols)

    for i in 1:n_steps
        data[i, 1] = times[i]
        col = 2
        for level_i in 1:n_levels
            data[i, col] = popLevels[i][level_i]
            col += 1
            for p in popLevelsM[i][level_i]
                data[i, col] = p
                col += 1
            end
        end
        data[i, col] = popIon[i]
        col += 1
        if n_cdad > 0
            for c in cdad[i]
                data[i, col] = c
                col += 1
            end
        end
    end

    writedlm("m_resolved_multilevel.dat", [header; data], '\t')
    println("Results saved.")

    if output
        results = Dict{String,Any}()
        results["times"] = times
        results["popLevels"] = popLevels
        results["popLevelsM"] = popLevelsM
        results["ionization"] = popIon
        results["cdad"] = cdad
        results["sys"] = sys
        results["ρ_history"] = ρ_history
        if PAD !== nothing
            results["PAD"]    = PAD
            results["θ_vals"] = θ_vals
            results["φ_vals"] = φ_vals
        end
        return results
    else
        return nothing
    end
end

# Dispatch — forwards both keywords
function Basics.perform(scheme::MResolvedMultiLevelScheme, comp::Computation;
                        output::Bool=true,
                        calcPAD::Bool=true,
                        piSettings::Union{Nothing, PhotoIonization.Settings}=nothing)
    return performMResolvedMultiLevel(scheme, comp, output=output,
                                       calcPAD=calcPAD,
                                       piSettings=piSettings)
end
