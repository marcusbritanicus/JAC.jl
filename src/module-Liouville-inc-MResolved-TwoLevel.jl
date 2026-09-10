# module-Liouville-inc-MResolved-MultiLevel.jl
# Implements M-resolved multi-level multi-photon Liouville evolution.
# Sequential absorption through arbitrary number of intermediate states.
#
# Physics: XUV excites ground → excited state. NIR photons drive a ladder
#          through intermediate Rydberg states until ionization from the highest.
#
# Basis: { |g⟩, |e⟩, |r₁⟩, |r₂⟩, ..., |rₙ⟩ }
#        where |rₙ⟩ is the highest state that ionizes.

using ..Basics, ..Defaults, ..Pulse, ..PhotoExcitation, ..PhotoIonization
using ..SelfConsistent, ..Radial, ..Nuclear, ..ManyElectron
using LinearAlgebra, DelimitedFiles, Printf

export MResolvedMultiLevelScheme, performMResolvedMultiLevel, levelLabel

# ----------------------------------------------------------------------
# MResolvedMultiLevelScheme - User-facing scheme
# ----------------------------------------------------------------------
"""
    MResolvedMultiLevelScheme

Scheme for M-resolved multi-level multi-photon ionization.

# Fields
- `levelSelection`   : LevelSelection with indices of all levels [g, e, r₁, ..., rₙ]
- `levelNotations`   : String labels for each level
- `gammaIon`         : Base ionization width (a.u.); if 0, computed from PhotoIonization
- `detunings`        : Optional vector of detunings (one per transition); if empty, auto-computed
- `includeCoherences`: Whether to track M-M' coherences
- `autoDetune`       : Whether to auto-compute detunings from level energies
"""
struct MResolvedMultiLevelScheme <: AbstractLiouvilleScheme
    levelSelection   ::LevelSelection
    levelNotations   ::Vector{String}
    gammaIon         ::Float64
    detunings        ::Vector{Float64}
    includeCoherences::Bool
    autoDetune       ::Bool
end

# Convenience constructor
function MResolvedMultiLevelScheme(indices::Vector{Int}, labels::Vector{String},
                                    gammaIon::Float64;
                                    detunings::Vector{Float64}=Float64[],
                                    includeCoherences::Bool=true,
                                    autoDetune::Bool=true)
    n_real_levels = length(indices) - 1
    n_transitions = n_real_levels - 1    # = length(indices) - 2
    if !isempty(detunings) && length(detunings) != n_transitions
        @warn "Number of detunings ($(length(detunings))) does not match number of transitions ($n_transitions). Using autoDetune."
        detunings = Float64[]
    end

    return MResolvedMultiLevelScheme(
        LevelSelection(false, indices, LevelSymmetry[]),
        labels,
        gammaIon,
        detunings,
        includeCoherences,
        autoDetune
    )
end

# Default constructor
function MResolvedMultiLevelScheme()
    return MResolvedMultiLevelScheme(
        LevelSelection(false, Int[], LevelSymmetry[]),
        String[],
        0.0,
        Float64[],
        true,
        true
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
# MResolvedMultiLevelSystem - Internal system struct
#
# NOTE: Level struct has NO config field. We store the reference
#       configurations and labels separately for display purposes.
# ----------------------------------------------------------------------
struct MResolvedMultiLevelSystem
    # Level information
    levels::Vector{Level}                        # All levels [g, e, r₁, ..., rₙ]
    levelConfigs::Vector{Configuration}          # Reference configurations (for labels)
    levelNotations::Vector{String}               # User labels
    J_values::Vector{AngularJ64}                 # J for each level
    n_sublevels::Vector{Int}                     # (2J+1) for each level
    totalDim::Int                                # Sum of all sublevels

    # Dipole couplings between adjacent levels
    # dipoles[i] = coupling between level i and i+1, size (n_{i+1} × n_i)
    dipoles::Vector{Matrix{ComplexF64}}

    # M-resolved ionization rates from the highest level
    gammaIon::Vector{Float64}

    # Detunings: deltas[i] is detuning for transition i → i+1
    deltas::Vector{Float64}

    # Pulses
    pulses::Vector{Pulse.AbstractPulse}          # [pump (XUV), ionizing (NIR)]
    omega_ref::Float64

    # Time parameters
    dt::Float64
    t_max::Float64

    # Index maps
    levelIndices::Vector{Dict{AngularM64, Int}}  # M -> global index within each level
    indexToLevel::Vector{Int}                    # Global index -> level index
    indexToM::Vector{AngularM64}                 # Global index -> M value
    levelStart::Vector{Int}                      # Starting global index for each level
end

# ----------------------------------------------------------------------
# Helper functions
# ----------------------------------------------------------------------
"""
    projections(J::AngularJ64)

Return the list of M values (as Rationals) for the given J.
"""
function projections(J::AngularJ64)
    j = Float64(J.num) / Float64(J.den)
    m_vals = -j:1.0:j
    return [rationalize(m) for m in m_vals]
end

"""
    getPolarizationComponent(pulse)

Determine the polarization component q of a pulse.
For now assumes circular polarization with q=+1.
"""
function getPolarizationComponent(pulse::Pulse.AbstractPulse)
    return 1
end

"""
    levelLabel(sys::MResolvedMultiLevelSystem, level_i::Int)

Return a display label for level_i. Prefers user notation, falls back to
configuration string, falls back to "level_N".
"""
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
# Compute reduced dipole using PhotoExcitation
# ----------------------------------------------------------------------
"""
    computeReducedDipole(initial, final, omega, grid, q, stokes)

Compute the reduced dipole matrix element ⟨final ‖ d ‖ initial⟩ / √(2J_f+1)
using PhotoExcitation's oscillator strength.
"""
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
# Compute M-resolved dipole between two levels
# ----------------------------------------------------------------------
"""
    computeMDipole(initial, final, omega, grid, q, stokes)

Compute dipole matrix elements ⟨J_f M_f | d_q | J_i M_i⟩ for all M_i, M_f.
Returns matrix of size (2J_f+1) × (2J_i+1).

Uses Wigner-Eckart theorem:
    ⟨J_f M_f | d_q | J_i M_i⟩ = (-1)^(J_f - M_f) √(2J_f+1)
                                × (J_f 1 J_i; -M_f q M_i) × d_red
"""
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
# Compute M-resolved ionization rates from the highest level
# ----------------------------------------------------------------------
"""
    computeMIonizationRates(excitedLevel, finalMultiplet, nm, grid, ionizingPulse, piSettings)

Compute M-resolved ionization rates Γ(M) for the given level.
Returns a vector of length (2J+1).
"""
function computeMIonizationRates(excitedLevel::Level,
                                 finalMultiplet::Multiplet,
                                 nm::Nuclear.Model, grid::Radial.Grid,
                                 ionizingPulse::Pulse.AbstractPulse,
                                 piSettings::PhotoIonization.Settings)
    Je = excitedLevel.J
    nE = Int(2 * Je.num // Je.den + 1)
    gammaM = zeros(Float64, nE)

    omega_ion = ionizingPulse.omega
    stokes = piSettings.stokes
    Me_list = projections(Je)

    for (idx, M) in enumerate(Me_list)
        mVal = Float64(M)
        settingsM = PhotoIonization.Settings(
            piSettings;
            mValue = mVal,
            photonEnergies = [omega_ion],
            stokes = stokes
        )

        # Wrap the single level in a multiplet
        excitedMultiplet = Multiplet("excited", [excitedLevel])

        lines = PhotoIonization.computeLines(
            finalMultiplet,
            excitedMultiplet,
            nm,
            grid,
            settingsM,
            output=true
        )

        total_sigma = 0.0
        for line in lines
            if line.initialLevel.index == excitedLevel.index
                total_sigma += line.crossSection.Coulomb
            end
        end

        # Γ = σ × photon_flux (a.u.)
        intensity = ionizingPulse.A0^2 / (8 * pi * Defaults.getDefaults("alpha"))
        photon_flux = intensity / omega_ion
        gammaM[idx] = total_sigma * photon_flux
    end

    return gammaM
end

# ----------------------------------------------------------------------
# Build the multi-level M-resolved system
# ----------------------------------------------------------------------
"""
    buildMResolvedMultiLevelSystem(scheme, comp)

Build the MResolvedMultiLevelSystem from the scheme and computation.
Performs SCF for each level, computes dipoles, detunings, and ionization rates.
"""
function buildMResolvedMultiLevelSystem(scheme::MResolvedMultiLevelScheme,
                                        comp::Computation)
    println("\n" * "="^60)
    println("BUILDING M-RESOLVED MULTI-LEVEL SYSTEM")
    println("="^60)

    # ------------------------------------------------------------------
    # 1. Count configurations and levels
    #
    # Design: comp.refConfigs[end] is the ION (used only for ionization).
    # Real levels are comp.refConfigs[1:end-1].
    # ------------------------------------------------------------------
    n_configs     = length(comp.refConfigs)
    n_real_levels = n_configs - 1                 # real physical levels
    n_transitions = n_real_levels - 1             # dipole transitions

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

    # ------------------------------------------------------------------
    # 2. SCF for real levels only
    # ------------------------------------------------------------------
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

    # ------------------------------------------------------------------
    # 3. SCF for the ion (last config)
    # ------------------------------------------------------------------
    println("\nRunning SCF for the ion (ionization channel)...")
    ionMultiplet = SelfConsistent.performSCF([comp.refConfigs[end]],
                                              comp.nuclearModel,
                                              comp.grid,
                                              comp.asfSettings)
    println("  Ion config: $(comp.refConfigs[end])")
    println("  Ion levels: $(length(ionMultiplet.levels))")

    # ------------------------------------------------------------------
    # 4. Pulses
    # ------------------------------------------------------------------
    if length(comp.pulses) < 2
        error("Need at least two pulses: XUV pump and NIR ionizing.")
    end

    p1 = (typeof(comp.pulses[1]) == Pulse.FelPulse) ?
         Pulse.convertPulse(comp.pulses[1]) : comp.pulses[1]
    p2 = (typeof(comp.pulses[2]) == Pulse.FelPulse) ?
         Pulse.convertPulse(comp.pulses[2]) : comp.pulses[2]

    # ------------------------------------------------------------------
    # 5. Detunings (one per transition)
    # ------------------------------------------------------------------
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

    # ------------------------------------------------------------------
    # 6. Dipole matrix elements between adjacent REAL levels
    # ------------------------------------------------------------------
    println("\nComputing dipole matrix elements...")
    q_pump = getPolarizationComponent(p1)
    q_nir  = getPolarizationComponent(p2)
    stokes = ExpStokes()

    dipoles = Vector{Matrix{ComplexF64}}(undef, n_transitions)

    for i in 1:n_transitions
        # Transition i couples level i → level i+1
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

    # ------------------------------------------------------------------
    # 7. Build index maps for M sublevels
    # ------------------------------------------------------------------
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

    # ------------------------------------------------------------------
    # 8. Labels for real levels
    # ------------------------------------------------------------------
    levelNotations = copy(scheme.levelNotations)
    while length(levelNotations) < n_real_levels
        push!(levelNotations, string(comp.refConfigs[length(levelNotations)+1]))
    end
    # Trim if user gave more labels than levels
    levelNotations = levelNotations[1:n_real_levels]

    # ------------------------------------------------------------------
    # 9. Ionization rates from the highest real level
    # ------------------------------------------------------------------
    println("\nComputing M-resolved ionization rates from highest level...")
    gammaIon = zeros(Float64, n_sublevels[end])

    if scheme.gammaIon > 0.0
        gammaIon .= scheme.gammaIon
        println("  Using uniform gammaIon = $(scheme.gammaIon) a.u.")
    else
        piSettings = PhotoIonization.Settings(
            PhotoIonization.Settings(),
            multipoles = [Basics.E1, Basics.E2],
            gauges     = [Basics.UseCoulomb, Basics.UseBabushkin],
            electronEnergies = [p2.omega + levels[end].energy],
            thetas     = collect(0:π/6:2π),
            phis       = collect(0:π/6:2π),
            mValue     = 0.5,
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

        gammaIon = computeMIonizationRates(levels[end], ionMultiplet,
                                            comp.nuclearModel, comp.grid,
                                            p2, piSettings)
        println("  Computed ionization rates: $gammaIon")
    end

    # ------------------------------------------------------------------
    # 10. Time parameters
    # ------------------------------------------------------------------
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

    # ------------------------------------------------------------------
    # 11. Build the system (only real levels)
    # ------------------------------------------------------------------
    sys = MResolvedMultiLevelSystem(
        levels,                                # Vector{Level}, length n_real_levels
        comp.refConfigs[1:n_real_levels],      # Vector{Configuration}
        levelNotations,                        # Vector{String}
        J_values,                              # Vector{AngularJ64}
        n_sublevels,                           # Vector{Int}
        totalDim,                              # Int
        dipoles,                               # Vector{Matrix{ComplexF64}}, length n_transitions
        gammaIon,                              # Vector{Float64}, length n_sublevels[end]
        deltas,                                # Vector{Float64}, length n_transitions
        comp.pulses,                           # Vector{Pulse.AbstractPulse}
        p1.omega,                              # Float64
        dt,                                    # Float64
        t_max,                                 # Float64
        levelIndices,                          # Vector{Dict{AngularM64,Int}}
        indexToLevel,                          # Vector{Int}
        indexToM,                              # Vector{AngularM64}
        levelStart                             # Vector{Int}
    )

    return sys
end

# ----------------------------------------------------------------------
# Multi-level Rabi frequency
# ----------------------------------------------------------------------
"""
    multiLevelRabi(t, sys, transition_idx)

Compute the Rabi frequency amplitude Ω(t) for transition `transition_idx`.
- transition_idx == 1: XUV pump
- transition_idx  > 1: NIR ionizing
"""
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
# Build multi-level Hamiltonian
# ----------------------------------------------------------------------
"""
    buildMResolvedMultiLevelHamiltonian(t, sys)

Build the full block-tridiagonal Hamiltonian:

    [ 0         Ω₁D₁†/2    0          0        ]
    [ Ω₁*D₁/2   -Δ₁        Ω₂D₂†/2    0        ]
    [ 0         Ω₂*D₂/2    -Δ₂        Ω₃D₃†/2  ]
    [ 0         0           Ω₃*D₃/2   -Δ₃      ]
"""
function buildMResolvedMultiLevelHamiltonian(t::Float64,
                                             sys::MResolvedMultiLevelSystem)
    N = sys.totalDim
    H = zeros(ComplexF64, N, N)
    n_levels = length(sys.levels)

    # Diagonal blocks: energies (detunings)
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

    # Off-diagonal blocks: couplings between adjacent levels
    for level_i in 1:(n_levels - 1)
        n_i = sys.n_sublevels[level_i]
        n_j = sys.n_sublevels[level_i + 1]
        start_i = sys.levelStart[level_i]
        start_j = sys.levelStart[level_i + 1]

        D = sys.dipoles[level_i]  # size (n_j × n_i)
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
# Multi-level density matrix derivative
# ----------------------------------------------------------------------
"""
    mResolvedMultiLevelDerivative(ρ, t, sys)

Liouville equation for M-resolved multi-level system:
    dρ/dt = -i[H, ρ] + Decay
Decay: M-dependent ionization from the highest level only.
"""
function mResolvedMultiLevelDerivative(ρ::Matrix{ComplexF64}, t::Float64,
                                       sys::MResolvedMultiLevelSystem)
    n_levels = length(sys.levels)
    N = sys.totalDim

    H = buildMResolvedMultiLevelHamiltonian(t, sys)
    dρ = -im * (H * ρ - ρ * H)

    # Add ionization decay from the highest level
    highest_idx = n_levels
    start_highest = sys.levelStart[highest_idx]
    n_highest = sys.n_sublevels[highest_idx]

    for j in 1:n_highest
        idxE = start_highest + j - 1
        gamma_j = sys.gammaIon[j]

        # Population decay
        dρ[idxE, idxE] -= gamma_j * ρ[idxE, idxE]

        # Coherence decay between sublevels of highest level
        for k in 1:n_highest
            if j != k
                idxE2 = start_highest + k - 1
                gamma_k = sys.gammaIon[k]
                dρ[idxE, idxE2] -= 0.5 * (gamma_j + gamma_k) * ρ[idxE, idxE2]
                dρ[idxE2, idxE] -= 0.5 * (gamma_j + gamma_k) * ρ[idxE2, idxE]
            end
        end

        # Coherence decay between highest and all other levels
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
# Extract M-resolved populations
# ----------------------------------------------------------------------
"""
    extractMResolvedMultiLevelPopulations(ρ, sys)

Extract:
- popLevelsM: M-resolved populations for each level
- popLevels:  total populations for each level
- popIon:     ionization yield
- cdad:       CDAD asymmetry for the highest level (if J > 0)
"""
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

    # CDAD for highest level
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
# Print state for debugging
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
# Main perform function
# ----------------------------------------------------------------------
"""
    performMResolvedMultiLevel(scheme, comp, output=true)

Run the M-resolved multi-level two-colour ionization computation.
"""
function performMResolvedMultiLevel(scheme::MResolvedMultiLevelScheme,
                                    comp::Computation,
                                    output::Bool=true)
    println("\n" * "="^60)
    println("M-RESOLVED MULTI-LEVEL TWO-COLOUR IONIZATION")
    println("Sequential multiphoton absorption through Rydberg states")
    println("="^60)

    # 1. Build system
    sys = buildMResolvedMultiLevelSystem(scheme, comp)

    # 2. Initial density matrix: lowest M of ground state
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

    # 3. Time span
    tspan = (0.0, sys.t_max)
    dt = sys.dt
    println("\nPropagation parameters:")
    println("  t_span = $(tspan[1]) to $(tspan[2]) a.u.")
    println("  dt = $dt a.u.")
    println("  Number of steps = $(Int(ceil((tspan[2]-tspan[1])/dt)))")

    # 4. Propagate
    println("\nStarting propagation...")
    times, ρ_history = propagateDensityMatrix(
        (ρ, t) -> mResolvedMultiLevelDerivative(ρ, t, sys),
        ρ0, tspan, dt
    )
    println("Propagation complete. $(length(times)) time steps saved.")

    # 5. Extract observables
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

    # 6. Print final populations
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

    # 7. Save results
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

    # 8. Return
    if output
        results = Dict{String,Any}()
        results["times"] = times
        results["popLevels"] = popLevels
        results["popLevelsM"] = popLevelsM
        results["ionization"] = popIon
        results["cdad"] = cdad
        results["sys"] = sys
        results["ρ_history"] = ρ_history
        return results
    else
        return nothing
    end
end

# Dispatch
function Basics.perform(scheme::MResolvedMultiLevelScheme, comp::Computation;
                        output::Bool=true)
    return performMResolvedMultiLevel(scheme, comp, output=output)
end
