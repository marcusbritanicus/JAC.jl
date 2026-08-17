# module-Liouville-inc-MResolved-TwoLevel.jl
# Implements M-resolved two-level two-colour Liouville evolution.
# Uses PhotoIonization's existing M-resolved functions for computing gammaM.

using ..Basics, ..Defaults, ..Pulse, ..PhotoExcitation, ..PhotoIonization
using ..SelfConsistent, ..Radial, ..Nuclear
using LinearAlgebra, DelimitedFiles, Printf

export MResolvedTwoLevelScheme, performMResolvedTwoLevel

# ----------------------------------------------------------------------
# MResolvedTwoLevelScheme - user-facing scheme
# ----------------------------------------------------------------------
struct MResolvedTwoLevelScheme <: AbstractLiouvilleScheme
    levelSelection   ::LevelSelection          # exactly two indices: ground and excited
    levelNotations   ::Vector{String}          # labels for output
    gammaBase        ::Float64                 # Base ionization width (will be M-resolved)
    detuning         ::Float64                 # Δ₁ = (E₁ - E₀ - ω_ref) / ħ (a.u.)
    includeCoherences::Bool                    # Whether to track M-M' coherences
end

# Convenience constructors
function MResolvedTwoLevelScheme(indices::Vector{Int}, labels::Vector{String}, gammaBase::Float64;
                                  detuning::Float64=0.0, includeCoherences::Bool=true)
    return MResolvedTwoLevelScheme(LevelSelection(false, indices, LevelSymmetry[]),
                                   labels, gammaBase, detuning, includeCoherences)
end

function MResolvedTwoLevelScheme()
    return MResolvedTwoLevelScheme(LevelSelection(false, Int[], LevelSymmetry[]),
                                   String[], 0.0, 0.0, true)
end

function Base.show(io::IO, scheme::MResolvedTwoLevelScheme)
    println(io, "MResolvedTwoLevelScheme:")
    println(io, "  levelSelection:    $(scheme.levelSelection)")
    println(io, "  levelNotations:    $(scheme.levelNotations)")
    println(io, "  gammaBase:         $(scheme.gammaBase) a.u.")
    println(io, "  detuning:          $(scheme.detuning) a.u.")
    println(io, "  includeCoherences: $(scheme.includeCoherences)")
    if !isempty(scheme.levelSelection.indices)
        for (i, idx) in enumerate(scheme.levelSelection.indices)
            label = i <= length(scheme.levelNotations) ? scheme.levelNotations[i] : "level$(idx)"
            println(io, "    level $i: index=$(idx), label=\"$(label)\"")
        end
    end
end

# ----------------------------------------------------------------------
# MResolvedSystem - internal system struct
# ----------------------------------------------------------------------
struct MResolvedSystem
    ground::Level
    excited::Level
    groundJ::AngularJ64
    excitedJ::AngularJ64
    nGround::Int                           # 2*Jg + 1
    nExcited::Int                          # 2*Je + 1
    totalDim::Int                          # nGround + nExcited
    dipole::Matrix{ComplexF64}             # dipole matrix elements (nE × nG)
    gammaM::Vector{Float64}                # M-resolved ionization rates [nE]
    delta::Float64                         # detuning
    pulses::Vector{Pulse.AbstractPulse}    # [pump, ionizing]
    omega_ref::Float64
    dt::Float64
    t_max::Float64
    groundIndices::Dict{AngularM64, Int}   # M -> global index
    excitedIndices::Dict{AngularM64, Int}  # M -> global index
    # For mapping back: global index -> (level, M)
    indexToLevel::Vector{Int}              # 1=ground, 2=excited
    indexToM::Vector{AngularM64}           # M value
end

# ----------------------------------------------------------------------
# Helper: get projections (magnetic sublevels) for a given J
# ----------------------------------------------------------------------
function projections(J::AngularJ64)
    j = Float64(J.num) / Float64(J.den)
    m_vals = -j:1.0:j
    # Convert to Rational{Int64} using rationalize
    return [rationalize(m) for m in m_vals]
end
# ----------------------------------------------------------------------
# Helper: determine polarization component q from a pulse
# ----------------------------------------------------------------------
function getPolarizationComponent(pulse::Pulse.AbstractPulse)
    # For GaussianSimplified, we need to infer from Stokes or just use q=+1
    # In a full implementation, this would come from the pulse's polarization
    # For now, assume circular polarization with q=+1 (right circular)
    return 1
end

# ----------------------------------------------------------------------
# Compute M-resolved dipole matrix elements
# ----------------------------------------------------------------------
function computeMDipole(ground::Level, excited::Level, omega::Float64,
                        grid::Radial.Grid, q::Int, stokes::ExpStokes)
    """
    Compute dipole matrix elements <J_e M_e | d_q | J_g M_g> for all M_g, M_e.
    Returns a matrix of size (2Je+1) × (2Jg+1).
    """
    # Get reduced matrix element from PhotoExcitation
    # Use the existing computeDipole function from the two-level module
    d_red = computeReducedDipole(ground, excited, omega, grid, q, stokes)

    Jg = ground.J
    Je = excited.J

    nG = Int( 2 * Jg.num //Jg.den + 1 )
    nE = Int( 2 * Je.num //Je.den + 1 )
    D = zeros(ComplexF64, nE, nG)

    # Get projections
    Mg_list = projections(Jg)
    Me_list = projections(Je)

    # Loop over magnetic sublevels
    for (i, Mg) in enumerate(Mg_list)
        for (j, Me) in enumerate(Me_list)
            # Wigner-Eckart theorem:
            # <Je Me | d_q | Jg Mg> = (-1)^{Je-Me} * (Je 1 Jg; -Me q Mg) * <Je || d || Jg>
            # where d_red = <Je || d || Jg> / sqrt(2Je+1)
            # So: <Je Me | d_q | Jg Mg> = (-1)^{Je-Me} * sqrt(2Je+1) * (Je 1 Jg; -Me q Mg) * d_red
            threej = wigner3j(Je.num//Je.den, 1, Jg.num//Jg.den, -Me, q, Mg)

            if abs(threej) > 1e-15
                phase = (-1.0)^(Float64(Je.num//Je.den) - Float64(Me))
                D[j, i] = phase * d_red * threej
            else
                D[j, i] = 0.0 + 0.0im
            end
        end
    end

    return D
end

# ----------------------------------------------------------------------
# Compute reduced dipole from PhotoExcitation
# ----------------------------------------------------------------------
function computeReducedDipole(initial::Level, final::Level, omega::Float64,
                              grid::Radial.Grid, q::Int, stokes::ExpStokes)
    """
    Computes the reduced dipole matrix element <final || d || initial> / sqrt(2Jf+1).
    Uses PhotoExcitation to get oscillator strength.
    """
    settings = PhotoExcitation.Settings([Basics.E1], [Basics.UseCoulomb],
                                        false, false, false, false,
                                        Basics.LineSelection(), 0.0, 0.0, 1e6,
                                        stokes)
    channels = PhotoExcitation.determineChannels(final, initial, settings)
    if isempty(channels)
        error("No E1 channel found for this transition.")
    end
    line = PhotoExcitation.Line(initial, final, omega,
                                Basics.EmProperty(0.,0.), Basics.EmProperty(0.,0.),
                                Basics.TensorComp[], true, channels)
    computed = PhotoExcitation.computeAmplitudesProperties(line, grid, settings,
                                                            printout=false)

    # Extract oscillator strength
    f_coul = computed.oscStrength.Coulomb
    if f_coul < 1e-15
        @warn "Oscillator strength is zero or very small: $f_coul"
        return 0.0 + 0.0im
    end

    # For J_ground = J_g, the reduced matrix element squared is:
    # |<Jf || d || Jg>|^2 = 3*(2Jg+1)/(2*omega) * f
    Jg = initial.J
    Jg_num = Basics.twice(Jg)  # 2Jg
    d_red_sq = 3 * (Jg_num + 1) / (2 * omega) * f_coul
    d_red = sqrt(d_red_sq)

    # Get phase from amplitude
    amp = 0.0 + 0.0im
    for ch in computed.channels
        if ch.gauge == Basics.Coulomb
            amp = ch.amplitude
            break
        end
    end
    if abs(amp) < 1e-15
        @warn "Amplitude is zero, using real d_red"
        return d_red + 0.0im
    end

    return d_red * (amp / abs(amp))
end

# ----------------------------------------------------------------------
# Compute M-resolved ionization rates using PhotoIonization
# ----------------------------------------------------------------------
# ----------------------------------------------------------------------
# Compute M-resolved ionization rates using PhotoIonization
# ----------------------------------------------------------------------
function computeMIonizationRates(excitedMultiplet::Multiplet,
                                 finalMultiplet::Multiplet,
                                 nm::Nuclear.Model, grid::Radial.Grid,
                                 ionizingPulse::Pulse.AbstractPulse,
                                 piSettings::PhotoIonization.Settings)
    """
    Compute M-resolved ionization rates Γ(M) for the excited level.
    Uses PhotoIonization with proper mValue settings for each M sublevel.
    Returns a vector of length (2Je+1).
    """
    Je = excitedMultiplet.levels[1].J
    nE = Int( 2 * Je.num // Je.den + 1 )
    gammaM = zeros(Float64, nE)

    # Get the ionizing photon energy
    omega_ion = ionizingPulse.omega

    # Determine the polarization of the ionizing pulse
    stokes = piSettings.stokes

    # Loop over excited sublevels M
    Me_list = projections(Je)

    for (idx, M) in enumerate(Me_list)
        println("\n  Computing gamma for M=$(Float64(M))...")

        # Create settings for this specific M
        # We need to set mValue = M (as a Float64)
        mVal = Float64(M)

        # Create a copy of the settings with mValue set
        settingsM = PhotoIonization.Settings(
            piSettings;
            mValue = mVal,
            photonEnergies = [omega_ion],
            stokes = stokes
        )

        # Now compute the photoionization lines for this M
        # We need to compute lines between the excited level (initial)
        # and the ion levels (final)
        lines = PhotoIonization.computeLines(
            finalMultiplet,  # final states (ion)
            excitedMultiplet, # initial states (excited atom)
            nm,
            grid,
            settingsM,
            output=true
        )

        # Find the line corresponding to our excited level
        total_sigma = 0.0
        for line in lines
            if line.initialLevel.index == excitedMultiplet.levels[1].index
                # Sum cross sections for all final states
                total_sigma += line.crossSection.Coulomb
            end
        end

        # Convert cross section to rate
        # In atomic units: Γ = σ * J_photon, where J_photon = I / ω
        # For a pulse, we use the peak intensity
        # A0 is field amplitude, intensity = A0^2 / (8π α)
        intensity = ionizingPulse.A0^2 / (8 * pi * Defaults.getDefaults("alpha"))
        photon_flux = intensity / omega_ion
        gammaM[idx] = total_sigma * photon_flux

        println("    sigma = $(total_sigma), gamma = $(gammaM[idx])")
    end

    return gammaM
end

# ----------------------------------------------------------------------
# Build the M-resolved system
# ----------------------------------------------------------------------
function buildMResolvedSystem(scheme::MResolvedTwoLevelScheme, comp::Computation)
    println("\n" * "="^60)
    println("BUILDING M-RESOLVED TWO-LEVEL SYSTEM")
    println("="^60)

    # Compute the multiplet (SCF)
    println("\nRunning SCF to get atomic structure...")
    # multiplet = SelfConsistent.performSCF(comp.refConfigs[1:2], comp.nuclearModel, comp.grid, comp.asfSettings)

    initialMultiplet = SelfConsistent.performSCF([comp.refConfigs[1]], comp.nuclearModel, comp.grid, comp.asfSettings)
    excitedMultiplet = SelfConsistent.performSCF([comp.refConfigs[2]], comp.nuclearModel, comp.grid, comp.asfSettings)
    finalMultiplet   = SelfConsistent.performSCF([comp.refConfigs[3]], comp.nuclearModel, comp.grid, comp.asfSettings)

    # Get ground and excited levels
    # idx_g = scheme.levelSelection.indices[1]
    # idx_e = scheme.levelSelection.indices[2]
    # ground = multiplet.levels[idx_g]
    # excited = multiplet.levels[idx_e]

    ground  = initialMultiplet.levels[1]
    excited = excitedMultiplet.levels[2]

    println( "===============================================================> $ground" )
    println( "===============================================================> $excited" )

    Jg = ground.J
    Je = excited.J
    nG = Int( 2.0 * Jg.num // Jg.den + 1 )
    nE = Int( 2.0 * Je.num // Je.den + 1 )
    totalDim = nG + nE

    println("\nSelected levels:")
    println("  Ground:   index=$(ground.index), J=$Jg, nG=$nG, E=$(ground.energy) a.u.")
    println("  Excited:  index=$(excited.index), J=$Je, nE=$nE, E=$(excited.energy) a.u.")
    println("  Total dimension: $totalDim")

    # Energy gap and detuning
    E_gap = excited.energy - ground.energy
    if isempty(comp.pulses)
        error("At least one pulse required.")
    end

    p1 = comp.pulses[1]
    p1 = (typeof(p1) == Pulse.FelPulse) ? Pulse.convertPulse(p1) : p1
    omega_ref = p1.omega
    delta = (scheme.detuning != 0.0) ? scheme.detuning : (E_gap - omega_ref)
    println("  Reference frequency: $(omega_ref) a.u.")
    println("  Detuning: $(delta) a.u.")

    # Compute dipole matrix elements
    q = getPolarizationComponent(p1)
    stokes = ExpStokes()

    D = computeMDipole(ground, excited, omega_ref, comp.grid, q, stokes)
    println("  Dipole matrix size: $(size(D))")

    # Build index maps
    groundIndices = Dict{AngularM64, Int}()
    excitedIndices = Dict{AngularM64, Int}()
    indexToLevel = Vector{Int}(undef, totalDim)
    indexToM = Vector{AngularM64}(undef, totalDim)

    # Ground state indices (1 to nG)
    Mg_list = projections(Jg)
    for (i, M) in enumerate(Mg_list)
        M_ang = AngularM64(M.num, M.den)
        groundIndices[M_ang] = i
        indexToLevel[i] = 1  # ground
        indexToM[i] = M_ang
    end

    # Excited state indices (nG+1 to nG+nE)
    Me_list = projections(Je)
    for (j, M) in enumerate(Me_list)
        M_ang = AngularM64(M.num, M.den)
        idx = nG + j
        excitedIndices[M_ang] = idx
        indexToLevel[idx] = 2  # excited
        indexToM[idx] = M_ang
    end

    println("  Ground indices: $groundIndices")
    println("  Excited indices: $excitedIndices")

    # Compute M-resolved ionization rates
    # Use the second pulse for ionization (if available)
    gammaM = zeros(Float64, nE)
    if length(comp.pulses) >= 2
        ionizingPulse = comp.pulses[2]
        ionizingPulse = (typeof(ionizingPulse) == Pulse.FelPulse) ? Pulse.convertPulse(ionizingPulse) : ionizingPulse

        # Setup photoionization settings with proper M-resolution
        # Using the correct PhotoIonization.Settings with mValue
        piSettings = PhotoIonization.Settings(
            PhotoIonization.Settings(),
            multipoles                      = [Basics.E1, Basics.E2],
            gauges                          = [Basics.UseCoulomb, Basics.UseBabushkin],
            electronEnergies                = [ionizingPulse.omega - (excited.energy - minimum([lvl.energy for lvl in excitedMultiplet.levels]))],
            thetas                          = collect(0:π/6:2π),
            phis                            = collect(0:π/6:2π),
            mValue                          = 0.5,  # This will be overridden per M
            calcAnisotropy                  = false,
            calcPartialCs                   = false,
            calcTimeDelay                   = false,
            calcNonE1AngleDifferentialCS    = false,
            calcTensors                     = false,
            printBefore                     = false,
            stokes                          = ExpStokes(),
            freeElectronShift               = 0.0,
            lValues                         = [0,1,2,3,4]
        )

        # Compute gammaM using the PhotoIonization functions
        gammaM = computeMIonizationRates(excitedMultiplet, finalMultiplet, comp.nuclearModel, comp.grid, ionizingPulse, piSettings)
        println( "==============================================================> TWO PULSES GIVEN. CALCULATING IONISATION WIDTH FOR IONIZING PULSE.", gammaM )
    else
        # Only one pulse: use uniform gamma from scheme.gammaBase
        gammaM = computeMIonizationRates(excitedMultiplet, finalMultiplet, comp.nuclearModel, comp.grid, comp.pulses[1], piSettings)
        println( "==============================================================> ONE PULSE GIVEN. CALCULATING IONISATION WIDTH FOR FIRST PULSE.", gammaM )
    end

    # Determine time step and max time
    dt = 0.1
    t_max = 0.0
    for p in comp.pulses
        p_conv = (typeof(p) == Pulse.FelPulse) ? Pulse.convertPulse(p) : p
        if typeof(p_conv) == Pulse.GaussianSimplified
            t_max = max(t_max, p_conv.timeDelay + 3*p_conv.fwhm)
        end
    end
    if t_max == 0.0
        # Estimate from Rabi frequency
        if typeof(p1) == Pulse.GaussianSimplified
            # Estimate max Rabi frequency
            Ω_max = 0.0
            for i in 1:nG, j in 1:nE
                Ω_max = max(Ω_max, abs(D[j, i] * p1.A0))
            end
            if Ω_max > 1e-12
                t_max = 2π / Ω_max * 10
                dt = 2π / Ω_max / 100
            else
                t_max = 500.0
                dt = 0.1
            end
        else
            t_max = 500.0
            dt = 0.1
        end
    end
    println("  Time step: $(dt) a.u.")
    println("  Max time: $(t_max) a.u.")

    # Build system
    sys = MResolvedSystem(
        ground, excited,
        Jg, Je,
        nG, nE, totalDim,
        D,
        gammaM,
        delta,
        comp.pulses,
        omega_ref,
        dt, t_max,
        groundIndices, excitedIndices,
        indexToLevel, indexToM
    )

    return sys
end

# ----------------------------------------------------------------------
# Rabi frequency for M-resolved system
# ----------------------------------------------------------------------
function rabiFrequency(t::Float64, sys::MResolvedSystem)
    """
    Computes the complex Rabi frequency Ω(t) for the pump pulse.
    """
    Ω = 0.0 + 0.0im

    # Use only the first pulse (pump)
    if isempty(sys.pulses)
        return Ω
    end

    pulse = sys.pulses[1]
    p = (typeof(pulse) == Pulse.FelPulse) ? Pulse.convertPulse(pulse) : pulse

    if typeof(p) != Pulse.GaussianSimplified
        return Ω
    end

    # Envelope (Gaussian)
    sigma = p.fwhm / (2 * sqrt(2 * log(2)))
    td = p.timeDelay
    env = exp(- (t - td)^2 / (2 * sigma^2))

    # Field amplitude
    E0 = p.A0

    # Rabi frequency: Ω = -d * E0 / 2
    # For the M-resolved case, this is the overall amplitude
    # The M-dependence is in the dipole matrix D
    Ω = -E0 / 2 * env

    return Ω
end

# ----------------------------------------------------------------------
# Build M-resolved Hamiltonian
# ----------------------------------------------------------------------
function buildMResolvedHamiltonian(t::Float64, sys::MResolvedSystem)
    """
    Build the full (nG+nE) × (nG+nE) Hamiltonian matrix.
    """
    nG = sys.nGround
    nE = sys.nExcited
    N = sys.totalDim
    H = zeros(ComplexF64, N, N)

    # Ground state energies (set to 0)
    # No diagonal coupling for ground

    # Excited state energies (-detuning)
    for j in 1:nE
        idx = nG + j
        H[idx, idx] = -sys.delta
    end

    # Coupling between ground and excited (Rabi)
    Ω = rabiFrequency(t, sys)
    D = sys.dipole  # (nE × nG) matrix

    # Coupling: H = -d·E/2
    for i in 1:nG
        for j in 1:nE
            # H[ground_i, excited_j] = 0.5 * Ω * D[j, i]
            # H[excited_j, ground_i] = 0.5 * conj(Ω) * conj(D[j, i])
            H[i, nG+j] = 0.5 * Ω * D[j, i]
            H[nG+j, i] = 0.5 * conj(Ω) * conj(D[j, i])
        end
    end

    return H
end

# ----------------------------------------------------------------------
# M-resolved density matrix derivative
# ----------------------------------------------------------------------
function mResolvedDensityMatrixDerivative(ρ::Matrix{ComplexF64}, t::Float64,
                                          sys::MResolvedSystem)
    """
    Liouville equation for M-resolved density matrix with M-dependent ionization.
    """
    nG = sys.nGround
    nE = sys.nExcited
    N = sys.totalDim

    # Build Hamiltonian
    H = buildMResolvedHamiltonian(t, sys)

    # Liouville: dρ/dt = -i [H, ρ]
    dρ = -im * (H * ρ - ρ * H)

    # Add M-dependent ionization decay for excited states
    # And coherence decay
    for j in 1:nE
        idxE = nG + j
        gamma_j = sys.gammaM[1]

        # Population decay of excited sublevel
        dρ[idxE, idxE] -= gamma_j * ρ[idxE, idxE]

        # Coherence decay with other excited sublevels
        for k in 1:nE
            if j != k
                idxE2 = nG + k
                gamma_k = sys.gammaM[1]
                # Coherence decay rate = (γ_j + γ_k) / 2
                dρ[idxE, idxE2] -= 0.5 * (gamma_j + gamma_k) * ρ[idxE, idxE2]
                dρ[idxE2, idxE] -= 0.5 * (gamma_j + gamma_k) * ρ[idxE2, idxE]
            end
        end

        # Coherence decay with ground sublevels
        for i in 1:nG
            dρ[idxE, i] -= 0.5 * gamma_j * ρ[idxE, i]
            dρ[i, idxE] -= 0.5 * gamma_j * ρ[i, idxE]
        end
    end

    return dρ
end

# ----------------------------------------------------------------------
# Extract M-resolved populations and observables
# ----------------------------------------------------------------------
function extractMResolvedPopulations(ρ::Matrix{ComplexF64}, sys::MResolvedSystem)
    """
    Extract M-resolved populations and compute observables including CDAD.
    """
    nG = sys.nGround
    nE = sys.nExcited

    # Ground state M-populations
    popGroundM = [real(ρ[i, i]) for i in 1:nG]

    # Excited state M-populations
    popExcitedM = [real(ρ[nG+j, nG+j]) for j in 1:nE]

    # Total populations
    popGround = sum(popGroundM)
    popExcited = sum(popExcitedM)
    popIon = 1.0 - popGround - popExcited

    # CDAD asymmetry: (pop(+M) - pop(-M)) / (pop(+M) + pop(-M))
    # For J_e, we pair +M and -M
    cdad = Float64[]
    Me_list = projections(sys.excitedJ)

    # Find pairs of ±M
    # Get 2J as integer
    twoJ = div(2 * sys.excitedJ.num, sys.excitedJ.den)
    cdad = Float64[]

    # Loop over M values from 1 to J (in half-integer steps)
    # M runs: 1//2, 1, 3//2, 2, ... up to J
    for m_num in 2:2:twoJ  # m_num = 2M, so M = m_num/2
        M_val = m_num // 2
        M_plus = AngularM64(M_val)
        M_minus = AngularM64(-M_val)

        if haskey(sys.excitedIndices, M_plus) && haskey(sys.excitedIndices, M_minus)
            idx_plus = sys.excitedIndices[M_plus]
            idx_minus = sys.excitedIndices[M_minus]
            p_plus = real(ρ[idx_plus, idx_plus])
            p_minus = real(ρ[idx_minus, idx_minus])
            if p_plus + p_minus > 1e-15
                push!(cdad, (p_plus - p_minus) / (p_plus + p_minus))
            else
                push!(cdad, 0.0)
            end
        end
    end
    return (popGroundM, popExcitedM, popGround, popExcited, popIon, cdad)
end

# ----------------------------------------------------------------------
# Print M-resolved state for debugging
# ----------------------------------------------------------------------
function printMResolvedState(ρ::Matrix{ComplexF64}, sys::MResolvedSystem, label::String="")
    nG = sys.nGround
    nE = sys.nExcited
    N = sys.totalDim

    println("\n$label")
    println("  Density matrix ($N×$N):")
    println("  Ground sublevels (M):")
    for i in 1:nG
        M = sys.indexToM[i]
        println("    M=$M: $(real(ρ[i,i]))")
    end
    println("  Excited sublevels (M):")
    for j in 1:nE
        idx = nG + j
        M = sys.indexToM[idx]
        println("    M=$M: $(real(ρ[idx,idx]))")
    end
    println("  Coherences (|ρ_ij| > 1e-6):")
    for i in 1:N
        for j in (i+1):N
            if abs(ρ[i,j]) > 1e-6
                println("    ρ[$i,$j] = $(ρ[i,j])")
            end
        end
    end
end

# ----------------------------------------------------------------------
# Main perform function
# ----------------------------------------------------------------------
function performMResolvedTwoLevel(scheme::MResolvedTwoLevelScheme, comp::Computation, output::Bool=true)
    println("\n" * "="^60)
    println("M-RESOLVED TWO-LEVEL TWO-COLOUR IONIZATION")
    println("Full Matrix Propagator with M-dependent Ionization")
    println("="^60)

    # 1. Build the system
    sys = buildMResolvedSystem(scheme, comp)

    # 2. Initial density matrix: ground state, M=0 (or thermal distribution)
    ρ0 = zeros(ComplexF64, sys.totalDim, sys.totalDim)

    # Start in ground M=0 (if J_g >= 0)
    if haskey(sys.groundIndices, AngularM64(0))
        idx0 = sys.groundIndices[AngularM64(0)]
        ρ0[idx0, idx0] = 1.0
        println("\nInitial state: ground M=0")
    else
        # If J_g doesn't have M=0 (e.g., J_g=1/2), start in M=+1/2
        first_M = projections(sys.groundJ)[1]
        first_M_ang = AngularM64(first_M.num, first_M.den)
        idx0 = sys.groundIndices[first_M_ang]
        ρ0[idx0, idx0] = 1.0
        println("\nInitial state: ground M=$(first_M)")
    end
    printMResolvedState(ρ0, sys, "Initial state")

    # 3. Time span
    tspan = (0.0, sys.t_max)
    dt = sys.dt
    println("\nPropagation parameters:")
    println("  t_span = $(tspan[1]) to $(tspan[2]) a.u.")
    println("  dt = $(dt) a.u.")
    println("  Number of steps = $(Int(ceil((tspan[2]-tspan[1])/dt)))")

    # 4. Propagate
    println("\nStarting propagation...")
    times, ρ_history = propagateDensityMatrix(
        (ρ, t) -> mResolvedDensityMatrixDerivative(ρ, t, sys),
        ρ0, tspan, dt
    )
    println("Propagation complete. $(length(times)) time steps saved.")

    # 5. Extract observables
    println("\nExtracting observables...")
    n_steps = length(times)
    popGround = Vector{Float64}(undef, n_steps)
    popExcited = Vector{Float64}(undef, n_steps)
    popIon = Vector{Float64}(undef, n_steps)
    popGroundM = Vector{Vector{Float64}}(undef, n_steps)
    popExcitedM = Vector{Vector{Float64}}(undef, n_steps)
    cdad = Vector{Vector{Float64}}(undef, n_steps)

    for i in 1:n_steps
        pGM, pEM, pG, pE, pI, c = extractMResolvedPopulations(ρ_history[i], sys)
        popGroundM[i] = pGM
        popExcitedM[i] = pEM
        popGround[i] = pG
        popExcited[i] = pE
        popIon[i] = pI
        cdad[i] = c
    end

    # 6. Print final populations
    println("\nFinal populations (at t = $(times[end]) a.u.):")
    println("  Ground total:  $(popGround[end])")
    for (j, M) in enumerate(projections(sys.groundJ))
        println("    M=$M: $(popGroundM[end][j])")
    end
    println("  Excited total: $(popExcited[end])")
    for (j, M) in enumerate(projections(sys.excitedJ))
        println("    M=$M: $(popExcitedM[end][j])")
    end
    println("  Ionization:    $(popIon[end])")
    if !isempty(cdad[end])
        println("  CDAD values:")
        for (i, asym) in enumerate(cdad[end])
            println("    ±$(i): $(asym)")
        end
    end

    # 7. Save results to file
    println("\nSaving results to 'm_resolved_results.dat'...")
    # Create header
    header = "Time\tGround_total\tExcited_total\tIonization"
    for M in projections(sys.groundJ)
        header *= "\tGround_M=$(M)"
    end
    for M in projections(sys.excitedJ)
        header *= "\tExcited_M=$(M)"
    end
    if !isempty(cdad[1])
        for i in 1:length(cdad[1])
            header *= "\tCDAD_±$(i)"
        end
    end

    # Build data matrix
    n_cols = 4 + length(projections(sys.groundJ)) + length(projections(sys.excitedJ)) + length(cdad[1])
    data = zeros(Float64, n_steps, n_cols)
    for i in 1:n_steps
        data[i, 1] = times[i]
        data[i, 2] = popGround[i]
        data[i, 3] = popExcited[i]
        data[i, 4] = popIon[i]
        col = 5
        for p in popGroundM[i]
            data[i, col] = p
            col += 1
        end
        for p in popExcitedM[i]
            data[i, col] = p
            col += 1
        end
        for c in cdad[i]
            data[i, col] = c
            col += 1
        end
    end

    writedlm("m_resolved_results.dat", [header; data], '\t')
    println("Results saved.")

    # 8. Return results
    if output
        results = Dict{String,Any}()
        results["times"] = times
        results["popGround"] = popGround
        results["popExcited"] = popExcited
        results["ionization"] = popIon
        results["popGroundM"] = popGroundM
        results["popExcitedM"] = popExcitedM
        results["cdad"] = cdad
        results["sys"] = sys
        results["ρ_history"] = ρ_history
        return results
    else
        return nothing
    end
end

# ----------------------------------------------------------------------
# Dispatch for perform
# ----------------------------------------------------------------------
function Basics.perform(scheme::MResolvedTwoLevelScheme, comp::Computation; output::Bool=true)
    return performMResolvedTwoLevel(scheme, comp, output=output)
end
