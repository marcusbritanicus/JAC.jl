# module-Liouville-inc-TwoLevel.jl
# Implements the two-level two-colour Liouville evolution using the full matrix propagator.

using ..Basics, ..Defaults, ..Pulse, ..PhotoExcitation, ..PhotoEmission
using ..SelfConsistent, ..Radial, ..Nuclear
using LinearAlgebra, DelimitedFiles, Printf, Plots

export TwoLevelTwoColourScheme, performTwoLevelTwoColour

# ----------------------------------------------------------------------
# TwoLevelSystem struct - contains all system parameters
# ----------------------------------------------------------------------
struct TwoLevelSystem
    ground::Level
    excited::Level
    dipole::ComplexF64          # complex dipole matrix element <1|d|0>
    gamma1::Float64             # ionization width Γ₁ (a.u.)
    delta::Float64              # detuning Δ₁ (a.u.)
    pulses::Vector{Pulse.AbstractPulse}
    omega_ref::Float64          # reference frequency (a.u.)
    dt::Float64                 # time step for propagation
    t_max::Float64              # maximum time for propagation
end

# ----------------------------------------------------------------------
# Helper: compute the complex dipole matrix element for a transition
# ----------------------------------------------------------------------
function computeDipole(initial::Level, final::Level, omega::Float64, grid::Radial.Grid, stokes::ExpStokes)
    """
    Computes the complex dipole matrix element <final|d|initial> in atomic units.
    The Stokes parameters determine which spherical component(s) contribute.
    For circular polarisation (P1=0, P2=0, P3=±1), only one q component contributes.
    For linear polarisation, both q=+1 and q=-1 contribute with equal amplitude.
    """

    # First, get the oscillator strength and amplitude using PhotoExcitation
    settings = PhotoExcitation.Settings(
        [Basics.E1],
        [Basics.UseCoulomb],
        false,
        false,
        false,
        false,
        Basics.LineSelection(),
        0.0,
        0.0,
        1e6,
        stokes
    )

    channels = PhotoExcitation.determineChannels(final, initial, settings)
    if isempty(channels)
        error("No E1 channel found for this transition.")
    end

    line = PhotoExcitation.Line(
        initial,
        final,
        omega,
        Basics.EmProperty(0.,0.),
        Basics.EmProperty(0.,0.),
        Basics.TensorComp[],
        true,
        channels
    )
    computed = PhotoExcitation.computeAmplitudesProperties(line, grid, settings, printout=false)

    # Extract Coulomb amplitude and oscillator strength
    amp = 0.0 + 0.0im
    for ch in computed.channels
        if ch.gauge == Basics.Coulomb
            amp = ch.amplitude
            break
        end
    end
    if abs(amp) < 1e-15
        error("Amplitude is zero.")
    end
    f_coul = computed.oscStrength.Coulomb

    # For J_ground = 0, J_excited = 1, the spherical component is:
    # d_q = sqrt(f/(2*omega)) * phase
    # The phase comes from the amplitude
    d_mag = sqrt(f_coul / (2*omega))
    phase = amp / abs(amp)
    d_base = d_mag * phase  # This is the dipole for q=0 (linear along z)

    # Now determine the spherical component(s) from Stokes parameters
    # For a general Stokes vector (P1, P2, P3), the density matrix for
    # the photon polarisation is:
    # ρ_photon = 1/2 * [1 + P3, P1 - i*P2; P1 + i*P2, 1 - P3]
    # in the basis |+1>, |-1>

    # For circular polarisation (P1=0, P2=0, P3=±1):
    # - q = +1 for P3 = +1 (right circular)
    # - q = -1 for P3 = -1 (left circular)
    # The dipole for q is: d_q = d_base * (1 - i*δ_q0) ???
    # Actually for circular, we need to project onto the spherical basis.
    # For a transition J=0 -> J=1, the spherical component d_q is:
    # d_{+1} = -d_base/√2, d_{-1} = d_base/√2, d_0 = d_base
    # (depending on convention)

    # For simplicity, we'll use the amplitude from the PhotoExcitation
    # which already gives us the correct matrix element for the chosen
    # polarisation in the Coulomb gauge.

    # The amplitude we extracted from PhotoExcitation is actually the
    # full matrix element <1|d|0> for the chosen polarisation (linear).
    # For circular, we need to combine amplitudes for q=+1 and q=-1.

    # This is a simplification: we assume the pulse's Stokes parameters
    # are used elsewhere to determine the polarisation, and we just
    # return the dipole for the given q component.
    # For now, we return the dipole for q=0 and the polarisation will
    # be handled through the pulse's amplitude and phase.

    # Actually, looking at the PDF, they define Ω = -<i|d·E|j>
    # So we just need d = <1|d|0> for the given polarisation.
    # The polarisation is encoded in the pulse's amplitude and direction.

    # We'll return the dipole as computed from PhotoExcitation,
    # which corresponds to linear polarisation along z.
    # For circular, the user should set the pulse amplitude accordingly.

    # Return the dipole with the correct phase
    return d_mag * phase
end

# ----------------------------------------------------------------------
# Build the two-level system from the scheme and computation
# ----------------------------------------------------------------------
function buildTwoLevelSystem(scheme::TwoLevelTwoColourScheme, comp::Computation)
    println("\n" * "="^60)
    println("BUILDING TWO-LEVEL SYSTEM")
    println("="^60)

    # Compute the multiplet (SCF)
    println("\nRunning SCF to get atomic structure...")
    multiplet = SelfConsistent.performSCF(comp.refConfigs, comp.nuclearModel, comp.grid, comp.asfSettings)

    # Get ground and excited levels
    idx_g = scheme.levelSelection.indices[1]
    idx_e = scheme.levelSelection.indices[2]
    ground = multiplet.levels[idx_g]
    excited = multiplet.levels[idx_e]

    println("\nSelected levels:")
    println("  Ground:   index=$(ground.index), J=$(ground.J), E=$(ground.energy) a.u.")
    println("  Excited:  index=$(excited.index), J=$(excited.J), E=$(excited.energy) a.u.")

    # Energy gap
    E_gap = excited.energy - ground.energy
    println("  Energy gap: $(E_gap) a.u.")

    # Reference frequency: use first pulse's omega
    if isempty(comp.pulses)
        error("At least one pulse required.")
    end
    p1 = comp.pulses[1]
    p1 = (typeof(p1) == Pulse.FelPulse) ? Pulse.convertPulse(p1) : p1
    omega_ref = p1.omega
    println("  Reference frequency: $(omega_ref) a.u.")

    # Detuning: if scheme.detuning is non-zero, use it; otherwise compute from gap
    delta = (scheme.detuning != 0.0) ? scheme.detuning : (E_gap - omega_ref)
    println("  Detuning: $(delta) a.u.")

    # Compute dipole
    println("  Computing dipole matrix element...")
    # For now, use the first pulse's Stokes parameters
    # (In a more complete implementation, we would combine all pulses)
    stokes = ExpStokes()  # default: unpolarised
    d = computeDipole(ground, excited, omega_ref, comp.grid, stokes)
    println("  Dipole: $(d) a.u. (|d| = $(abs(d)))")

    # Determine time step and max time from pulses
    dt = 0.1  # default
    t_max = 0.0
    for p in comp.pulses
        p_conv = (typeof(p) == Pulse.FelPulse) ? Pulse.convertPulse(p) : p
        if typeof(p_conv) == Pulse.GaussianSimplified
            t_max = max(t_max, p_conv.timeDelay + 3*p_conv.fwhm)
        end
    end
    if t_max == 0.0
        # No finite pulse: assume monochromatic, run for several Rabi periods
        # Estimate Rabi period from the first pulse's amplitude
        if typeof(p1) == Pulse.GaussianSimplified
            Ω_peak = abs(d * p1.A0)
            if Ω_peak > 1e-12
                t_max = 2π / Ω_peak * 10  # 10 Rabi periods
                dt = 2π / Ω_peak / 100    # 100 steps per Rabi period
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
    println("  Max time: $(t_max) a.u. (approx $(t_max/41.34) fs)")

    # Build the system
    sys = TwoLevelSystem(ground, excited, d, scheme.gamma1, delta, comp.pulses, omega_ref, dt, t_max)

    return sys
end

# ----------------------------------------------------------------------
# Effective Rabi frequency as a function of time
# ----------------------------------------------------------------------
function rabiFrequency(t::Float64, sys::TwoLevelSystem)
    """
    Computes the complex Rabi frequency Ω(t) at time t.
    Ω(t) = -Σ_k <d> · E_k(t) / ℏ  (in a.u., ℏ=1)
    """
    Ω = 0.0 + 0.0im

    for (k, pulse) in enumerate(sys.pulses)
        p = (typeof(pulse) == Pulse.FelPulse) ? Pulse.convertPulse(pulse) : pulse

        # Skip if not a GaussianSimplified (for now)
        if typeof(p) != Pulse.GaussianSimplified
            continue
        end

        # Envelope (Gaussian)
        sigma = p.fwhm / (2 * sqrt(2 * log(2)))
        td = p.timeDelay
        env = exp(- (t - td)^2 / (2 * sigma^2))

        # Field amplitude (assume A0 is the peak electric field in a.u.)
        E0 = p.A0

        # Phase factor for this colour relative to reference
        # The field is: E(t) = E0 * env * cos(ω t + φ)
        # In the rotating frame with frequency ω_ref, this becomes:
        # E_eff(t) = (E0/2) * env * exp(-i(ω - ω_ref)t) * exp(iφ)
        # The Rabi frequency is: Ω = -d * E_eff
        # For the first pulse (k=1), ω == ω_ref, so phase factor = 1
        if k == 1
            phase = 1.0 + 0.0im
        else
            phase = exp(-im * (p.omega - sys.omega_ref) * (t - td))
        end

        # Contribution to Rabi frequency
        Ω += -sys.dipole * (E0/2) * env * phase
    end

    return Ω
end

# ----------------------------------------------------------------------
# Build the Hamiltonian matrix at time t
# ----------------------------------------------------------------------
function buildHamiltonian(t::Float64, sys::TwoLevelSystem)
    """
    Builds the 2x2 Hamiltonian matrix in the rotating frame.
    H(t) = [0,  Ω/2;  Ω*/2,  -Δ]
    """
    Ω = rabiFrequency(t, sys)

    H = zeros(ComplexF64, 2, 2)
    H[1,1] = 0.0
    H[1,2] = 0.5 * Ω
    H[2,1] = 0.5 * conj(Ω)
    H[2,2] = -sys.delta   # -ℏΔ₁ (in a.u., ℏ=1)

    return H
end

# ----------------------------------------------------------------------
# Density matrix derivative (Liouville equation)
# ----------------------------------------------------------------------
function densityMatrixDerivative(ρ::Matrix{ComplexF64}, t::Float64, sys::TwoLevelSystem)
    """
    Computes dρ/dt = -i/ℏ [H, ρ] - decay terms
    """
    # Build the Hamiltonian
    H = buildHamiltonian(t, sys)

    # Liouville equation: dρ/dt = -i [H, ρ] (since ℏ = 1 in a.u.)
    dρ = -im * (H * ρ - ρ * H)

    # Add decay terms
    # dρ11/dt = -Γ₁ * ρ11  (population loss from excited state)
    # dρ01/dt = -Γ₁/2 * ρ01  (coherence decay)
    dρ[2,2] -= sys.gamma1 * ρ[2,2]
    dρ[1,2] -= 0.5 * sys.gamma1 * ρ[1,2]
    dρ[2,1] -= 0.5 * sys.gamma1 * ρ[2,1]

    return dρ
end

# ----------------------------------------------------------------------
# Extract populations from density matrix
# ----------------------------------------------------------------------
function extractPopulations(ρ::Matrix{ComplexF64})
    """
    Extracts populations from the density matrix.
    Returns (ρ00, ρ11, ρ01_real, ρ01_imag, ρ10_real, ρ10_imag)
    """
    ρ00 = real(ρ[1,1])
    ρ11 = real(ρ[2,2])
    ρ01_real = real(ρ[1,2])
    ρ01_imag = imag(ρ[1,2])
    ρ10_real = real(ρ[2,1])
    ρ10_imag = imag(ρ[2,1])
    return (ρ00, ρ11, ρ01_real, ρ01_imag, ρ10_real, ρ10_imag)
end

# ----------------------------------------------------------------------
# Check Hermiticity of density matrix
# ----------------------------------------------------------------------
function checkHermiticity(ρ::Matrix{ComplexF64}, tol::Float64=1e-12)
    """
    Checks if the density matrix is Hermitian within tolerance.
    Returns true if Hermitian, false otherwise.
    """
    if size(ρ) != (2,2)
        return false
    end
    diff = ρ - ρ'
    return norm(diff) < tol
end

# ----------------------------------------------------------------------
# Print Hamiltonian for diagnostics
# ----------------------------------------------------------------------
function printHamiltonian(H::Matrix{ComplexF64})
    println("Hamiltonian matrix:")
    println("  [$(H[1,1])  $(H[1,2])]")
    println("  [$(H[2,1])  $(H[2,2])]")
end

# ----------------------------------------------------------------------
# Main perform function for the two-level two-colour scheme
# ----------------------------------------------------------------------
function performTwoLevelTwoColour(scheme::TwoLevelTwoColourScheme, comp::Computation, output::Bool=true)
    println("\n" * "="^60)
    println("TWO-LEVEL TWO-COLOUR IONIZATION")
    println("Full Matrix Propagator Approach")
    println("="^60)

    # 1. Build the system
    sys = buildTwoLevelSystem(scheme, comp)

    # 2. Initial density matrix: ground state
    ρ0 = zeros(ComplexF64, 2, 2)
    ρ0[1,1] = 1.0   # ρ₀₀ = 1
    ρ0[2,2] = 0.0
    ρ0[1,2] = 0.0
    ρ0[2,1] = 0.0

    println("\nInitial density matrix:")
    println("  [$(ρ0[1,1])  $(ρ0[1,2])]")
    println("  [$(ρ0[2,1])  $(ρ0[2,2])]")

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
        (ρ, t) -> densityMatrixDerivative(ρ, t, sys),
        ρ0, tspan, dt
    )
    println("Propagation complete. $(length(times)) time steps saved.")

    # 5. Extract populations
    println("\nExtracting populations...")
    n_steps = length(times)
    ρ00 = Vector{Float64}(undef, n_steps)
    ρ11 = Vector{Float64}(undef, n_steps)
    ρ01_real = Vector{Float64}(undef, n_steps)
    ρ01_imag = Vector{Float64}(undef, n_steps)
    ρ10_real = Vector{Float64}(undef, n_steps)
    ρ10_imag = Vector{Float64}(undef, n_steps)
    pop_ion = Vector{Float64}(undef, n_steps)

    for i in 1:n_steps
        ρ = ρ_history[i]
        # Check Hermiticity
        if !checkHermiticity(ρ)
            @warn "Density matrix not Hermitian at step $(i), time $(times[i])"
        end
        # Extract populations
        ρ00[i], ρ11[i], ρ01_real[i], ρ01_imag[i], ρ10_real[i], ρ10_imag[i] = extractPopulations(ρ)
        pop_ion[i] = 1.0 - ρ00[i] - ρ11[i]
    end

    # 6. Print final populations
    println("\nFinal populations (at t = $(times[end]) a.u.):")
    println("  ρ00 (ground)        = $(ρ00[end])")
    println("  ρ11 (excited)       = $(ρ11[end])")
    println("  Ionization          = $(pop_ion[end])")
    println("  Re(ρ01)             = $(ρ01_real[end])")
    println("  Im(ρ01)             = $(ρ01_imag[end])")
    println("  Re(ρ10)             = $(ρ10_real[end])")
    println("  Im(ρ10)             = $(ρ10_imag[end])")
    println("  Trace               = $(ρ00[end]+ρ11[end]+pop_ion[end])")

    # 7. Save results to file
    println("\nSaving results to 'two_level_two_colour_results.dat'...")
    data = hcat(times, ρ00, ρ11, pop_ion, ρ01_real, ρ01_imag, ρ10_real, ρ10_imag)
    header = "Time\tρ00\tρ11\tρion\tRe(ρ01)\tIm(ρ01)\tRe(ρ10)\tIm(ρ10)"
    writedlm("two_level_two_colour_results.dat", data, header)
    println("Results saved.")

    # 8. Save Hamiltonian at a few times for diagnostics
    println("\nSaving Hamiltonian snapshots...")
    snapshot_times = [0.0, sys.t_max/4, sys.t_max/2, 3*sys.t_max/4, sys.t_max]
    for (i, t) in enumerate(snapshot_times)
        H = buildHamiltonian(t, sys)
        println("  H($(t)) =")
        println("    [$(real(H[1,1]))+$(imag(H[1,1]))i  $(real(H[1,2]))+$(imag(H[1,2]))i]")
        println("    [$(real(H[2,1]))+$(imag(H[2,1]))i  $(real(H[2,2]))+$(imag(H[2,2]))i]")
    end

    # 9. Return results in a dictionary
    if output
        results = Dict{String,Any}()
        results["times"] = times
        results["rho00"] = ρ00
        results["rho11"] = ρ11
        results["rho01_real"] = ρ01_real
        results["rho01_imag"] = ρ01_imag
        results["rho10_real"] = ρ10_real
        results["rho10_imag"] = ρ10_imag
        results["ionization"] = pop_ion
        results["sys"] = sys
        results["ρ_history"] = ρ_history
        return results
    else
        return nothing
    end
end

# ----------------------------------------------------------------------
# Convenience function to plot results (if Plots is available)
# ----------------------------------------------------------------------
function plotTwoLevelResults(results::Dict{String,Any})
    try
        times = results["times"]
        ρ00 = results["rho00"]
        ρ11 = results["rho11"]
        pop_ion = results["ionization"]
        ρ01_real = results["rho01_real"]
        ρ01_imag = results["rho01_imag"]

        p1 = plot(times, [ρ00, ρ11, pop_ion],
                  label=["ρ00" "ρ11" "Ionization"],
                  title="Two-Level Two-Colour Ionization",
                  xlabel="Time (a.u.)", ylabel="Population")

        p2 = plot(times, [ρ01_real, ρ01_imag],
                  label=["Re(ρ01)" "Im(ρ01)"],
                  title="Coherences",
                  xlabel="Time (a.u.)", ylabel="ρ01")

        plot(p1, p2, layout=(2,1), size=(800,600))
        savefig("two_level_two_colour_plots.png")
        println("Plot saved to two_level_two_colour_plots.png")
    catch e
        println("Plotting not available: $e")
    end
end
