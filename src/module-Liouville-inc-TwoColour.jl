# module-Liouville-inc-TwoColour.jl

using ..Basics, ..Defaults, ..Pulse, ..PhotoExcitation, ..PhotoEmission, ..PhotoIonization, ..Continuum, ..Nuclear
using JLD2, DelimitedFiles, Plots, LinearAlgebra
using WignerSymbols

# Define the envelope function
function envelope( pulse::Pulse.GaussianSimplified, t::Float64 )
    sigma = pulse.fwhm / ( 2 * sqrt( 2 * log( 2 ) ) )
    t_shift = t - pulse.timeDelay
    return exp( -t_shift^2 / ( 2 * sigma^2 ) )
end

# Define the carrier function
function carrier( pulse::Pulse.GaussianSimplified, t::Float64 )
    t_shift = t - pulse.timeDelay
    return cos( pulse.omega * t_shift )
end

# Convert Intensity to Field
function fix_field_amplitude( pulse::Pulse.GaussianSimplified )::Float64
    # Your pulse.A0 currently holds I_au ( 14.25 or 1.425 )
    # Correct A0 should be: sqrt( 8πα * I_au ) / ω_au
    I_au = pulse.A0  # because that's what you mistakenly stored
    ω_au = pulse.omega
    α = 1 / 137.036

    return sqrt( 8π * α * I_au ) / ω_au
end

function pulseFunction( pulse::Pulse.GaussianSimplified, t::Float64 )
    return fix_field_amplitude( pulse ) * envelope( pulse, t ) * carrier( pulse, t )
end
"""
`struct TwoColourLevel`
    defines a struct to comprise the level information for a Liouville time evolution.
"""
struct TwoColourLevel
    leadingConfig      ::Configuration
    leadingNotation    ::String
    level              ::Level
    isContinuum        ::Bool
end

function TwoColourLevel()
    TwoColourLevel( Configuration(), "xx", Level(), false )
end

function Base.show( io::IO, level::TwoColourLevel )
    println( io, "leadingConfig:          $( level.leadingConfig )" )
    println( io, "leadingNotation:        $( level.leadingNotation )" )
    println( io, "level:                  $( level.level )" )
    println( io, "isContinuum:            $( level.isContinuum )" )
end

# ============================================================================
# Statistical Tensor Basis
# ============================================================================

"""
`StatisticalTensorBasis` - Manages the mapping from levels to statistical tensors.
"""
struct StatisticalTensorBasis
    n_levels::Int
    n_tensors::Int
    level_indices::Vector{Dict{Tuple{Int,Int}, Int}}
    level_J::Vector{Float64}
    level_energies::Vector{Float64}

    function StatisticalTensorBasis( levels::Vector{TwoColourLevel} )
        n_levels = length( levels )
        level_indices = Vector{Dict{Tuple{Int,Int}, Int}}()
        level_J = Float64[]
        level_energies = Float64[]
        global_idx = 1

        for level in levels
            J_val = level.level.J
            J_float = Float64( J_val.num ) / Float64( J_val.den )
            push!( level_J, J_float )
            push!( level_energies, level.level.energy )

            tensors = Dict{Tuple{Int,Int}, Int}()

            if level.isContinuum || J_float == 0.0
                tensors[ ( 0, 0 ) ] = global_idx
                global_idx += 1
            else
                K_max = Int( round( 2*J_float ) )
                for K in 0:K_max
                    for Q in -K:K
                        tensors[ ( K, Q ) ] = global_idx
                        global_idx += 1
                    end
                end
            end
            push!( level_indices, tensors )
        end

        return new( n_levels, global_idx - 1, level_indices, level_J, level_energies )
    end
end

function Base.show( io::IO, basis::StatisticalTensorBasis )
    println( io, "StatisticalTensorBasis:" )
    println( io, "  Total tensors: $( basis.n_tensors )" )
    println( io, "  Total levels: $( basis.n_levels )" )
    for i in 1:basis.n_levels
        println( io, "    Level $i: J=$( basis.level_J[ i ] ), tensors=$( length( basis.level_indices[ i ] ) )" )
    end
end

# Export the new functions
export StatisticalTensorBasis


"""
    build_coupling_hamiltonian( basis::StatisticalTensorBasis, reduced_dipoles::Dict, E_field::Float64, q::Int )

Build coupling Hamiltonian in statistical tensor basis for given field strength and polarization q.
"""
# function build_coupling_hamiltonian( basis::StatisticalTensorBasis, reduced_dipoles::Dict{Tuple{Int,Int}, Float64}, E_field::Float64, q::Int )
#     H = zeros( ComplexF64, basis.n_tensors, basis.n_tensors )

#     if abs( E_field ) < 1e-12
#         return H
#     end

#     for i in 1:basis.n_levels
#         for j in 1:basis.n_levels
#             i == j && continue

#             key = ( min( i,j ), max( i,j ) )
#             if !haskey( reduced_dipoles, key )
#                 continue
#             end
#             d_reduced = reduced_dipoles[ key ]
#             d_reduced == 0.0 && continue

#             strength = -d_reduced * E_field

#             tensors_i = basis.level_indices[ i ]
#             tensors_j = basis.level_indices[ j ]
#             threej = 0
#             for ( key_i, idx_i ) in tensors_i
#                 ( K_i, Q_i ) = key_i
#                 for ( key_j, idx_j ) in tensors_j
#                     ( K_j, Q_j ) = key_j

#                     if abs( K_j - K_i ) > 1
#                         continue
#                     end
#                     if Q_j != Q_i + q
#                         continue
#                     end
#                     if K_i == 0 && K_j == 0
#                         continue
#                     end

#                     # j = K/2, m = Q/2 using HalfInt ( no Rational simplification )
#                     j1 = K_j/2
#                     j2 = 1      # = 1
#                     j3 = K_i/2
#                     m1 = -Q_j/2
#                     m2 = q    # = q
#                     m3 = Q_i/2

#                    try
#                         threej = float( wigner3j( j1, j2, j3, m1, m2, m3 ) )
#                     catch e
#                         if isa( e, DomainError )
#                             threej = 0.0
#                         else
#                             rethrow()
#                         end
#                     end

#                     H[ idx_i, idx_j ] = strength * sqrt( 3 ) * threej
#                     H[ idx_j, idx_i ] = conj( H[ idx_i, idx_j ] )
#                 end
#             end
#         end
#     end

#     return H
# end

# function build_coupling_hamiltonian(basis::StatisticalTensorBasis,
#                                     reduced_dipoles::Dict{Tuple{Int,Int}, Float64},
#                                     E_field::Float64, q::Int)
#     H = zeros(ComplexF64, basis.n_tensors, basis.n_tensors)

#     if abs(E_field) < 1e-12
#         return H
#     end

#     for i in 1:basis.n_levels
#         for j in 1:basis.n_levels
#             i == j && continue

#             key = (min(i,j), max(i,j))
#             if !haskey(reduced_dipoles, key)
#                 continue
#             end
#             d_reduced = reduced_dipoles[key]
#             d_reduced == 0.0 && continue

#             strength = -d_reduced * E_field

#             J_i = basis.level_J[i]
#             J_j = basis.level_J[j]

#             tensors_i = basis.level_indices[i]
#             tensors_j = basis.level_indices[j]

#             for (key_i, idx_i) in tensors_i
#                 (K_i, Q_i) = key_i
#                 for (key_j, idx_j) in tensors_j
#                     (K_j, Q_j) = key_j

#                     # Selection rules
#                     if abs(K_j - K_i) > 1
#                         continue
#                     end
#                     if Q_j != Q_i + q
#                         continue
#                     end

#                     # The statistical tensor coupling formula from Bartschat Eq. (11.31)
#                     threej = 0
#                     try
#                         threej = wigner3j(K_i, 1, K_j, Q_i, q, -Q_j)

#                     catch e
#                         threej = 0

#                     end

#                     if abs(threej) < 1e-12
#                         continue
#                     end

#                     sixj = wigner6j(K_i, 1, K_j, J_i, J_i, J_j)

#                     # Phase factor - use cis for half-integer safety
#                     phase = cis(π * (K_i + J_i - J_j))

#                     prefactor = sqrt((2*K_i+1)*(2*K_j+1)*(2*J_i+1)*(2*J_j+1))

#                     coupling = strength * phase * prefactor * threej * sixj

#                     # For debugging
#                     if K_i == 0 && K_j == 1
#                         println("K=0→K=1 coupling: i=$i, j=$j, K_i=$K_i, K_j=$K_j, Q_i=$Q_i, Q_j=$Q_j, coupling=$coupling")
#                     end

#                     H[idx_i, idx_j] = coupling
#                     H[idx_j, idx_i] = conj(coupling)
#                 end
#             end
#         end
#     end

#     return H
# end

function build_coupling_hamiltonian(basis::StatisticalTensorBasis, reduced_dipoles::Dict{Tuple{Int,Int}, Float64}, E_field::Float64, q::Int)
    H = zeros(ComplexF64, basis.n_tensors, basis.n_tensors)

    if abs(E_field) < 1e-12
        return H
    end

    # First, collect all unique reduced dipoles we'll need
    # For intra-level couplings, we need the dipole matrix elements within the same level
    # These come from the same transition dipole, but connecting different sublevels

    for i in 1:basis.n_levels
        for j in 1:basis.n_levels
            # Get the appropriate reduced dipole
            if i != j
                key = (min(i,j), max(i,j))
                if !haskey(reduced_dipoles, key)
                    continue
                end
                d_reduced = reduced_dipoles[key]
            else
                # For intra-level couplings, use the same dipole from any connected level
                # Find a level k that connects to i
                d_reduced = 0.0
                for k in 1:basis.n_levels
                    if k != i
                        key = (min(i,k), max(i,k))
                        if haskey(reduced_dipoles, key)
                            d_reduced = reduced_dipoles[key]
                            break
                        end
                    end
                end
                d_reduced == 0.0 && continue
            end

            strength = -d_reduced * E_field

            J_i = basis.level_J[i]
            J_j = basis.level_J[j]

            tensors_i = basis.level_indices[i]
            tensors_j = basis.level_indices[j]

            for (key_i, idx_i) in tensors_i
                (K_i, Q_i) = key_i
                for (key_j, idx_j) in tensors_j
                    (K_j, Q_j) = key_j

                    # Selection rules
                    if abs(K_j - K_i) > 1
                        continue
                    end
                    if Q_j != Q_i + q
                        continue
                    end

                    # Calculate 3j symbol
                    threej = 0.0
                    try
                        threej = wigner3j(K_i, 1, K_j, Q_i, q, -Q_j)
                    catch
                        threej = 0.0
                    end

                    if abs(threej) < 1e-12
                        continue
                    end

                    # Calculate 6j symbol - this handles the recoupling
                    sixj = wigner6j(K_i, 1, K_j, J_i, J_i, J_j)

                    # Phase factor
                    phase = (-1.0)^(K_i) * (-1.0)^(J_i - J_j)

                    prefactor = sqrt((2*K_i+1)*(2*K_j+1)*(2*J_i+1)*(2*J_j+1))

                    coupling = strength * phase * prefactor * threej * sixj

                    # Debug for intra-level couplings (i == j)
                    if i == j && abs(coupling) > 1e-12
                        println("INTRA-LEVEL COUPLING: level $i, (K=$K_i,Q=$Q_i) <-> (K=$K_j,Q=$Q_j): $coupling")
                    end

                    H[idx_i, idx_j] = coupling
                    H[idx_j, idx_i] = conj(coupling)
                end
            end
        end
    end

    return H
end

# ============================================================================
# Time Evolution in Statistical Tensor Basis
# ============================================================================

"""
    solve_liouville_tensor( basis::StatisticalTensorBasis, H0::Matrix{ComplexF64},
                          coupling_func::Function, ρ0::Vector{ComplexF64},
                          tspan::Tuple{Float64,Float64}, dt::Float64 )

Solve Liouville equation in statistical tensor basis.
"""
# function solve_liouville_tensor( basis::StatisticalTensorBasis, H0::Matrix{ComplexF64},
#                                 coupling_func::Function, ρ0::Vector{ComplexF64},
#                                 tspan::Tuple{Float64,Float64}, dt::Float64 )

#     n_steps = Int( ceil( ( tspan[ 2 ] - tspan[ 1 ] ) / dt ) )
#     times = Float64[ tspan[ 1 ] ]
#     ρ_history = [ copy( ρ0 ) ]

#     ρ = copy( ρ0 )
#     t = tspan[ 1 ]

#     println( "\nTime evolution: $n_steps steps" )

#     for step in 1:n_steps
#         # Get coupling at current time <------------------
#         H_coup = coupling_func( t )
#         H_total = H0 + H_coup

#         # RK4 for vectorized density matrix
#         function dρdt( ρ_vec )
#             ρ_mat = reshape( ρ_vec, basis.n_tensors, basis.n_tensors )
#             dρ = -im * ( H_total * ρ_mat - ρ_mat * H_total )
#             return vec( dρ )
#         end

#         k1 = dρdt( ρ )
#         k2 = dρdt( ρ + dt/2 * k1 )
#         k3 = dρdt( ρ + dt/2 * k2 )
#         k4 = dρdt( ρ + dt * k3 )

#         ρ += dt/6 * ( k1 + 2k2 + 2k3 + k4 )
#         t += dt

#         if step % 100 == 0 || step == n_steps
#             push!( times, t )
#             push!( ρ_history, copy( ρ ) )
#         end
#     end

#     return times, ρ_history
# end


# ============================================================================
# Time Evolution in Statistical Tensor Basis
# ============================================================================

"""
    solve_liouville_tensor( basis::StatisticalTensorBasis, H0::Matrix{ComplexF64},
                          coupling_func::Function, ρ0::Matrix{ComplexF64},
                          tspan::Tuple{Float64,Float64}, dt::Float64 )

Solve Liouville equation in statistical tensor basis using RK4.
"""
function solve_liouville_tensor( basis::StatisticalTensorBasis, H0::Matrix{ComplexF64},
                                coupling_func::Function, ρ0::Matrix{ComplexF64},
                                tspan::Tuple{Float64,Float64}, dt::Float64 )

    n_steps = Int( ceil( ( tspan[ 2 ] - tspan[ 1 ] ) / dt ) )
    times = Float64[ tspan[ 1 ] ]
    ρ_history = [ copy( ρ0 ) ]

    ρ = copy( ρ0 )
    t = tspan[ 1 ]

    println( "\nTime evolution: $n_steps steps" )

    # Pre-allocate workspace
    n = basis.n_tensors

    for step in 1:n_steps
        # Get coupling at current time
        H_coup = coupling_func( t )
        H_total = H0 + H_coup

        # Define derivative for matrix
        function dρdt( ρ_mat::Matrix{ComplexF64} )
            return -im * ( H_total * ρ_mat - ρ_mat * H_total )
        end

        # RK4 for matrix
        k1 = dρdt( ρ )
        k2 = dρdt( ρ + dt/2 * k1 )
        k3 = dρdt( ρ + dt/2 * k2 )
        k4 = dρdt( ρ + dt * k3 )

        ρ += dt/6 * ( k1 + 2k2 + 2k3 + k4 )
        t += dt

        if step % 100 == 0 || step == n_steps
            push!( times, t )
            push!( ρ_history, copy( ρ ) )
        end
    end

    return times, ρ_history
end

export solve_liouville_tensor


# ============================================================================
# Initialize Levels from Scheme and Multiplet
# ============================================================================

"""
    initializeLevels( scheme::TwoColourScheme, multiplet::Multiplet )

Initialize levels for two-color computation.
"""
function initializeLevels( scheme::TwoColourScheme, multiplet::Multiplet )
    liouvilleLevels = TwoColourLevel[]
    noLevels = length( scheme.levelSelection.indices )

    for ( idx, index ) in enumerate( scheme.levelSelection.indices )
        for level in multiplet.levels
            if index == level.index
                leadingConf = Basics.extractConfiguration( Basics.LeadingConfiguration(), level )
                liouvLevel = TwoColourLevel( leadingConf, scheme.levelNotations[ idx ], level, false )
                push!( liouvilleLevels, liouvLevel )
            end
        end
    end

    # Add loss channel
    push!( liouvilleLevels, TwoColourLevel( Configuration( "[He]" ), scheme.levelNotations[ end ], Level(), true ) )

    # Display levels
    println( " " )
    println( "  Selected Two-Color levels:" )
    println( " " )
    for ( idx, level ) in enumerate( liouvilleLevels )
        sa = "       " * string( idx ) * " )  "
        sa = sa * string( level.leadingConfig ) * "   "
        sa = sa * string( level.leadingNotation ) * "   "
        println( sa )
    end
    println( " " )

    return liouvilleLevels
end


function get_oscillator_dipole( initial_level::Level, final_level::Level, omega::Float64, grid::Radial.Grid )
    println( "\n--- get_oscillator_dipole ---" )
    println( "Initial level index: $( initial_level.index ), J: $( initial_level.J )" )
    println( "Final level index: $( final_level.index ), J: $( final_level.J )" )
    println( "Omega: $omega" )

    # 1. Setup settings specifically for E1 transitions
    settings = PhotoExcitation.Settings( [ Basics.E1 ], [ Basics.UseCoulomb ], false, false, false, false,
                                        Basics.LineSelection(), 0.0, 0.0, 1.0e6, Basics.ExpStokes() )

    # 2. Determine the channels
    channels = PhotoExcitation.determineChannels( final_level, initial_level, settings )
    println( "Number of channels found: $( length( channels ) )" )

    if isempty( channels )
        @warn "No E1 channels found for oscillator strength calculation."
        return 0.0 + 0.0im
    end

    # Print channel info safely
    for ( i, ch ) in enumerate( channels )
        println( "Channel $i: type=$( typeof( ch ) )" )
        for name in fieldnames( typeof( ch ) )
            try
                println( "  $name = $( getfield( ch, name ) )" )
            catch
                println( "  $name = <error accessing>" )
            end
        end
    end

    # 3. Create and compute the line properties
    line = PhotoExcitation.Line( initial_level, final_level, omega,
                                Basics.EmProperty( 0.,0. ), Basics.EmProperty( 0.,0. ),
                                Basics.TensorComp[], true, channels )

    computed_line = PhotoExcitation.computeAmplitudesProperties( line, grid, settings, printout=false )

    println( "Computed line oscStrength.Coulomb: $( computed_line.oscStrength.Coulomb )" )
    println( "Computed line oscStrength.Babushkin: $( computed_line.oscStrength.Babushkin )" )
    println( "Number of computed channels: $( length( computed_line.channels ) )" )

    # 4. Extract f and convert to dipole magnitude: |d| = sqrt( 3f / ( 2ω ) )
    f_coulomb = computed_line.oscStrength.Coulomb
    println( "f_coulomb = $f_coulomb" )

    d_mag = sqrt( 3 * f_coulomb / ( 2 * omega ) )
    println( "d_mag = $d_mag" )

    # 5. Extract phase from the first channel's amplitude
    phase = 1.0 + 0.0im
    if !isempty( computed_line.channels )
        amp = computed_line.channels[ 1 ].amplitude
        println( "Channel 1 amplitude: $amp" )
        if abs( amp ) > 1e-15
            phase = amp / abs( amp )
        end
    end

    result = d_mag * phase
    println( "Final dipole ( complex ): $result" )
    println( "--- end get_oscillator_dipole ---" )

    return result
end


# function populations_from_tensors(ρ_mat, basis, level_idx, J::Float64)
#     tensors = basis.level_indices[level_idx]
#     K_max = Int(round(2J))

#     r_K0 = zeros(Float64, K_max+1)
#     for K in 0:K_max
#         if haskey(tensors, (K, 0))
#             idx = tensors[(K, 0)]
#             r_K0[K+1] = real(ρ_mat[idx, idx])
#         end
#     end

#     m_vals = -J:1.0:J
#     pops = zeros(Float64, length(m_vals))

#     norm = sqrt(2J + 1)  # Normalization factor for Option 2

#     for (i, m) in enumerate(m_vals)
#         total = 0.0
#         for K in 0:K_max
#             # Use 'm' not 'M' here
#             threej = wigner3j(J, J, K, m, -m, 0.0)
#             if abs(threej) > 1e-15
#                 total += (2K+1) * threej * r_K0[K+1]
#             end
#         end
#         pops[i] = total / norm  # Divide here for Option 2
#     end

#     return m_vals, pops
# end

function populations_from_tensors(ρ_mat, basis, level_idx, J::Float64)
    tensors = basis.level_indices[level_idx]
    K_max = Int(round(2J))
    r_K0 = zeros(Float64, K_max+1)
    for K in 0:K_max
        if haskey(tensors, (K, 0))
            idx = tensors[(K, 0)]
            r_K0[K+1] = real(ρ_mat[idx, idx])
        end
    end
    m_vals = -J:1.0:J
    pops = zeros(Float64, length(m_vals))
    denominator = 2J + 1
    for (i, m) in enumerate(m_vals)
        total = 0.0
        for K in 0:K_max
            threej = wigner3j(J, J, K, m, -m, 0.0)
            abs(threej) < 1e-15 && continue
            phase = (-1.0)^(J - m)
            total += phase * sqrt(2K+1) * threej * r_K0[K+1]
        end
        pops[i] = total / denominator
    end
    return m_vals, pops
end

function perform_statistical_tensor( scheme::TwoColourScheme, computation::Computation; output::Bool=true )
    println( "\n" * "="^60 )
    println( "STATISTICAL TENSOR RABI OSCILLATIONS" )
    println( "="^60 )

    # Convert pulses
    pulses = Pulse.AbstractPulse[]
    for pulse in computation.pulses
        if typeof( pulse ) == Pulse.GaussianSimplified
            push!( pulses, pulse )
        elseif typeof( pulse ) == Pulse.FelPulse
            push!( pulses, Pulse.convertPulse( pulse ) )
        else
            error( "Unknown pulse = $pulse" )
        end
    end

    # Get atomic structure
    println( "\n📊 Computing atomic structure..." )
    multiplet = SelfConsistent.performSCF( computation.refConfigs, computation.nuclearModel, computation.grid, computation.asfSettings )

    # Initialize levels and density matrix
    levels = initializeLevels( scheme, multiplet )
    noLevels = length( levels )
    println( "Number of levels: $noLevels" )

    for ( idx, level ) in enumerate( levels )
        println( "  Level $idx: $( level.leadingNotation ), energy = $( level.level.energy ) a.u." )
    end

    # Build statistical tensor basis
    basis = StatisticalTensorBasis( levels )
    println( basis )

    # Atomic Hamiltonian
    H0 = zeros( ComplexF64, basis.n_tensors, basis.n_tensors )
    min_energy = minimum( [ level.level.energy for level in levels ] )
    for i in 1:basis.n_levels
        E = basis.level_energies[ i ] - min_energy
        for idx in values( basis.level_indices[ i ] )
            H0[ idx, idx ] = E
        end
    end

    # Calculate the dipole using the helper function
    omega_trans = levels[ 2 ].level.energy - levels[ 1 ].level.energy
    d_complex = get_oscillator_dipole( levels[ 1 ].level, levels[ 2 ].level, omega_trans, computation.grid )

    println( "==== DIAGNOSTICS ( dipole matrix elements ) ====" )
    println( d_complex, " ", abs( d_complex ) )

    # Convert to real for the Dictionary builder
    d_reduced_val = abs( d_complex )

    # Map the reduced dipole to the levels
    reduced_dipoles = Dict( ( 1, 2 ) => d_reduced_val )

    # Build the template
    H_interaction_template = build_coupling_hamiltonian( basis, reduced_dipoles, 1.0, 0 )

    # After building H_interaction_template
    println("============================> ", size(H_interaction_template), " <============================" )

    # Add this debug block:
    println("\n=== Checking excited state intra-level couplings ===")
    for idx in [5,6,7,8]  # Indices for excited state (level 2)
        for jdx in [5,6,7,8]
            if idx != jdx && abs(H_interaction_template[idx, jdx]) > 1e-10
                println("  H[$idx,$jdx] = ", H_interaction_template[idx, jdx])
            end
        end
    end

    # Also check couplings from excited K=1 to excited K=0 specifically:
    println("\n=== Specific couplings to excited K=0 (index 5) ===")
    for idx in [6,7,8]
        if abs(H_interaction_template[5, idx]) > 1e-10
            println("  H[5,$idx] = ", H_interaction_template[5, idx])
            println("  H[$idx,5] = ", H_interaction_template[idx, 5])
        end
    end

    println("============================> ------------------ <============================" )


    # After building H_interaction_template
    println("\n=== COUPLING STRENGTH CHECK ===")
    peak_field = fix_field_amplitude(pulses[1])
    println("Peak field: $peak_field")
    println("Reduced dipole: $d_reduced_val")
    rabi_freq = abs(d_reduced_val * peak_field)
    println("Rabi frequency: $rabi_freq Hz (atomic units: $rabi_freq a.u.)")
    println("Rabi period: $(2π/rabi_freq) a.u.")

    # Check if any coupling exists between ground and excited state tensors
    g_tensors = basis.level_indices[1]
    e_tensors = basis.level_indices[2]
    println("\nGround state tensors: $g_tensors")
    println("Excited state tensors: $e_tensors")

    for ((k1,q1), idx1) in g_tensors
        for ((k2,q2), idx2) in e_tensors
            val = H_interaction_template[idx1, idx2]
            if abs(val) > 1e-10
                println("Coupling: (K=$k1,Q=$q1) <-> (K=$k2,Q=$q2): $val")
            end
        end
    end

    println( "\n" * "="^60 )
    println( "DIAGNOSTICS" )
    println( "="^60 )

    println( "Number of levels: ", basis.n_levels )
    println( "Number of tensors: ", basis.n_tensors )
    println( "Reduced dipole value: ", d_reduced_val )
    println( "" )

    # Print level structure
    println( "Level J values: ", basis.level_J )
    println( "Level energies: ", basis.level_energies )
    println( "" )

    # Print tensor indices for each level
    for i in 1:basis.n_levels
        println( "Level $i tensors:" )
        for ( ( K,Q ), idx ) in basis.level_indices[ i ]
            println( "  K=$K, Q=$Q -> global index $idx" )
        end
    end

    # Check what the coupling builder sees
    println( "\nTesting coupling for q=0, E=1.0:" )
    for i in 1:basis.n_levels
        for j in i+1:basis.n_levels
            key = ( i, j )
            if haskey( reduced_dipoles, key )
                println( "  Dipole( $i,$j ) = $( reduced_dipoles[ key ] )" )
            else
                println( "  No dipole for ( $i,$j )" )
            end

            # Check tensor compatibility
            tensors_i = basis.level_indices[ i ]
            tensors_j = basis.level_indices[ j ]
            println( "    Level $i has $( length( tensors_i ) ) tensors, Level $j has $( length( tensors_j ) ) tensors" )

            for ( key_i, idx_i ) in tensors_i
                for ( key_j, idx_j ) in tensors_j
                    K_i, Q_i = key_i
                    K_j, Q_j = key_j
                    if abs( K_j - K_i ) <= 1 && Q_j == Q_i + 0
                        println( "    Compatible: ( $K_i,$Q_i )->( $K_j,$Q_j )" )
                    end
                end
            end
        end
    end

    # Count non-zeros properly
    nz = 0
    for i in 1:size( H_interaction_template,1 )
        for j in 1:size( H_interaction_template,2 )
            if abs( H_interaction_template[ i,j ] ) > 1e-15
                nz += 1
            end
        end
    end

    println( "\nNon-zero elements in H_template: $nz" )
    # Get pulse and define coupling
    pulse = computation.pulses[ 1 ]
    if typeof( pulse ) == Pulse.FelPulse
        pulse = Pulse.convertPulse( pulse )
    end

    # Get pulse
    pulse = computation.pulses[ 1 ]
    if typeof( pulse ) == Pulse.FelPulse
        pulse = Pulse.convertPulse( pulse )
    end

    function coupling_func( t )
        return H_interaction_template .* pulseFunction( pulse, t )
    end

    # Convert J from an AngularJ64 object to a number ( e.g., 1/2 -> 0.5 )
    J_g_val = levels[ 1 ].level.J.num / levels[ 1 ].level.J.den

    ρ0 = zeros( ComplexF64, basis.n_tensors, basis.n_tensors )
    idx_g00 = basis.level_indices[ 1 ][ ( 0,0 ) ]

    # Use the numeric value for the calculation
    ρ0[ idx_g00, idx_g00 ] = sqrt( 2 * J_g_val + 1 )

    # After building H_interaction_template, add these diagnostics:

    println( "\n" * "="^60 )
    println( "DIAGNOSTICS" )
    println( "="^60 )

    # 1. Check if template has non-zero elements
    non_zero = count( !iszero, H_interaction_template )
    println( "Non-zero elements in H_template: $non_zero out of $( basis.n_tensors^2 )" )

    # 2. Check coupling between ground and excited state
    println( "\nTemplate coupling elements:" )
    println( "  H_template norm: ", LinearAlgebra.norm( H_interaction_template ) )

    # 3. Check peak field and Rabi frequency
    peak_field = fix_field_amplitude( pulse )
    println( "\nPeak field amplitude: $peak_field a.u." )
    println( "Reduced dipole: $d_reduced_val a.u." )
    println( "Peak Rabi frequency: $( abs( d_reduced_val * peak_field ) ) a.u." )
    println( "Rabi period: $( 2π / abs( d_reduced_val * peak_field ) ) a.u." )

    # 4. Print some matrix elements to verify structure
    println( "\nSample matrix elements:" )
    for i in 1:min( 10, basis.n_tensors )
        for j in 1:min( 10, basis.n_tensors )
            if abs( H_interaction_template[ i,j ] ) > 1e-10
                println( "  H[ $i,$j ] = ", H_interaction_template[ i,j ] )
            end
        end
    end

    # 5. Check initial density matrix
    println( "\nInitial density matrix ( first 10x10 ):" )
    for i in 1:min( 10, basis.n_tensors )
        for j in 1:min( 10, basis.n_tensors )
            if abs( ρ0[ i,j ] ) > 1e-10
                println( "  ρ0[ $i,$j ] = ", ρ0[ i,j ] )
            end
        end
    end

    # 6. Verify level indices
    println( "\nLevel indices:" )
    for ( i, idx_dict ) in enumerate( basis.level_indices )
        println( "  Level $i ( J=$( basis.level_J[ i ] ), E=$( basis.level_energies[ i ] ) ):" )
        for ( ( K,Q ), idx ) in sort( collect( idx_dict ), by=x->x[ 2 ] )
            println( "    K=$K, Q=$Q -> index $idx" )
        end
    end

    # 7. Check Hamiltonian at peak field
    H_peak = H0 + H_interaction_template .* peak_field
    println( "\nTotal H at peak ( first 10x10 ):" )
    for i in 1:min( 10, basis.n_tensors )
        row_str = ""
        for j in 1:min( 10, basis.n_tensors )
            if abs( H_peak[ i,j ] ) > 1e-10
                row_str *= " $( round( H_peak[ i,j ], digits=6 ) )"
            end
        end
        if row_str != ""
            println( "  Row $i:$row_str" )
        end
    end

    #  Run time evolution
    t_max = pulse.timeDelay + 2 * pulse.fwhm + 100.0
    dt = 0.1
    tspan = (0.0, t_max)
    println("\n⏱️ Running time evolution: t_max = $t_max, dt = $dt")
    times, ρ_history = solve_liouville_tensor(basis, H0, coupling_func, ρ0, tspan, dt)

    # ============================================================
    # 🟢 NOW process the results – ρ_history is defined here
    # ============================================================

    # Define J values as Float64
    J_g_val = levels[1].level.J.num / levels[1].level.J.den
    J_e_val = levels[2].level.J.num / levels[2].level.J.den
    idx_g00 = basis.level_indices[1][(0,0)]

    # Storage
    pop_ground = zeros(length(times))
    # Get m values for excited state (from first snapshot)
    m_vals, _ = populations_from_tensors(ρ_history[1], basis, 2, J_e_val)
    n_m = length(m_vals)
    pop_excited_m = zeros(length(times), n_m)

    # Loop over time steps
    for (i, ρ) in enumerate(ρ_history)
        # Ground state population
        pop_ground[i] = real(ρ[idx_g00, idx_g00]) / sqrt(2*J_g_val + 1)
        # Excited state m-resolved populations
        _, pops = populations_from_tensors(ρ, basis, 2, J_e_val)
        pop_excited_m[i, :] = pops
    end

    # Save to file
    data = hcat(times, pop_ground, pop_excited_m)
    writedlm("m_resolved_populations.dat", data, '\t')
    println("\n✅ Data saved to m_resolved_populations.dat")

    # (Optional) Print final populations
    println("\n=== Final Populations ===")
    println("  Ground: $(round(pop_ground[end], digits=6))")
    for (j, m) in enumerate(m_vals)
        println("  Excited m=$m: $(round(pop_excited_m[end,j], digits=6))")
    end # Time evolution


    # Save data: columns = time, ground_pop, pop(m=-J), pop(m=-J+1), ..., pop(m=+J)
    data = hcat(times, pop_ground, pop_excited_m)
    writedlm("stat_tensor_data.dat", data, '\t')   # or keep old filename
    println("\n✅ Data saved to stat_tensor_data.dat")

    # Print final populations
    println("\n=== Final Populations ===")
    println("  Ground: $(round(pop_ground[end], digits=6))")
    for (j, m) in enumerate(m_vals)
        println("  Excited m=$m: $(round(pop_excited_m[end, j], digits=6))")
    end
    # Also print total excited population (optional)
    total_excited = sum(pop_excited_m[end, :])
    println("  Total excited: $(round(total_excited, digits=6))")

    # Store in results dictionary
    results = Dict{String, Any}()
    results["times"] = times
    results["pop_ground"] = pop_ground
    results["pop_excited_m"] = pop_excited_m
    results["m_vals"] = m_vals
    # If you still want a combined "populations" array for compatibility, you can make one:
    populations_compat = hcat(pop_ground, sum(pop_excited_m, dims=2))
    results["populations"] = populations_compat   # optional

    # Debug prints (keep as is)
    gap = levels[2].level.energy - levels[1].level.energy
    println("DEBUG: Pulse Frequency  = $(pulse.omega)")
    println("DEBUG: Atomic Energy Gap = $gap")
    println("DEBUG: Detuning          = $(pulse.omega - gap)")

    return results
end
