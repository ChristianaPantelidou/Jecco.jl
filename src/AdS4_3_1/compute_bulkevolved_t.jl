
function compute_bulkevolved_t!(bulkevol_t::BulkEvolved,
                                bulkconstrain::BulkConstrained, gauge_t::Gauge,
                                bulkevol::BulkEvolved, boundary::Boundary,
                                gauge::Gauge, sys::System, ::EvolTest0)

    B1_t, G_t = unpack(bulkevol_t)
    # B1  , G   = unpack(bulkevol)

    fill!(B1_t,  0)
    fill!(G_t,   0)

    nothing
end


function compute_bulkevolved_t!(bulkevol_t::BulkEvolved,
                                bulkconstrain::BulkConstrained, gauge_t::Gauge,
                                bulkevol::BulkEvolved, boundary::Boundary,
                                gauge::Gauge, sys::System{Inner}, evoleq::AffineNull)
    uu  = sys.ucoord
    Du  = sys.Du
    Dx  = sys.Dx
    Dy  = sys.Dy


    Nu, Nx, Ny = size(sys)

    B1_t, G_t= unpack(bulkevol_t)

  
    # remaining inner grid points
    @fastmath @inbounds for j in 1:Ny
        @inbounds for i in 1:Nx
            xi    = gauge.xi[1,i,j]
            xi_t  = gauge_t.xi[1,i,j]
            @inbounds @simd for a in 1:Nu
                u      = uu[a]
                u3     = u * u * u 

                B1     = bulkevol.B1[a,i,j]
                G      = bulkevol.G[a,i,j]

                B1d    = bulkconstrain.B1d[a,i,j]
                Gd     = bulkconstrain.Gd[a,i,j]
                A      = bulkconstrain.A[a,i,j]

                B1_u   = Du(bulkevol.B1, a,i,j)
                G_u    = Du(bulkevol.G,  a,i,j)

		B1_t[a,i,j] = B1_u *
                               (-2 * u * u * xi_t + A * u3 +
                                (xi * u + 1) * (xi * u + 1))/2 +
                                B1d

		G_t[a,i,j] = G1_u *
                               (-2 * u * u * xi_t + A * u3 +
                                (xi * u + 1) * (xi * u + 1))/2 +
                                G1d
            end
        end
    end


    nothing
end


function compute_bulkevolved_t!(bulkevol_t::BulkEvolved,
                                bulkconstrain::BulkConstrained, gauge_t::Gauge,
                                bulkevol::BulkEvolved, boundary::Boundary,
                                gauge::Gauge, sys::System{Outer}, evoleq::AffineNull)
    uu  = sys.ucoord
    Du  = sys.Du
    Dx  = sys.Dx
    Dy  = sys.Dy

    Nu, Nx, Ny = size(sys)

    B1_t, G_t = unpack(bulkevol_t)

    @fastmath @inbounds for j in 1:Ny
        @inbounds for i in 1:Nx
            xi_t  = gauge_t.xi[1,i,j]
            @inbounds @simd for a in 1:Nu
                u      = uu[a]
                u2     = u * u

                B1d    = bulkconstrain.B1d[a,i,j]
                Gd     = bulkconstrain.Gd[a,i,j]
                A      = bulkconstrain.A[a,i,j]

                B1_u   = Du(bulkevol.B1, a,i,j)
                G_u    = Du(bulkevol.G,  a,i,j)

		B1_t[a,i,j] = B1d + u2 * (A/2 - xi_t) * B1_u
		G_t[a,i,j]  = Gd  + u2 * (A/2 - xi_t) * G_u
            end
        end
    end

    nothing
end

function sync_bulkevolved!(bulkevols_t, bulkconstrains, gauge_t::Gauge,
                           systems::SystemPartition, evoleq::AffineNull)
    Nsys = length(systems)

    @inbounds for i in 1:Nsys-1
        sync_bulkevolved!(bulkevols_t[i], bulkevols_t[i+1], bulkconstrains[i+1],
                          gauge_t, systems[i], systems[i+1], evoleq)
    end

    nothing
end

function sync_bulkevolved!(bulkevol1_t::BulkEvolved, bulkevol2_t::BulkEvolved,
                           bulkconstrain2::BulkConstrained, gauge_t::Gauge,
                           sys1::System{Outer}, sys2::System{Outer}, evoleq::AffineNull)
    u0 = sys2.ucoord[1]

    _, Nx, Ny = size(sys2)

    B11_t, G1_t = unpack(bulkevol1_t)
    B12_t, G2_t = unpack(bulkevol2_t)

    @fastmath @inbounds for j in 1:Ny
        @inbounds for i in 1:Nx
            xi_t   = gauge_t.xi[1,i,j]
            A      = bulkconstrain2.A[1,i,j]

            # characteristic speed
            c = u0 * u0 * (A/2 - xi_t)

            # if c > 0, mode is entering grid1 from grid2;
            # if c < 0, mode is entering grid2 from grid1.
            # we assume here that grids merely touch at the interface
            if c > 0
                B11_t[end,i,j]  = B12_t[1,i,j]
                G1_t[end,i,j]   = G2_t[1,i,j]
            elseif c < 0
                B12_t[1,i,j]  = B11_t[end,i,j]
                G2_t[1,i,j]   = G1_t[end,i,j]
            end
        end
    end

    nothing
end

function sync_bulkevolved!(bulkevol1_t::BulkEvolved, bulkevol2_t::BulkEvolved,
                           bulkconstrain2::BulkConstrained, gauge_t::Gauge,
                           sys1::System{Inner}, sys2::System{Outer}, evoleq::AffineNull)
    u0  = sys2.ucoord[1]
    u02 = u0 * u0
    u03 = u0 * u02
    u04 = u02 * u02

    _, Nx, Ny = size(sys2)

    B11_t, G1_t = unpack(bulkevol1_t)
    B12_t, G2_t = unpack(bulkevol2_t)

    @fastmath @inbounds for j in 1:Ny
        @inbounds for i in 1:Nx
            xi_t   = gauge_t.xi[1,i,j]
            A      = bulkconstrain2.A[1,i,j]

            # characteristic speed
            c = u0 * u0 * (A/2 - xi_t)

            # if c > 0, mode is entering grid1 from grid2;
            # if c < 0, mode is entering grid2 from grid1.
            # we assume here that grids merely touch at the interface
            if c > 0
                B11_t[end,i,j]  = B12_t[1,i,j] / u03
                G1_t[end,i,j]   = G2_t[1,i,j] / u03
            elseif c < 0
                B12_t[1,i,j]  = u03 * B11_t[end,i,j]
                G2_t[1,i,j]   = u03 * G1_t[end,i,j]
            end
        end
    end


    nothing
end
