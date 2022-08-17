
function compute_boundary_t!(boundary_t::Boundary, bulkevol::BulkEvolved,
                             boundary::Boundary, gauge::Gauge, sys::System, ::EvolTest0)

    a3_t, fx1_t, fy1_t = unpack(boundary_t)
    # a3  , fx1  , fy1   = unpack(boundary)

    fill!(a3_t,  0)
    fill!(fx1_t, 0)
    fill!(fy1_t, 0)

    nothing
end

function compute_boundary_t!(boundary_t::Boundary, bulk::BulkEvolved,
                             boundary::Boundary, gauge::Gauge, sys::System{Inner},
                             evoleq::AffineNull)
    Du  = sys.Du
    Dx  = sys.Dx
    Dy  = sys.Dy


    _, Nx, Ny = size(sys)

    a3_t, fx1_t, fy1_t = unpack(boundary_t)

    @fastmath @inbounds for j in 1:Ny
        @inbounds for i in 1:Nx
            xi      = gauge.xi[1,i,j]
            xi3     = xi*xi*xi

            xi_x    = Dx(gauge.xi, 1,i,j)
            xi_y    = Dy(gauge.xi, 1,i,j)

            b13_x   = Dx(bulk.B1, 1,i,j)
            b13_y   = Dy(bulk.B1, 1,i,j)

            g3_x    = Dx(bulk.G, 1,i,j)
            g3_y    = Dy(bulk.G, 1,i,j)

            a3_x    = Dx(boundary.a3, 1,i,j)
            a3_y    = Dy(boundary.a3, 1,i,j)

            fx1_x   = Dx(boundary.fx1, 1,i,j)
            fy1_y   = Dy(boundary.fy1, 1,i,j)

            a3_t[1,i,j]  = -3//2 * (fx1_x + fy1_y)

            fx1_t[1,i,j] = g3_y +b13_x - 1//3 * a3_x 
            fy1_t[1,i,j] = g3_x - b13_y - 1//3 * a3_y 
        end
    end

    nothing
end
