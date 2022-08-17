
Base.@kwdef struct BlackBrane_xi1{T} <: InitialData
    a30           :: T   = 1.0
    # guess for the AH position
    AH_pos        :: T   = 1.0
    xi_0          :: T   = 0.0
    xi_Ax         :: T   = 0.0
    xi_nx         :: Int = 1
    xi_Ay         :: T   = 0.0
    xi_ny         :: Int = 1
    xmax          :: T
    xmin          :: T
    ymax          :: T
    ymin          :: T
    ahf           :: AHF = AHF()
end


abstract type ID_ConstantAH  <: InitialData end

Base.@kwdef struct BlackBrane{T} <: ID_ConstantAH
    energy_dens   :: T   = 1.0
    AH_pos        :: T   = 1.0
    ahf           :: AHF = AHF()
end

Base.@kwdef struct BlackBranePert{T} <: ID_ConstantAH
    energy_dens   :: T   = 1.0
    AH_pos        :: T   = 1.0
    B1_amp        :: T   = 0.0
    B1_nx         :: Int = 1
    B1_ny         :: Int = 2
    G_amp         :: T   = 0.0
    G_nx          :: Int = 1
    G_ny          :: Int = 2
    a3_ampx       :: T   = 0.0
    a3_ampy       :: T   = 0.0
    a3_kx         :: Int = 1
    a3_ky         :: Int = 1
    fx1_ampx      :: T   = 0.0
    fx1_ampy      :: T   = 0.0
    fx1_kx        :: Int = 1
    fx1_ky        :: Int = 1
    fy1_ampx      :: T   = 0.0
    fy1_ampy      :: T   = 0.0
    fy1_kx        :: Int = 1
    fy1_ky        :: Int = 1
    xi0           :: T   = 0.0
    xmax          :: T
    xmin          :: T
    ymax          :: T
    ymin          :: T
    ahf           :: AHF = AHF()
end


Base.@kwdef struct QNM_1D{T} <: InitialData
    energy_dens :: T   = 1.0
    AH_pos      :: T   = 1.0
    ahf         :: AHF = AHF()
end


function (id::InitialData)(bulkconstrains, bulkevols, bulkderivs, boundary::Boundary,
                           gauge::Gauge, horizoncache::HorizonCache, systems::SystemPartition,
                           evoleq::EvolutionEquations)
    _, Nx, Ny = size(systems[end])
    AH_pos    = id.AH_pos
    xi        = getxi(gauge)

    # function to solve the nested system
    nested = Nested(systems, bulkconstrains)

    init_data!(boundary, systems[1],   id)

    init_data!(gauge,    systems[end], id)
    init_data!(bulkevols, gauge, systems, id)

    # solve nested system for the constrained variables
    nested(bulkevols, boundary, gauge, evoleq)

    # find the Apparent Horizon
    sigma = similar(gauge.xi)
    fill!(sigma, 1/AH_pos)  # initial guess
    find_AH!(sigma, bulkconstrains[end], bulkevols[end], bulkderivs[end], gauge,
             horizoncache, systems[end], id.ahf)

    nothing
end



function (id::ID_ConstantAH)(bulkconstrains, bulkevols, bulkderivs, boundary::Boundary,
                           gauge::Gauge, horizoncache::HorizonCache, systems::SystemPartition,
                           evoleq::EvolutionEquations)
    _, Nx, Ny = size(systems[end])
    AH_pos    = id.AH_pos
    xi        = getxi(gauge)

    # function to solve the nested system
    nested = Nested(systems, bulkconstrains)

    init_data!(boundary, systems[1],   id)

    init_data!(gauge,    systems[end], id)
    init_data!(bulkevols, gauge, systems, id)

    # solve nested system for the constrained variables
    nested(bulkevols, boundary, gauge, evoleq)

    # find the Apparent Horizon
    sigma = similar(gauge.xi)
    fill!(sigma, 1/AH_pos)  # initial guess
    find_AH!(sigma, bulkconstrains[end], bulkevols[end], bulkderivs[end], gauge,
             horizoncache, systems[end], id.ahf)

    # assuming that the AH has been found, we now update xi and the bulk variables
    for j in 1:Ny
        for i in 1:Nx
            xi[1,i,j] += -1 / AH_pos + sigma[1,i,j]
        end
    end

    init_data!(bulkevols, gauge, systems, id)

    # solve nested system for the constrained variables
    nested(bulkevols, boundary, gauge, evoleq)

    # AH should now be at u = AH_pos

    nothing
end

function init_data!(bulkevols, gauge::Gauge, systems::SystemPartition,
                    id::InitialData)
    # the Ref() makes its argument a scalar with respect to broadcast
    init_data!.(bulkevols, Ref(gauge), systems, Ref(id))
end

function init_data!(bulk::BulkEvolved, gauge::Gauge, sys::System{Inner},
                    id::InitialData)
    Nu, Nx, Ny = size(sys)
    xx = sys.xcoord
    yy = sys.ycoord
    uu = sys.ucoord

    B1  = getB1(bulk)
    G   = getG(bulk)
    xi  = getxi(gauge)


    for j in 1:Ny
        for i in 1:Nx
            for a in 1:Nu
                u       = uu[a]
                x       = xx[i]
                y       = yy[j]
                xi_ij   = xi[1,i,j]
                aux     = 1 + xi_ij * u
                aux3    = aux * aux * aux
                aux4    = aux * aux3
                u_old   = u / aux
                B1_old  = analytic_B1(u_old, x, y, id)
                G_old   = analytic_G(u_old, x, y, id)

                B1[a,i,j]  = B1_old / aux4
                G[a,i,j]   = G_old  / aux4
            end
        end
    end


    bulk
end

function init_data!(bulk::BulkEvolved, gauge::Gauge, sys::System{Outer},
                    id::InitialData)
    Nu, Nx, Ny = size(sys)
    xx = sys.xcoord
    yy = sys.ycoord
    uu = sys.ucoord

    B1  = getB1(bulk)
    G   = getG(bulk)
    xi  = getxi(gauge)

    for j in 1:Ny
        for i in 1:Nx
            for a in 1:Nu
                u         = uu[a]
                x         = xx[i]
                y         = yy[j]
                xi_ij     = xi[1,i,j]
                aux       = 1 + xi_ij * u
                aux3      = aux * aux * aux
                aux4      = aux * aux3
                u_old     = u / aux
                B1_old    = analytic_B1(u_old, x, y, id)
                G_old     = analytic_G(u_old, x, y, id)
                B1_inner  = B1_old / aux4
                G_inner   = G_old  / aux4

                B1[a,i,j]  = u^3 * B1_inner
                G[a,i,j]   = u^3 * G_inner
            end
        end
    end

    

    bulk
end


# BlackBrane_xi1

analytic_B1(u, x, y, id::BlackBrane_xi1)  = 0
analytic_G(u, x, y, id::BlackBrane_xi1)   = 0

function init_data!(ff::Boundary, sys::System, id::BlackBrane_xi1)
    a30 = id.a30

    a3  = geta3(ff)
    fx1 = getfx1(ff)
    fy1 = getfy1(ff)

    fill!(a3, a30)
    fill!(fx1, 0)
    fill!(fy1, 0)

    ff
end

function init_data!(ff::Gauge, sys::System, id::BlackBrane_xi1)
    _, Nx, Ny = size(sys)
    xx = sys.xcoord
    yy = sys.ycoord
    xi  = getxi(ff)

    xmax  = id.xmax
    xmin  = id.xmin
    ymax  = id.ymax
    ymin  = id.ymin

    for j in 1:Ny
        for i in 1:Nx
            x         = xx[i]
            y         = yy[j]

            xi[1,i,j] = id.xi_0 +
                id.xi_Ax * sin( 2 * π * id.xi_nx * (xmax-x)/(xmax-xmin) ) +
                id.xi_Ay * sin( 2 * π * id.xi_ny * (ymax-y)/(ymax-ymin) )
        end
    end

    ff
end


# BlackBrane initial data

analytic_B1(u, x, y, id::BlackBrane)  = 0
analytic_G(u, x, y, id::BlackBrane)   = 0

function init_data!(ff::Boundary, sys::System, id::BlackBrane)
    a30 = -id.energy_dens/2

    a3  = geta3(ff)
    fx1 = getfx1(ff)
    fy1 = getfy1(ff)

    fill!(a3, a30)
    fill!(fx1, 0)
    fill!(fy1, 0)

    ff
end

function init_data!(ff::Gauge, sys::System, id::BlackBrane)
    a30     = -id.energy_dens/2
    AH_pos  = id.AH_pos
    xi0     = (-a30)^0.25 - 1/AH_pos

    xi  = getxi(ff)

    fill!(xi, xi0)

    ff
end


# BlackBranePert initial data


function analytic_B1(u, x, y, id::BlackBranePert)
    # add the perturbation on B1
    pert_amp = id.B1_amp
    xmax     = id.xmax
    xmin     = id.xmin
    ymax     = id.ymax
    ymin     = id.ymin
    # number of maxima in each direction
    nx       = id.B1_nx
    ny       = id.B1_ny

    pert_amp * sin( 2 * π * nx * (xmax-x)/(xmax-xmin) ) *
        sin( -2 * π * ny * (ymax-y)/(ymax-ymin) )
end


function analytic_G(u, x, y, id::BlackBranePert)
    # add the perturbation on G
    pert_amp = id.G_amp
    xmax     = id.xmax
    xmin     = id.xmin
    ymax     = id.ymax
    ymin     = id.ymin
    # number of maxima in each direction
    nx       = id.G_nx
    ny       = id.G_ny

    pert_amp * sin( 2 * π * nx * (xmax-x)/(xmax-xmin) ) *
        sin( -2 * π * ny * (ymax-y)/(ymax-ymin) )
end

function init_data!(ff::Boundary, sys::System{Inner}, id::BlackBranePert)
    epsilon = id.energy_dens

    # a3 perturbation amplitude
    ampx     = id.a3_ampx
    ampy     = id.a3_ampy
    fx1_ampx = id.fx1_ampx
    fx1_ampy = id.fx1_ampy
    fy1_ampx = id.fy1_ampx
    fy1_ampy = id.fy1_ampy
    # number of maxima
    kx     = id.a3_kx
    ky     = id.a3_ky
    fx1_kx = id.fx1_kx
    fx1_ky = id.fx1_ky
    fy1_kx = id.fy1_kx
    fy1_ky = id.fy1_ky
    
    xmax = id.xmax
    xmin = id.xmin
    xmid = (xmax + xmin) / 2
    ymax = id.ymax
    ymin = id.ymin
    ymid = (ymax + ymin) / 2

    _, Nx, Ny = size(sys)
    xx = sys.xcoord
    yy = sys.ycoord

    a30   = (-epsilon) / 2

    a3  = geta3(ff)
    fx1 = getfx1(ff)
    fy1 = getfy1(ff)

    fill!(a3, a30)
    fill!(fx1, 0)
    fill!(fy1, 0)

    for j in 1:Ny
        for i in 1:Nx
            x = xx[i]
            y = yy[j]
            a3[1,i,j]  += -a30 * ( ampx * cos(2 * π * kx * (x-xmid)/(xmax-xmin)) +
                                   ampy * cos(2 * π * ky * (y-ymid)/(ymax-ymin)) )

            fx1[1,i,j] += fx1_ampx * cos(2 * π * fx1_kx * (x-xmid)/(xmax-xmin)) +
                           fx1_ampy * cos(2 * π * fx1_ky * (y-ymid)/(ymax-ymin))

            fy1[1,i,j] += fy1_ampx * cos(2 * π * fy1_kx * (x-xmid)/(xmax-xmin)) +
                          fy1_ampy * cos(2 * π * fy1_ky * (y-ymid)/(ymax-ymin))
        end
    end

    ff
end

function init_data!(ff::Gauge, sys::System, id::BlackBranePert)
    a30     = -id.energy_dens/2
    AH_pos  = id.AH_pos

    # TODO: this guess works best for the conformal case. is there a better one?
    if id.xi0 == 0
        xi0 = (-a30)^0.25 - 1/AH_pos
    else
        xi0 = id.xi0
    end

    xi  = getxi(ff)

    fill!(xi, xi0)

    ff
end

analytic_B1(u, x, y) = 0
analytic_G(u, x, y)  = 0

function init_data!(ff::Boundary, sys::System)
    a3  = geta3(ff)
    fx1 = getfx1(ff)
    fy1 = getfy1(ff)

    epsilon = id.energy_dens


    a30 = (-epsilon) / 2

    fill!(a3, a30)
    fill!(fx1, 0)
    fill!(fy1, 0)

    ff
end

function init_data!(ff::Gauge, sys::System)
    epsilon = id.energy_dens
    AH_pos  = id.AH_pos

    a30 = (-epsilon) / 2

    # TODO: this guess works best for the conformal case. is there a better one?
    xi0 = (-a30)^0.25 - 1/AH_pos

    xi  = getxi(ff)

    fill!(xi, xi0)

    ff
end


#QNM in 1D initial data
analytic_B1(u, x, y, id::QNM_1D)  = 3/2*0.1*u^8
analytic_G(u, x, y, id::QNM_1D)   = 0

function init_data!(ff::Boundary, sys::System, id::QNM_1D)
    a3  = geta3(ff)
    fx1 = getfx1(ff)
    fy1 = getfy1(ff)

    epsilon = id.energy_dens

    a30 = (-epsilon) / 2

    fill!(a3, a30)
    fill!(fx1, 0)
    fill!(fy1, 0)

    ff
end

function init_data!(ff::Gauge, sys::System, id::QNM_1D)
    epsilon = id.energy_dens
    AH_pos  = id.AH_pos

    a30 = (-epsilon) / 2

    xi0 = 0

    xi  = getxi(ff)

    fill!(xi, xi0)

    ff
end
