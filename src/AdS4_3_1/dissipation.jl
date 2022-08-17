
function apply_dissipation_3D!(var_out, var, sys::System)
    Nu, Nx, Ny = size(sys)

    DKOx = sys.DKOx
    DKOy = sys.DKOy

    copyto!(var_out, var)

    @fastmath @inbounds @threads for j in 1:Ny
        @inbounds for i in 1:Nx
            @inbounds for a in 1:Nu
                var_out[a,i,j] += DKOx(var, a,i,j) + DKOy(var, a,i,j)
            end
        end
    end

    nothing
end

function apply_dissipation_2D!(var_out, var, sys::System)
    _, Nx, Ny = size(sys)

    DKOx = sys.DKOx
    DKOy = sys.DKOy

    copyto!(var_out, var)

    @fastmath @inbounds @threads for j in 1:Ny
        @inbounds for i in 1:Nx
            var_out[1,i,j] += DKOx(var, 1,i,j) + DKOy(var, 1,i,j)
        end
    end

    nothing
end

function apply_dissipation!(boundary::Boundary, cache::Boundary, sys::System)
    a3  = geta3(boundary)
    fx1 = getfx1(boundary)
    fy1 = getfy1(boundary)

    a3_cache  = geta3(cache)
    fx1_cache = getfx1(cache)
    fy1_cache = getfy1(cache)

    # the loops in these functions are threaded, so it's probably not worth it
    # to @spawn here
    apply_dissipation_2D!( a3_cache,  a3, sys)
    apply_dissipation_2D!(fx1_cache, fx1, sys)
    apply_dissipation_2D!(fy1_cache, fy1, sys)

    copyto!( a3,  a3_cache)
    copyto!(fx1, fx1_cache)
    copyto!(fy1, fy1_cache)

    nothing
end

function apply_dissipation!(gauge::Gauge, cache::Gauge, sys::System)
    xi = getxi(gauge)
    xi_cache = getxi(cache)

    apply_dissipation_2D!(xi_cache, xi, sys)
    copyto!(xi, xi_cache)

    nothing
end

function apply_dissipation!(bulkevol::BulkEvolved, cache::BulkEvolved,
                            sys::System)
    B1  = getB1(bulkevol)
    G   = getG(bulkevol)

    B1_cache  = getB1(cache)
    G_cache   = getG(cache)


    # the loops in these functions are threaded, so it's probably not worth it
    # to @spawn here
    apply_dissipation_3D!( B1_cache,  B1, sys)
    apply_dissipation_3D!(  G_cache,   G, sys)


    copyto!( B1,  B1_cache)
    copyto!(  G,   G_cache)

    nothing
end

# exponential filtering
function (filters::Filters)(bulkevol::BulkEvolved)
    @sync begin
        @spawn filters.exp_filter(bulkevol.B1)
        @spawn filters.exp_filter(bulkevol.G)
    end
    nothing
end
