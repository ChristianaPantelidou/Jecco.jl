
"""
    compute_energy(a3)

Returns the normalized energy density ``ϵ``.
"""
function compute_energy(a3)
    return -2 * a3
   
end

"""
    compute_flux(f2)

Returns the normalized flux ``Ttx``.
"""
function compute_flux(f2)
    return 3*f2 
end

"""
    compute_pressure_x(a3, b13)

Returns the normalized pressure in the x direction ``pₓ``.
"""
function compute_pressure_x(a3, b13)
    -a3+3*b13
end

"""
    compute_pressure_y(a3, b13)

Returns the normalized pressure in the y direction ``p_y``.
"""
function compute_pressure_y(a3, b13)
    return -a3/3-b13
end


"""
    compute_pressure_xy(a3)

Returns the normalized pressure in the xy direction ``p_xy``.
"""
function compute_pressure_xy(g3)
        return 3* g3
end


function compute_temperature(Du_A_uAH, uAH)
        return (-0.5 * uAH .^ 2 .* Du_A_uAH)/(2*pi)
end

function compute_entropy(S_uAH, uAH)
        return pi*S_uAH.^2
end

function get_energy(ts::OpenPMDTimeSeries; it::Int, verbose::Bool=false)
    a3,   chart = get_field(ts, it=it, field="a3", verbose=verbose)
    
    en = compute_energy(a3)

    en, chart
end

function get_Jx(ts::OpenPMDTimeSeries; it::Int, verbose::Bool=false)
    fx1, chart = get_field(ts, it=it, field="fx1", verbose=verbose)

    Jx = compute_flux(fx1)

    Jx, chart
end

function get_Jy(ts::OpenPMDTimeSeries; it::Int, verbose::Bool=false)
    fy1, chart = get_field(ts, it=it, field="fy1", verbose=verbose)


    Jy = compute_flux(fy1)

    Jy, chart
end

function get_px(ts::OpenPMDTimeSeries; it::Int, verbose::Bool=false)
    a3,   chart = get_field(ts, it=it, field="a3", verbose=verbose)
    b13,  chart = get_field(ts, it=it, field="b13", verbose=verbose)


    px = compute_pressure_x(a3, b13)

    px, chart
end

function get_py(ts::OpenPMDTimeSeries; it::Int, verbose::Bool=false)
    a3,   chart = get_field(ts, it=it, field="a3", verbose=verbose)
    b13,  chart = get_field(ts, it=it, field="b13", verbose=verbose)

    py = compute_pressure_y(a3, b13)

    py, chart
end


function get_pxy(ts::OpenPMDTimeSeries; it::Int, verbose::Bool=false)
    g3,   chart = get_field(ts, it=it, field="g3", verbose=verbose)


    pxy = compute_pressure_xy(g3)

    pxy, chart
end


#Remember that when doing get_field with a given it we always take the file whose it is the closest to the specified one.
function get_temperature(ts_const::OpenPMDTimeSeries, ts_diag::OpenPMDTimeSeries; it::Int, verbose::Bool=false)
    cmax = 0
    i    = 0

    sigma, _ = get_field(ts_diag, it=it, field="sigma", verbose=verbose)
    uAH      = 1 ./sigma

    while cmax == 0
        i += 1
        try #We find what is the deepest grid.
            get_field(ts_const, it=it, field="A c=$i", verbose=verbose)
        catch
            cmax = i-1
            @warn "maximum value for c is $(i-1)"
        end
    end
    A, chart = get_field(ts_const, it=it, field="A c=$cmax", verbose=verbose)
    u, x, y  = chart[:]
    Nx       = length(x)
    Ny       = length(y)
    Nu       = length(u)
    der      = ChebDeriv(1,u[1],u[end],length(u))
    Du       = der.D
    uinterp  = ChebInterpolator(u[1],u[end],length(u))
    Du_A_uAH = similar(sigma)
    for j in 1:Ny
        for i in 1:Nx
            Du_A = Du*A[:,i,j]
            Du_A_uAH[1,i,j] = uinterp(view(Du_A,:))(uAH[1,i,j])
        end
    end

    T = compute_temperature(Du_A_uAH, uAH)

    T, chart
end


#Here it interpolates S over u, and evaluated at u_AH for every i,j
function get_entropy(ts_const::OpenPMDTimeSeries, ts_diag::OpenPMDTimeSeries; it::Int, verbose::Bool=false)
    cmax = 0
    i    = 0

    sigma, _ = get_field(ts_diag, it=it, field="sigma", verbose=verbose)
    uAH      = 1 ./sigma


    while cmax == 0
        i += 1
        try #We find what is the deepest grid.
            get_field(ts_const, it=it, field="S c=$i", verbose=verbose)
        catch
            cmax = i-1
            @warn "maximum value for c is $(i-1)"
        end
    end

    S, chart = get_field(ts_const, it=it, field="S c=$cmax", verbose=verbose)
    u, x, y = chart[:]
    Nx      = length(x)
    Ny      = length(y)
    Nu      = length(u)
    uinterp = ChebInterpolator(u[1],u[end],length(u))
    S_uAH   = similar(sigma)
    for j in 1:Ny
        for i in 1:Nx
            S_uAH[1,i,j] = uinterp(view(S, :,i,j))(uAH[1,i,j])
        end
    end
    entr = compute_entropy(S_uAH, uAH)

    entr, chart
end

#=
Computes the fluid velocity and the local frame (diagonal) stress tensor. I think that
ut will always be 1, if not we can always hand ux/ut and uy/ut, which directly are the speeds measured
in the LAB frame, i.e. dx/dt and dy/dt.
=#
function compute_local_VEVs(e, Jx, Jy, px, py, pxy)
    T = [e -Jx -Jy; -Jx px pxy; -Jy pxy py]
    values, vectors = eigen(T)
    norms = -vectors[1,:].^2 + vectors[2,:].^2 + vectors[3,:].^2
    index = findfirst(==(1), norms.<0)
    ut = sign(vectors[1,index])*vectors[1,index]/(-norms[index])^0.5
    ux = sign(vectors[1,index])*vectors[2,index]/(-norms[index])^0.5
    uy = sign(vectors[1,index])*vectors[3,index]/(-norms[index])^0.5
    e_local = values[index]
    p1_local = values[1:end .!=index][1]
    p2_local = values[1:end .!=index][2]

    return ut, ux, uy, e_local, p1_local, p2_local
end
