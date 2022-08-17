
function output_writer(u::EvolVars, chart2D::Chart, atlas, tinfo::Jecco.TimeInfo,
                       io::InOut, potential::Potential)
    Nsys = length(atlas)

    # output structures
    out_bdry  = Jecco.Output(io.out_dir, "boundary_", tinfo)
    out_gauge = Jecco.Output(io.out_dir, "gauge_", tinfo)
    out_bulk  = Jecco.Output(io.out_dir, "bulk_", tinfo)

    boundary  = getboundary(u)
    gauge     = getgauge(u)
    bulkevols = getbulkevolvedpartition(u)

    # NamedTuple with potential parameters
    params = parameters(potential)

    #=
    for analysis, it's useful to have the boundary behaviour of the bulk fields,
    so let's output those as well in the boundary file. note that the slicing
    operation [1,:,:] converts 3D arrays into 2D arrays; since we want the same
    shape as in the remaining boundary fields -- 3D arrays of size (1,Nx,Ny) --
    we must reshape them first.
    =#
    b13  = reshape(bulkevols[1].B1[1,:,:],  size(chart2D))
    g3   = reshape(bulkevols[1].G[1,:,:],   size(chart2D))

    # output fields
    boundary_fields = (
        Jecco.Field("a3",   boundary.a3,  chart2D),
        Jecco.Field("fx1",  boundary.fx1, chart2D),
        Jecco.Field("fy1",  boundary.fy1, chart2D),
        Jecco.Field("b13",  b13,          chart2D),
        Jecco.Field("g3",   g3,           chart2D),
    )
    gauge_fields = Jecco.Field("xi", gauge.xi, chart2D)
    bulkevols_fields = ntuple(i -> (
        Jecco.Field("B1 c=$i",  bulkevols[i].B1,  atlas[i]),
        Jecco.Field("G c=$i",   bulkevols[i].G,   atlas[i])
    ), Nsys)

    last_output_boundary_t = -io.out_boundary_every_t
    last_output_gauge_t    = -io.out_gauge_every_t
    last_output_bulk_t     = -io.out_bulk_every_t

    function (u::EvolVars)
        boundary  = getboundary(u)
        gauge     = getgauge(u)
        bulkevols = getbulkevolvedpartition(u)

        it = tinfo.it
        tt = tinfo.t

        do_output_boundary = false
        do_output_gauge    = false
        do_output_bulk     = false

        if (io.out_boundary_every > 0 && it % io.out_boundary_every == 0)
            do_output_boundary = true
        end
        if (io.out_gauge_every > 0 && it % io.out_gauge_every == 0)
            do_output_gauge = true
        end
        if (io.out_bulk_every > 0 && it % io.out_bulk_every == 0)
            do_output_bulk = true
        end

        if (io.out_boundary_every_t > 0 &&
            tt >= last_output_boundary_t + io.out_boundary_every_t - 1e-12)
            do_output_boundary     = true
            last_output_boundary_t = tt
        end
        if (io.out_gauge_every_t > 0 &&
            tt >= last_output_gauge_t + io.out_gauge_every_t - 1e-12)
            do_output_gauge     = true
            last_output_gauge_t = tt
        end
        if (io.out_bulk_every_t > 0 &&
            tt >= last_output_bulk_t + io.out_bulk_every_t - 1e-12)
            do_output_bulk     = true
            last_output_bulk_t = tt
        end

        if do_output_boundary
            boundary_fields[1].data = boundary.a3
            boundary_fields[2].data = boundary.fx1
            boundary_fields[3].data = boundary.fy1
            @views copyto!(boundary_fields[4].data, bulkevols[1].B1[1,:,:])
            @views copyto!(boundary_fields[6].data, bulkevols[1].G[1,:,:])


            # write data
            out_bdry(boundary_fields, params=params)
        end

        if do_output_gauge
            gauge_fields.data = gauge.xi
            # write data
            out_gauge(gauge_fields, params=params)
        end

        if do_output_bulk
            @inbounds for i in 1:Nsys
                bulkevols_fields[i][1].data = bulkevols[i].B1
                bulkevols_fields[I][2].data = bulkevols[i].G
            end
            # write data
            out_bulk.(bulkevols_fields, params=params)
        end

        nothing
    end
end


function output_writer(bulkconstrains::BulkPartition{Nsys,BulkConstrained{T}}, atlas,
                       tinfo::Jecco.TimeInfo, io::InOut, potential::Potential,
                       ) where {Nsys,T}
    @assert Nsys == length(atlas)

    # NamedTuple with potential parameters
    params = parameters(potential)

    # output structure
    out  = Jecco.Output(io.out_dir, "constrained_", tinfo)

    # output fields
    fields = ntuple(i -> (
        Jecco.Field("S c=$i",    bulkconstrains[i].S,    atlas[i]),
        Jecco.Field("Fx c=$i",   bulkconstrains[i].Fx,   atlas[i]),
        Jecco.Field("Fy c=$i",   bulkconstrains[i].Fy,   atlas[i]),
        Jecco.Field("B1d c=$i",  bulkconstrains[i].B1d,  atlas[i]),
        Jecco.Field("Gd c=$i",   bulkconstrains[i].Gd,   atlas[i]),
        Jecco.Field("Sd c=$i",   bulkconstrains[i].Sd,   atlas[i]),
        Jecco.Field("A c=$i",    bulkconstrains[i].A,    atlas[i])
    ), Nsys)

    last_output_t = -io.out_bulkconstrained_every_t

    function (bulkconstrains)
        it = tinfo.it
        tt = tinfo.t

        do_output = false

        if (io.out_bulkconstrained_every > 0 && it % io.out_bulkconstrained_every == 0)
            do_output = true
        end

        if (io.out_bulkconstrained_every_t > 0 &&
            tt >= last_output_t + io.out_bulkconstrained_every_t - 1e-12)
            do_output = true
            last_output_t = tt
        end

        if do_output
            @inbounds for i in 1:Nsys
                fields[i][1].data = bulkconstrains[i].S
                fields[i][2].data = bulkconstrains[i].Fx
                fields[i][3].data = bulkconstrains[i].Fy
                fields[i][4].data = bulkconstrains[i].B1d
                fields[I][5].data = bulkconstrains[i].Gd
                fields[I][6].data = bulkconstrains[i].Sd
                fields[I][7].data = bulkconstrains[i].A
            end
            # write data
            out.(fields, params=params)
        end

        nothing
    end
end


function checkpoint_writer(u::EvolVars, chart2D::Chart, atlas, tinfo::Jecco.TimeInfo,
                           io::InOut)
    Nsys = length(atlas)

    # output structure
    out  = Jecco.Output(io.checkpoint_dir, "checkpoint_it", tinfo)

    boundary  = getboundary(u)
    gauge     = getgauge(u)
    bulkevols = getbulkevolvedpartition(u)

    # output fields
    boundary_fields = (
        Jecco.Field("a3",   boundary.a3,  chart2D),
        Jecco.Field("fx1",  boundary.fx1, chart2D),
        Jecco.Field("fy1",  boundary.fy1, chart2D),
    )
    gauge_fields = Jecco.Field("xi", gauge.xi, chart2D)
    bulkevols_fields = ntuple(i -> (
        Jecco.Field("B1 c=$i",  bulkevols[i].B1,  atlas[i]),
        Jecco.Field("G c=$i",   bulkevols[i].G,   atlas[i]),
    ), Nsys)

    function (u::EvolVars)
        boundary  = getboundary(u)
        gauge     = getgauge(u)
        bulkevols = getbulkevolvedpartition(u)

        boundary_fields[1].data = boundary.a3
        boundary_fields[2].data = boundary.fx1
        boundary_fields[3].data = boundary.fy1

        gauge_fields.data = gauge.xi

        @inbounds for i in 1:Nsys
            bulkevols_fields[i][1].data = bulkevols[i].B1
            bulkevols_fields[I][2].data = bulkevols[i].G
        end

        # write data
        out(boundary_fields)
        out(gauge_fields)
        out.(bulkevols_fields)

        nothing
    end
end


# restore all bulk evolved fields
function restore!(bulkevols::BulkPartition{Nsys}, ts::OpenPMDTimeSeries,
                  it::Int) where {Nsys}
    for i in 1:Nsys
        B1, chart = get_field(ts, it=it, field="B1 c=$i")
        @assert size(B1) == size(bulkevols[i].B1)
        copyto!(bulkevols[i].B1, B1)

        G, chart = get_field(ts, it=it, field="G c=$i")
        @assert size(G) == size(bulkevols[i].G)
        copyto!(bulkevols[i].G, G)

    end

    nothing
end

# restore all boundary fields
function restore!(boundary::Boundary, ts::OpenPMDTimeSeries, it::Int)

    a3, chart = get_field(ts, it=it, field="a3")
    @assert size(a3) == size(boundary.a3)
    copyto!(boundary.a3, a3)

    fx1, chart = get_field(ts, it=it, field="fx1")
    @assert size(fx1) == size(boundary.fx1)
    copyto!(boundary.fx1, fx1)

    fy1, chart = get_field(ts, it=it, field="fy1")
    @assert size(fy1) == size(boundary.fy1)
    copyto!(boundary.fy1, fy1)

    nothing
end

# restore gauge field
function restore!(gauge::Gauge, ts::OpenPMDTimeSeries, it::Int)
    xi, chart = get_field(ts, it=it, field="xi")
    @assert size(xi) == size(gauge.xi)
    copyto!(gauge.xi, xi)
    nothing
end


# recover all evolved variables from a checkpoint file
function recover(bulkevols::BulkPartition, boundary::Boundary, gauge::Gauge,
                 recovery_dir::String)
    ts = try
        OpenPMDTimeSeries(recovery_dir; prefix="checkpoint_it")
    catch e
        throw(e)
    end
    # grab last iteration of the timeseries, which should be the latest checkpoint
    it = ts.iterations[end]
    println("INFO: Recovering from checkpoint file $(ts.files[end])")

    restore!(bulkevols, ts, it)
    restore!(boundary, ts, it)
    restore!(gauge, ts, it)

    ts.current_iteration, ts.current_t
end

# for non-verbose output
vprint = x -> nothing

# for verbose output
# vprint(x) = println(x)
