function MPI(f, g, x, d, lb, ub; TS=["block","block"], merge=false, md=3, SO=[1;1], β=1, QUIET=false, solver="Mosek")
    n = length(x)
    m = length(g)
    fsupp,fcoe,flt,df = polys_info(f, x)
    gsupp,gcoe,glt,dg = polys_info(g, x)
    tsupp=zeros(UInt8, n, 1)
    basis = Vector{Array{UInt8,2}}(undef, m+1)
    basis[1] = get_basis(n, d)
    for i = 1:m
        basis[i+1] = get_basis(n, d-Int(ceil(dg[i]/2)))
        tsupp = [tsupp gsupp[i]]
    end
    tsupp = [tsupp get_Lsupp(n, tsupp, fsupp, flt)]
    sign_type = UInt8.(isodd.(tsupp))
    sign_type = sortslices(sign_type, dims=2)
    sign_type = unique(sign_type, dims=2)
    vsupp = get_basis(n, 2d+1-maximum(df))
    ind = [bfind(sign_type, size(sign_type, 2), UInt8.(isodd.(vsupp[:,i])))!=0 for i=1:size(vsupp,2)]
    vsupp = vsupp[:, ind]
    tsupp1 = [vsupp get_Lsupp(n, vsupp, fsupp, flt)]
    tsupp1 = sortslices(tsupp1, dims=2)
    tsupp1 = unique(tsupp1, dims=2)
    blocks1,vsupp,status = get_vblocks(n, m, 2d+1-maximum(df), tsupp1, vsupp, fsupp, flt, gsupp, glt, basis, TS=TS[1], SO=SO[1], merge=merge, md=md, QUIET=QUIET)
    if status == 1
        tsupp1 = get_tsupp(n, m, gsupp, glt, basis, blocks1)
        tsupp2 = sortslices(vsupp, dims=2)
        tsupp2 = unique(tsupp2, dims=2)
        blocks2,status = get_blocks(n, m, tsupp2, gsupp, glt, basis, TS=TS[2], SO=SO[2], merge=merge, md=md, QUIET=QUIET)
        if status == 1
            tsupp2 = get_tsupp(n, m, gsupp, glt, basis, blocks2)
            moment = get_moment(n, tsupp2, lb, ub)
            opt,wcoe = MPI_SDP(n, m, fsupp, fcoe, flt, gsupp, gcoe, glt, basis, tsupp1, tsupp2, vsupp, blocks1, blocks2, moment, β=β, QUIET=QUIET, solver=solver)
            ind = [abs(wcoe[i])>1e-6 for i=1:size(tsupp2, 2)]
            wsupp = [prod(x.^tsupp2[:,i]) for i=1:size(tsupp2, 2)]
            w = wcoe[ind]'*wsupp[ind]
            return opt,w
        end
    end
    return nothing,nothing
end

function MPI_SDP(n, m, fsupp, fcoe, flt, gsupp, gcoe, glt, basis, tsupp1, tsupp2, vsupp, blocks1, blocks2, moment; β=1, QUIET=false, solver="Mosek")
    if solver == "COSMO"
        model = Model(optimizer_with_attributes(COSMO.Optimizer))
    else
        model = Model(optimizer_with_attributes(Mosek.Optimizer))
    end
    set_optimizer_attribute(model, MOI.Silent(), QUIET)
    ltsupp1 = size(tsupp1, 2)
    ltsupp2 = size(tsupp2, 2)
    coeff1 = add_putinar!(model, m, tsupp1, gsupp, gcoe, glt, basis, blocks1)
    coeff2 = add_putinar!(model, m, tsupp2, gsupp, gcoe, glt, basis, blocks2)
    coeff3 = add_putinar!(model, m, tsupp2, gsupp, gcoe, glt, basis, blocks2)
    vcoe = @variable(model, [1:size(vsupp,2)])
    vfcoe = dvf(n, tsupp1, vsupp, vcoe, fsupp, fcoe, flt, β=β)
    @constraint(model, coeff1.==vfcoe)
    coeff4 = copy(coeff2)
    coeff4[1] -= 1
    for i = 1:size(vsupp, 2)
        Locb = bfind(tsupp2, ltsupp2, vsupp[:,i])
        coeff4[Locb] -= vcoe[i]
    end
    @constraint(model, coeff4.==coeff3)
    ind = [abs(moment[i])>1e-8 for i=1:ltsupp2]
    @objective(model, Min, moment[ind]'*coeff2[ind])
    optimize!(model)
    status=termination_status(model)
    if status!=MOI.OPTIMAL
       println("termination status: $status")
       status=primal_status(model)
       println("solution status: $status")
    end
    wcoe = value.(coeff2)
    objv = objective_value(model)
    return objv,wcoe
end

function dvf(n, tsupp, vsupp, vcoe, fsupp, fcoe, flt; β=1)
    ltsupp=size(tsupp, 2)
    vfcoe=[AffExpr(0) for i=1:ltsupp]
    lvsupp=size(vsupp, 2)
    for j=1:lvsupp
        locb=bfind(tsupp, ltsupp, vsupp[:, j])
        @inbounds add_to_expression!(vfcoe[locb], β, vcoe[j])
    end
    for i=1:n
        temp=zeros(UInt8, n)
        temp[i]=1
        for j=1:lvsupp
            if vsupp[i, j]>0
                for k=1:flt[i]
                    locb=bfind(tsupp, ltsupp, vsupp[:, j]-temp+fsupp[i][:, k])
                    vfcoe[locb]-=vsupp[i, j]*fcoe[i][k]*vcoe[j]
                    # @inbounds add_to_expression!(vfcoe[locb], -vsupp[i, j]*fcoe[i][k], vcoe[j])
                end
            end
        end
    end
    return vfcoe
end

function UPO(f, h, g, x, d; TS="block", SO=1, merge=false, md=3, QUIET=false, solver="Mosek")
    n = length(x)
    m = length(g)
    fsupp,fcoe,flt,df = polys_info(f, x)
    hsupp,hcoe = poly_info(h, x)
    gsupp,gcoe,glt,dg = polys_info(g, x)
    tsupp = hsupp
    basis = Vector{Array{UInt8,2}}(undef, m+1)
    basis[1] = get_basis(n, d)
    for i = 1:m
        basis[i+1] = get_basis(n, d-Int(ceil(dg[i]/2)))
        tsupp = [tsupp gsupp[i]]
    end
    tsupp = [tsupp get_Lsupp(n, tsupp, fsupp, flt)]
    sign_type = UInt8.(isodd.(tsupp))
    sign_type = sortslices(sign_type, dims=2)
    sign_type = unique(sign_type, dims=2)
    vsupp = get_basis(n, 2d+1-maximum(df))
    ind = [bfind(sign_type, size(sign_type, 2), UInt8.(isodd.(vsupp[:,i])))!=0 for i=1:size(vsupp,2)]
    vsupp = vsupp[:, ind]
    tsupp = get_Lsupp(n, vsupp, fsupp, flt)
    blocks,vsupp,status = get_vblocks(n, m, 2d+1-maximum(df), tsupp, vsupp, fsupp, flt, gsupp, glt, basis, TS=TS, SO=SO, merge=merge, md=md, QUIET=QUIET)
    if status == 1
        tsupp = get_tsupp(n, m, gsupp, glt, basis, blocks)
        opt = UPO_SDP(n, m, fsupp, fcoe, flt, hsupp, hcoe, gsupp, gcoe, glt, basis, tsupp, vsupp, blocks, QUIET=QUIET, solver=solver)
        return opt
    else
        return nothing
    end
end

function UPO_SDP(n, m, fsupp, fcoe, flt, hsupp, hcoe, gsupp, gcoe, glt, basis, tsupp, vsupp, blocks; QUIET=false, solver="Mosek")
    if solver == "COSMO"
        model = Model(optimizer_with_attributes(COSMO.Optimizer))
    else
        model = Model(optimizer_with_attributes(Mosek.Optimizer))
    end
    set_optimizer_attribute(model, MOI.Silent(), QUIET)
    coeff1 = add_putinar!(model, m, tsupp, gsupp, gcoe, glt, basis, blocks)
    ltsupp = size(tsupp, 2)
    coeff2 = [AffExpr(0) for i=1:ltsupp]
    vcoe = @variable(model, [1:size(vsupp,2)])
    for i = 1:n
        temp = zeros(UInt8, n)
        temp[i] = 1
        for j = 1:size(vsupp, 2)
            if vsupp[i, j] > 0
                for k = 1:flt[i]
                    locb = bfind(tsupp, ltsupp, vsupp[:, j]-temp+fsupp[i][:, k])
                    coeff2[locb] -= vsupp[i, j]*fcoe[i][k]*vcoe[j]
                    # @inbounds add_to_expression!(coeff2[locb], -vsupp[i, j]*fcoe[i][k], vcoe[j])
                end
            end
        end
    end
    for i = 1:size(hsupp, 2)
        Locb = bfind(tsupp, ltsupp, hsupp[:,i])
        coeff2[Locb] -= hcoe[i]
    end
    λ = @variable(model)
    coeff2[1] += λ
    @constraint(model, coeff1.==coeff2)
    @objective(model, Min, λ)
    optimize!(model)
    status = termination_status(model)
    if status != MOI.OPTIMAL
       println("termination status: $status")
       status = primal_status(model)
       println("solution status: $status")
    end
    objv = objective_value(model)
    println("optimum = $objv")
    return objv
end

function BEE(f, h, o, g, x, d; TS=["block","block"], SO=[1;1], merge=false, md=3, QUIET=false, solver="Mosek")
    n = length(x)
    fsupp,fcoe,flt,df = polys_info(f, x)
    hsupp,hcoe = poly_info(h, x)
    osupp,ocoe,olt,deo = polys_info(o, x)
    gsupp,gcoe,glt,dg = polys_info(g, x)
    m1 = length(o)
    m2 = length(g)
    tsupp = hsupp
    basis = Vector{Array{UInt8,2}}(undef, m1+1)
    basis[1] = get_basis(n, d)
    for i = 1:m1
        basis[i+1] = get_basis(n, d-Int(ceil(dg[i]/2)))
        tsupp = [tsupp osupp[i]]
    end
    for i = 1:m2
        tsupp = [tsupp gsupp[i]]
    end
    tsupp = [tsupp get_Lsupp(n, tsupp, fsupp, flt)]
    sign_type = UInt8.(isodd.(tsupp))
    sign_type = sortslices(sign_type, dims=2)
    sign_type = unique(sign_type, dims=2)
    vsupp = get_basis(n, 2d+1-maximum(df))
    ind = [bfind(sign_type, size(sign_type,2), UInt8.(isodd.(vsupp[:,i])))!=0 for i=1:size(vsupp,2)]
    vsupp = vsupp[:, ind]
    tsupp1 = get_Lsupp(n, vsupp, fsupp, flt)
    blocks1,vsupp,status = get_vblocks(n, m1, 2d+1-maximum(df), tsupp1, vsupp, fsupp, flt, osupp, olt, basis, TS=TS[1], SO=SO[1], merge=merge, QUIET=QUIET)
    if status == 1
        tsupp1 = get_tsupp(n, m1, osupp, olt, basis, blocks1)
        tsupp2 = sortslices(vsupp, dims=2)
        tsupp2 = unique(tsupp2, dims=2)
        blocks2,status = get_blocks(n, m1, tsupp2, osupp, olt, basis, TS=TS[2], SO=SO[2], merge=merge, md=md, QUIET=QUIET)
        if status == 1
            tsupp2 = get_tsupp(n, m1, osupp, olt, basis, blocks2)
            opt = BEE_SDP(n, m1, m2, fsupp, fcoe, flt, hsupp, hcoe, osupp, ocoe, olt, gsupp, gcoe, glt, basis, tsupp1,
            tsupp2, vsupp, blocks1, blocks2, QUIET=QUIET, solver=solver)
            return opt
        end
    end
    return nothing
end

function BEE_SDP(n, m1, m2, fsupp, fcoe, flt, hsupp, hcoe, osupp, ocoe, olt, gsupp, gcoe, glt, basis, tsupp1, tsupp2,
    vsupp, blocks1, blocks2; QUIET=false, solver="Mosek")
    if solver == "COSMO"
        model = Model(optimizer_with_attributes(COSMO.Optimizer))
    else
        model = Model(optimizer_with_attributes(Mosek.Optimizer))
    end
    set_optimizer_attribute(model, MOI.Silent(), QUIET)
    coeff1 = add_putinar!(model, m1, tsupp1, osupp, ocoe, olt, basis, blocks1)
    coeff2 = add_putinar!(model, m1, tsupp2, osupp, ocoe, olt, basis, blocks2)
    coeff3 = add_putinar!(model, m2, tsupp2, gsupp, gcoe, glt, basis, blocks2, numeq=m2)
    coeff4 = [AffExpr(0) for i=1:size(tsupp1,2)]
    vcoe = @variable(model, [1:size(vsupp,2)])
    for i = 1:n
        temp = zeros(UInt8, n)
        temp[i] = 1
        for j = 1:size(vsupp, 2)
            if vsupp[i, j] > 0
                for k = 1:flt[i]
                    locb = bfind(tsupp1, size(tsupp1,2), vsupp[:,j]-temp+fsupp[i][:,k])
                    coeff4[locb] -= vsupp[i,j]*fcoe[i][k]*vcoe[j]
                end
            end
        end
    end
    @constraint(model, coeff1.==coeff4)
    coeff5 = [AffExpr(0) for i=1:size(tsupp2,2)]
    for i = 1:size(vsupp,2)
        Locb = bfind(tsupp2, size(tsupp2,2), vsupp[:,i])
        coeff5[Locb] += vcoe[i]
    end
    for i = 1:size(hsupp,2)
        Locb = bfind(tsupp2, size(tsupp2,2), hsupp[:,i])
        coeff5[Locb] -= hcoe[i]
    end
    @constraint(model, coeff2.==coeff5)
    coeff6 = [AffExpr(0) for i=1:size(tsupp2,2)]
    for i = 1:size(vsupp,2)
        Locb = bfind(tsupp2, size(tsupp2,2), vsupp[1:n,i])
        coeff6[Locb] -= vcoe[i]
    end
    λ = @variable(model)
    coeff6[1] += λ
    @constraint(model, coeff3.==coeff6)
    @objective(model, Min, λ)
    optimize!(model)
    status = termination_status(model)
    if status != MOI.OPTIMAL
       println("termination status: $status")
       status = primal_status(model)
       println("solution status: $status")
    end
    objv = objective_value(model)
    println("optimum = $objv")
    return objv
end
