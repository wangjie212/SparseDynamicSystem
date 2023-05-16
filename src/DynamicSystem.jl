function MPI(f, g, x, d, lb, ub; TS=["block","block"], merge=false, md=3, SO=[1;1], β=1, QUIET=false)
    n = length(x)
    m = length(g)
    fsupp,fcoe,flt,df = polys_info(f, x)
    gsupp,gcoe,glt,dg = polys_info(g, x)
    basis = Vector{Array{UInt8,2}}(undef, m+1)
    basis[1] = get_basis(n, d)
    for i = 1:m
        basis[i+1] = get_basis(n, d-Int(ceil(dg[i]/2)))
    end
    dv = 2d+1-maximum(df)
    if TS[1] != false
        tsupp = zeros(UInt8, n, 1)
        for i = 1:m
            tsupp = [tsupp gsupp[i]]
        end
        tsupp = [tsupp get_Lsupp(n, tsupp, fsupp, flt)]
        tsupp = sortslices(tsupp, dims=2)
        tsupp = unique(tsupp, dims=2)
        vsupp = tsupp[:, [sum(tsupp[:,i])<=dv for i=1:size(tsupp,2)]]
        tsupp1 = [vsupp get_Lsupp(n, vsupp, fsupp, flt)]
        tsupp1 = sortslices(tsupp1, dims=2)
        tsupp1 = unique(tsupp1, dims=2)
    else
        vsupp = get_basis(n, dv)
        tsupp1 = nothing
        tsupp = nothing
    end
    blocks1,vsupp,tsupp,status = get_vblocks(n, m, dv, tsupp1, tsupp, vsupp, fsupp, flt, gsupp, glt, basis, TS=TS[1], SO=SO[1], merge=merge, md=md, QUIET=QUIET)
    if status == 1
        tsupp1 = get_tsupp(n, m, gsupp, glt, basis, blocks1)
        if TS[1] != false
            tsupp = sortslices(tsupp, dims=2)
            tsupp = unique(tsupp, dims=2)
        end
        blocks2,_,_,_,_,status = get_blocks(n, m, tsupp, gsupp, glt, basis, TS=TS[2], SO=SO[2], merge=merge, md=md, QUIET=QUIET)
        if status == 1
            tsupp2 = get_tsupp(n, m, gsupp, glt, basis, blocks2)
            moment = get_moment(n, tsupp2, lb, ub)
            opt,wcoe = MPI_SDP(n, m, fsupp, fcoe, flt, gsupp, gcoe, glt, basis, tsupp1, tsupp2, vsupp, blocks1, blocks2, moment, β=β, QUIET=QUIET)
            ind = [abs(wcoe[i])>1e-6 for i=1:size(tsupp2, 2)]
            wsupp = [prod(x.^tsupp2[:,i]) for i=1:size(tsupp2, 2)]
            w = wcoe[ind]'*wsupp[ind]
            return opt,w
        end
    end
    return nothing,nothing
end

function MPI_SDP(n, m, fsupp, fcoe, flt, gsupp, gcoe, glt, basis, tsupp1, tsupp2, vsupp, blocks1, blocks2, moment; β=1, QUIET=false)
    model = Model(optimizer_with_attributes(Mosek.Optimizer))
    set_optimizer_attribute(model, MOI.Silent(), QUIET)
    # ltsupp1 = size(tsupp1, 2)
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
    status = termination_status(model)
    if status != MOI.OPTIMAL
       println("termination status: $status")
       status = primal_status(model)
       println("solution status: $status")
    end
    wcoe = value.(coeff2)
    objv = objective_value(model)
    return objv,wcoe
end

function dvf(n, tsupp, vsupp, vcoe, fsupp, fcoe, flt; β=1)
    ltsupp = size(tsupp, 2)
    vfcoe = [AffExpr(0) for i=1:ltsupp]
    lvsupp = size(vsupp, 2)
    for j = 1:lvsupp
        locb = bfind(tsupp, ltsupp, vsupp[:, j])
        @inbounds add_to_expression!(vfcoe[locb], β, vcoe[j])
    end
    for i = 1:length(flt)
        temp = zeros(UInt8, n)
        temp[i] = 1
        for j = 1:lvsupp
            if vsupp[i, j] > 0
                for k = 1:flt[i]
                    locb = bfind(tsupp, ltsupp, vsupp[:, j]-temp+fsupp[i][:, k])
                    vfcoe[locb] -= vsupp[i, j]*fcoe[i][k]*vcoe[j]
                    # @inbounds add_to_expression!(vfcoe[locb], -vsupp[i, j]*fcoe[i][k], vcoe[j])
                end
            end
        end
    end
    return vfcoe
end

function UPO(f, h, g, x, d; TS="block", SO=1, merge=false, md=3, QUIET=false)
    n = length(x)
    m = length(g)
    fsupp,fcoe,flt,df = polys_info(f, x)
    hsupp,hcoe = poly_info(h, x)
    gsupp,gcoe,glt,dg = polys_info(g, x)
    basis = Vector{Array{UInt8,2}}(undef, m+1)
    basis[1] = get_basis(n, d)
    for i = 1:m
        basis[i+1] = get_basis(n, d-Int(ceil(dg[i]/2)))
    end
    dv = 2d+1-maximum(df)
    if TS != false
        tsupp = hsupp
        for i = 1:m
            tsupp = [tsupp gsupp[i]]
        end
        tsupp = [tsupp get_Lsupp(n, tsupp, fsupp, flt)]
        tsupp = sortslices(tsupp, dims=2)
        tsupp = unique(tsupp, dims=2)
        vsupp = tsupp[:, [sum(tsupp[:,i])<=dv for i=1:size(tsupp,2)]]
        tsupp1 = get_Lsupp(n, vsupp, fsupp, flt)
    else
        vsupp = get_basis(n, dv)
        tsupp1 = nothing
        tsupp = nothing
    end
    blocks,vsupp,tsupp,status = get_vblocks(n, m, dv, tsupp1, tsupp, vsupp, fsupp, flt, gsupp, glt, basis, TS=TS, SO=SO, merge=merge, md=md, QUIET=QUIET)
    if status == 1
        tsupp1 = get_tsupp(n, m, gsupp, glt, basis, blocks)
        opt = UPO_SDP(n, m, fsupp, fcoe, flt, hsupp, hcoe, gsupp, gcoe, glt, basis, tsupp1, vsupp, blocks, QUIET=QUIET)
        return opt
    else
        return nothing
    end
end

function UPO_SDP(n, m, fsupp, fcoe, flt, hsupp, hcoe, gsupp, gcoe, glt, basis, tsupp, vsupp, blocks; QUIET=false)
    model = Model(optimizer_with_attributes(Mosek.Optimizer))
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

function BEE(f, h, o, g, x, d; TS=["block","block"], SO=[1;1], merge=false, md=3, QUIET=false)
    n = length(x)
    fsupp,fcoe,flt,df = polys_info(f, x)
    hsupp,hcoe = poly_info(h, x)
    osupp,ocoe,olt,deo = polys_info(o, x)
    gsupp,gcoe,glt,dg = polys_info(g, x)
    m1 = length(o)
    m2 = length(g)
    basis = Vector{Array{UInt8,2}}(undef, m1+1)
    basis[1] = get_basis(n, d)
    for i = 1:m1
        basis[i+1] = get_basis(n, d-Int(ceil(dg[i]/2)))
    end
    dv = 2d+1-maximum(df)
    if TS[1] != false
        tsupp = hsupp
        for i = 1:m1
            tsupp = [tsupp osupp[i]]
        end
        for i = 1:m2
            tsupp = [tsupp gsupp[i]]
        end
        tsupp = [tsupp get_Lsupp(n, tsupp, fsupp, flt)]
        tsupp = sortslices(tsupp, dims=2)
        tsupp = unique(tsupp, dims=2)
        vsupp = tsupp[:, [sum(tsupp[:,i])<=dv for i=1:size(tsupp,2)]]
        tsupp1 = get_Lsupp(n, vsupp, fsupp, flt)
    else
        vsupp = get_basis(n, dv)
        tsupp1 = nothing
        tsupp = nothing
    end
    blocks1,vsupp,tsupp,status = get_vblocks(n, m1, dv, tsupp1, tsupp, vsupp, fsupp, flt, osupp, olt, basis, TS=TS[1], SO=SO[1], merge=merge, QUIET=QUIET)
    if status == 1
        tsupp1 = get_tsupp(n, m1, osupp, olt, basis, blocks1)
        if TS[1] != false
            tsupp = sortslices(tsupp, dims=2)
            tsupp = unique(tsupp, dims=2)
        end
        blocks2,_,_,_,_,status = get_blocks(n, m1, tsupp, osupp, olt, basis, TS=TS[2], SO=SO[2], merge=merge, md=md, QUIET=QUIET)
        if status == 1
            tsupp2 = get_tsupp(n, m1, osupp, olt, basis, blocks2)
            opt = BEE_SDP(n, m1, m2, fsupp, fcoe, flt, hsupp, hcoe, osupp, ocoe, olt, gsupp, gcoe, glt, basis, tsupp1,
            tsupp2, vsupp, blocks1, blocks2, QUIET=QUIET)
            return opt
        end
    end
    return nothing
end

function BEE_SDP(n, m1, m2, fsupp, fcoe, flt, hsupp, hcoe, osupp, ocoe, olt, gsupp, gcoe, glt, basis, tsupp1, tsupp2,
    vsupp, blocks1, blocks2; QUIET=false)
    model = Model(optimizer_with_attributes(Mosek.Optimizer))
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

function ROA(f, p, q, x, T, d, lb, ub; TS=["block","block"], merge=false, md=3, SO=[1;1], β=1, QUIET=false)
    n = length(x)
    m1 = length(p)
    m2 = length(q)
    fsupp,fcoe,flt,df = polys_info(f, x)
    psupp,pcoe,plt,dp = polys_info(p, x)
    qsupp,qcoe,qlt,dq = polys_info(q, x)
    basis1 = Vector{Array{UInt8,2}}(undef, m1+1)
    basis1[1] = get_basis(n, d)
    psupp0 = Vector{Array{UInt8,2}}(undef, m1+1)
    for i = 1:m1
        basis1[i+1] = get_basis(n, d-Int(ceil(dp[i]/2)))
        psupp0[i] = [psupp[i]; zeros(UInt8, 1, size(psupp[i],2))]
    end
    psupp0[m1+1] = zeros(UInt8, n+1, 2)
    psupp0[m1+1][n+1, 1] = 1
    psupp0[m1+1][n+1, 2] = 2
    pcoe0 = [pcoe; [[T;-1]]]
    plt0 = [plt;2]
    dp0 = [dp;2]
    basis2 = Vector{Array{UInt8,2}}(undef, m2+1)
    basis2[1] = basis1[1]
    for i = 1:m2
        basis2[i+1] = get_basis(n, d-Int(ceil(dq[i]/2)))
    end
    basis0 = Vector{Array{UInt8,2}}(undef, m1+2)
    basis0[1] = get_basis(n+1, d)
    for i = 1:m1+1
        basis0[i+1] = get_basis(n+1, d-Int(ceil(dp0[i]/2)))
    end
    dv = 2d+1-maximum(df)
    if TS[1] != false
        tsupp = zeros(UInt8, n, 1)
        for i = 1:m1
            tsupp = [tsupp psupp[i]]
        end
        for i = 1:m2
            tsupp = [tsupp qsupp[i]]
        end
        tsupp = [tsupp get_Lsupp(n, tsupp, fsupp, flt)]
        tsupp = [[tsupp; zeros(UInt8, 1, size(tsupp,2))] [tsupp; ones(UInt8, 1, size(tsupp,2))]]
        tsupp = sortslices(tsupp, dims=2)
        tsupp = unique(tsupp, dims=2)
        vsupp = tsupp[:, [sum(tsupp[:,i])<=dv for i=1:size(tsupp,2)]]
        for i = 1:n
            fsupp[i] = [fsupp[i]; zeros(UInt8, 1, size(fsupp[i],2))]
        end
        tsupp0 = [vsupp get_Lsupp(n+1, vsupp, fsupp, flt)]
        tsupp0 = sortslices(tsupp0, dims=2)
        tsupp0 = unique(tsupp0, dims=2)
    else
        vsupp = get_basis(n+1, dv)
        for i = 1:n
            fsupp[i] = [fsupp[i]; zeros(UInt8, 1, size(fsupp[i],2))]
        end
        tsupp0 = nothing
        tsupp = nothing
    end
    blocks0,vsupp,tsupp,status = get_vblocks(n+1, m1+1, dv, tsupp0, tsupp, vsupp, fsupp, flt, psupp0, plt0, basis0, TS=TS[1], SO=SO[1], merge=merge, md=md, QUIET=QUIET)
    if status == 1
        tsupp0 = get_tsupp(n+1, m1+1, psupp0, plt0, basis0, blocks0)
        if TS[1] != false
            tsupp = sortslices(tsupp[1:n,:], dims=2)
            tsupp = unique(tsupp, dims=2)
        end
        blocks1,_,_,_,_,status = get_blocks(n, m1, tsupp, psupp, plt, basis1, TS=TS[2], SO=SO[2], merge=merge, md=md, QUIET=QUIET)
        blocks2,_,_,_,_,status = get_blocks(n, m2, tsupp, qsupp, qlt, basis2, TS=TS[2], SO=SO[2], merge=merge, md=md, QUIET=QUIET)
        if status == 1
            tsupp1 = get_tsupp(n, m1, psupp, plt, basis1, blocks1)
            tsupp2 = get_tsupp(n, m2, qsupp, qlt, basis2, blocks2)
            moment = get_moment(n, tsupp1, lb, ub)
            opt,vcoe = ROA_SDP(n, m1, m2, T, fsupp, fcoe, flt, psupp0, pcoe0, plt0, psupp, pcoe, plt, qsupp, qcoe, qlt, basis0, basis1, basis2, tsupp0, tsupp1, tsupp2, vsupp, blocks0, blocks1, blocks2, moment, β=β, QUIET=QUIET)
            ind = [vsupp[n+1,i] == 0&&abs(vcoe[i])>1e-6 for i=1:size(vsupp, 2)]
            vsuppx = [prod(x.^vsupp[1:n,i]) for i=1:size(vsupp, 2)]
            v0 = vcoe[ind]'*vsuppx[ind]
            return opt,v0
        end
    end
    return nothing,nothing
end

function ROA_SDP(n, m1, m2, T, fsupp, fcoe, flt, psupp0, pcoe0, plt0, psupp, pcoe, plt, qsupp, qcoe, qlt, basis0, basis1, basis2, tsupp0, tsupp1, tsupp2, vsupp, blocks0, blocks1, blocks2, moment; β=1, QUIET=false)
    model = Model(optimizer_with_attributes(Mosek.Optimizer))
    set_optimizer_attribute(model, MOI.Silent(), QUIET)
    ltsupp0 = size(tsupp0, 2)
    ltsupp1 = size(tsupp1, 2)
    ltsupp2 = size(tsupp2, 2)
    coeff0 = add_putinar!(model, m1+1, tsupp0, psupp0, pcoe0, plt0, basis0, blocks0)
    coeff1 = add_putinar!(model, m1, tsupp1, psupp, pcoe, plt, basis1, blocks1)
    coeff2 = add_putinar!(model, m1, tsupp1, psupp, pcoe, plt, basis1, blocks1)
    coeff3 = add_putinar!(model, m2, tsupp2, qsupp, qcoe, qlt, basis2, blocks2)
    vcoe = @variable(model, [1:size(vsupp,2)])
    vfcoe = [AffExpr(0) for i=1:ltsupp0]
    for i = 1:n
        temp = zeros(UInt8, n+1)
        temp[i] = 1
        for j = 1:size(vsupp,2)
            if vsupp[i, j] > 0
                for k = 1:flt[i]
                    locb = bfind(tsupp0, ltsupp0, vsupp[:, j]-temp+fsupp[i][:, k])
                    vfcoe[locb] -= vsupp[i, j]*fcoe[i][k]*vcoe[j]
                end
            end
        end
    end
    for i = 1:length(vcoe)
        if vsupp[n+1, i] > 0
            temp = zeros(UInt8, n+1)
            temp[n+1] = 1
            locb = bfind(tsupp0, ltsupp0, vsupp[:, i]-temp)
            vfcoe[locb] -= vsupp[n+1, i]*vcoe[i]
        end
    end
    @constraint(model, coeff0.==vfcoe)
    coeff4 = copy(coeff1)
    coeff4[1] -= 1
    for i = 1:size(vsupp, 2)
        if vsupp[n+1,i] == 0
            Locb = bfind(tsupp1, ltsupp1, vsupp[1:n,i])
            coeff4[Locb] -= vcoe[i]
        end
    end
    @constraint(model, coeff4.==coeff2)
    coeff5 = [AffExpr(0) for i=1:ltsupp2]
    for i = 1:size(vsupp, 2)
        Locb = bfind(tsupp2, ltsupp2, vsupp[1:n,i])
        coeff5[Locb] += vcoe[i]*T^vsupp[n+1,i]
    end
    @constraint(model, coeff5.==coeff3)
    ind = [abs(moment[i])>1e-8 for i=1:ltsupp1]
    @objective(model, Min, moment[ind]'*coeff1[ind])
    optimize!(model)
    status = termination_status(model)
    if status != MOI.OPTIMAL
       println("termination status: $status")
       status = primal_status(model)
       println("solution status: $status")
    end
    objv = objective_value(model)
    return objv,value.(vcoe)
end

function GA(f, g, x, d, lb, ub; TS=["block","block"], merge=false, md=3, SO=[1;1], β=1, QUIET=false)
    n = length(x)
    m = length(g)
    fsupp,fcoe,flt,df = polys_info(f, x)
    gsupp,gcoe,glt,dg = polys_info(g, x)
    basis = Vector{Array{UInt8,2}}(undef, m+1)
    basis[1] = get_basis(n, d)
    for i = 1:m
        basis[i+1] = get_basis(n, d-Int(ceil(dg[i]/2)))
    end
    dv = 2d+1-maximum(df)
    if TS[1] != false
        tsupp=zeros(UInt8, n, 1)
        for i = 1:m
            tsupp = [tsupp gsupp[i]]
        end
        tsupp = [tsupp get_Lsupp(n, tsupp, fsupp, flt)]
        tsupp = sortslices(tsupp, dims=2)
        tsupp = unique(tsupp, dims=2)
        vsupp = tsupp[:, [sum(tsupp[:,i])<=dv for i=1:size(tsupp,2)]]
        tsupp1 = [vsupp get_Lsupp(n, vsupp, fsupp, flt)]
        tsupp1 = sortslices(tsupp1, dims=2)
        tsupp1 = unique(tsupp1, dims=2)
    else
        vsupp = get_basis(n, dv)
        tsupp1 = nothing
        tsupp = nothing
    end
    blocks1,vsupp,tsupp,status = get_vblocks(n, m, dv, tsupp1, tsupp, vsupp, fsupp, flt, gsupp, glt, basis, TS=TS[1], SO=SO[1], merge=merge, md=md, QUIET=QUIET)
    if status == 1
        tsupp1 = get_tsupp(n, m, gsupp, glt, basis, blocks1)
        if TS[1] != false
            tsupp = sortslices(tsupp, dims=2)
            tsupp = unique(tsupp, dims=2)
        end
        blocks2,_,_,_,_,status = get_blocks(n, m, tsupp, gsupp, glt, basis, TS=TS[2], SO=SO[2], merge=merge, md=md, QUIET=QUIET)
        if status == 1
            tsupp2 = get_tsupp(n, m, gsupp, glt, basis, blocks2)
            moment = get_moment(n, tsupp2, lb, ub)
            opt,wcoe = GA_SDP(n, m, fsupp, fcoe, flt, gsupp, gcoe, glt, basis, tsupp1, tsupp2, vsupp, blocks1, blocks2, moment, β=β, QUIET=QUIET)
            ind = [abs(wcoe[i])>1e-6 for i=1:size(tsupp2, 2)]
            wsupp = [prod(x.^tsupp2[:,i]) for i=1:size(tsupp2, 2)]
            w = wcoe[ind]'*wsupp[ind]
            return opt,w
        end
    end
    return nothing,nothing
end

function GA_SDP(n, m, fsupp, fcoe, flt, gsupp, gcoe, glt, basis, tsupp1, tsupp2, vsupp, blocks1, blocks2, moment; β=1, QUIET=false)
    model = Model(optimizer_with_attributes(Mosek.Optimizer))
    set_optimizer_attribute(model, MOI.Silent(), QUIET)
    ltsupp1 = size(tsupp1, 2)
    ltsupp2 = size(tsupp2, 2)
    coeff1 = add_putinar!(model, m, tsupp1, gsupp, gcoe, glt, basis, blocks1)
    coeff2 = add_putinar!(model, m, tsupp1, gsupp, gcoe, glt, basis, blocks1)
    coeff3 = add_putinar!(model, m, tsupp2, gsupp, gcoe, glt, basis, blocks2)
    coeff4 = add_putinar!(model, m, tsupp2, gsupp, gcoe, glt, basis, blocks2)
    pcoe = @variable(model, [1:size(vsupp,2)])
    pfcoe = dvf(n, tsupp1, vsupp, pcoe, fsupp, fcoe, flt, β=β)
    @constraint(model, coeff1.==pfcoe)
    qcoe = @variable(model, [1:size(vsupp,2)])
    qfcoe = dvf(n, tsupp1, vsupp, qcoe, fsupp, fcoe, flt, β=β)
    @constraint(model, coeff2.==qfcoe)
    coeff5 = copy(coeff3)
    coeff5[1] -= 1
    for i = 1:size(vsupp, 2)
        Locb = bfind(tsupp2, ltsupp2, vsupp[:,i])
        coeff5[Locb] -= pcoe[i] + qcoe[i]
    end
    @constraint(model, coeff5.==coeff4)
    ind = [abs(moment[i])>1e-8 for i=1:ltsupp2]
    @objective(model, Min, moment[ind]'*coeff2[ind])
    optimize!(model)
    status = termination_status(model)
    if status != MOI.OPTIMAL
       println("termination status: $status")
       status = primal_status(model)
       println("solution status: $status")
    end
    wcoe = value.(coeff2)
    objv = objective_value(model)
    return objv,wcoe
end
