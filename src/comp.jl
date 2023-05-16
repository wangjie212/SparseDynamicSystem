function Tacchi(f1, f2, g1, g2, x, d, lb, ub, cliques; β=1, QUIET=false)
    n = length(x)
    m1 = length(g1)
    m2 = length(g2)
    fsupp1,fcoe1,flt1,df1 = polys_info(f1, x)
    fsupp2,fcoe2,flt2,df2 = polys_info(f2, x)
    gsupp1,gcoe1,glt1,dg1 = polys_info(g1, x)
    gsupp2,gcoe2,glt2,dg2 = polys_info(g2, x)
    n1 = length(cliques[1])
    # n2 = length(cliques[2])
    dv = 2d + 1 - max(maximum(df1), maximum(df2))
    basis1 = Vector{Array{UInt8,2}}(undef, m1+1)
    basis1[1] = get_basis(n, d, var=cliques[1])
    for i = 1:m1
        basis1[i+1] = get_basis(n, d-Int(ceil(dg1[i]/2)), var=cliques[1])
    end
    vsupp1 = get_basis(n, dv, var=cliques[1])
    tsupp1 = get_basis(n, 2d, var=cliques[1])
    basis2 = Vector{Array{UInt8,2}}(undef, m2+1)
    basis2[1] = get_basis(n, d, var=cliques[2])
    for i = 1:m2
        basis2[i+1] = get_basis(n, d-Int(ceil(dg2[i]/2)), var=cliques[2])
    end
    tsupp2 = get_basis(n, 2d, var=cliques[2])
    vsupp2 = get_basis(n, dv, var=cliques[2])
    tsupp = [tsupp1 tsupp2]
    tsupp = sortslices(tsupp, dims=2)
    tsupp = unique(tsupp, dims=2)
    ltsupp = size(tsupp, 2)
    ts = get_basis(n1, 2d)
    moment = get_moment(n1, ts, lb, ub)
    model = Model(optimizer_with_attributes(Mosek.Optimizer))
    set_optimizer_attribute(model, MOI.Silent(), QUIET)
    coeff1 = add_putinar!(model, tsupp, m1, gsupp1, gcoe1, glt1, basis1, m2, gsupp2, gcoe2, glt2, basis2)
    coeff2 = add_putinar!(model, tsupp, m1, gsupp1, gcoe1, glt1, basis1, m2, gsupp2, gcoe2, glt2, basis2)
    coeff3 = add_putinar!(model, tsupp, m1, gsupp1, gcoe1, glt1, basis1, m2, gsupp2, gcoe2, glt2, basis2)
    vcoe1 = @variable(model, [1:size(vsupp1,2)])
    vcoe2 = @variable(model, [1:size(vsupp2,2)])
    vfcoe = dvf(n, tsupp, vsupp1, vcoe1, fsupp1, fcoe1, flt1, vsupp2, vcoe2, fsupp2, fcoe2, flt2, β=β)
    @constraint(model, coeff1.==vfcoe)
    coeff4 = copy(coeff2)
    coeff4[1] -= 2
    for i = 1:size(vsupp1, 2)
        Locb = bfind(tsupp, ltsupp, vsupp1[:,i])
        coeff4[Locb] -= vcoe1[i]
    end
    for i = 1:size(vsupp2, 2)
        Locb = bfind(tsupp, ltsupp, vsupp2[:,i])
        coeff4[Locb] -= vcoe2[i]
    end
    @constraint(model, coeff4.==coeff3)
    obj = AffExpr(0)
    for i = 1:length(moment)
        if abs(moment[i]) > 1e-8
            Locb = bfind(tsupp, ltsupp, tsupp1[:,i])
            obj += moment[i]*coeff2[Locb]
        end
    end
    for i = 1:length(moment)
        if abs(moment[i]) > 1e-8
            Locb = bfind(tsupp, ltsupp, tsupp2[:,i])
            obj += moment[i]*coeff2[Locb]
        end
    end
    @objective(model, Min, obj)
    optimize!(model)
    status = termination_status(model)
    if status != MOI.OPTIMAL
       println("termination status: $status")
       status = primal_status(model)
       println("solution status: $status")
    end
    wcoe = value.(coeff2)
    opt = objective_value(model)
    ind = [abs(wcoe[i])>1e-6 for i=1:size(tsupp, 2)]
    wsupp = [prod(x.^tsupp[:,i]) for i=1:size(tsupp, 2)]
    w = wcoe[ind]'*wsupp[ind]
    return opt,w
end

function add_putinar!(model, tsupp, m1, gsupp1, gcoe1, glt1, basis1, m2, gsupp2, gcoe2, glt2, basis2)
    ltsupp = size(tsupp, 2)
    cons = [AffExpr(0) for i=1:ltsupp]
    bs = size(basis1[1], 2)
    pos = @variable(model, [1:bs, 1:bs], PSD)
    for j = 1:bs, r = j:bs
        bi = basis1[1][:,j] + basis1[1][:,r]
        Locb = bfind(tsupp, ltsupp, bi)
        if j == r
           @inbounds add_to_expression!(cons[Locb], pos[j,r])
        else
           @inbounds add_to_expression!(cons[Locb], 2, pos[j,r])
        end
    end
    for k = 1:m1
        bs = size(basis1[k+1], 2)
        pos = @variable(model, [1:bs, 1:bs], PSD)
        for j = 1:bs, r = j:bs, s = 1:glt1[k]
            bi = basis1[k+1][:,j] + basis1[k+1][:,r] + gsupp1[k][:,s]
            Locb = bfind(tsupp, ltsupp, bi)
            if j == r
               @inbounds add_to_expression!(cons[Locb], gcoe1[k][s], pos[j,r])
            else
               @inbounds add_to_expression!(cons[Locb], 2*gcoe1[k][s], pos[j,r])
            end
        end
    end
    bs = size(basis2[1], 2)
    pos = @variable(model, [1:bs, 1:bs], PSD)
    for j = 1:bs, r = j:bs
        bi = basis2[1][:,j] + basis2[1][:,r]
        Locb = bfind(tsupp, ltsupp, bi)
        if j == r
           @inbounds add_to_expression!(cons[Locb], pos[j,r])
        else
           @inbounds add_to_expression!(cons[Locb], 2, pos[j,r])
        end
    end
    for k = 1:m2
        bs = size(basis2[k+1], 2)
        pos = @variable(model, [1:bs, 1:bs], PSD)
        for j = 1:bs, r = j:bs, s = 1:glt2[k]
            bi = basis2[k+1][:,j] + basis2[k+1][:,r] + gsupp2[k][:,s]
            Locb = bfind(tsupp, ltsupp, bi)
            if j == r
               @inbounds add_to_expression!(cons[Locb], gcoe2[k][s], pos[j,r])
            else
               @inbounds add_to_expression!(cons[Locb], 2*gcoe2[k][s], pos[j,r])
            end
        end
    end
    return cons
end

function dvf(n, tsupp, vsupp1, vcoe1, fsupp1, fcoe1, flt1, vsupp2, vcoe2, fsupp2, fcoe2, flt2; β=1)
    ltsupp = size(tsupp, 2)
    vfcoe = [AffExpr(0) for i=1:ltsupp]
    lvsupp1 = size(vsupp1, 2)
    for j = 1:lvsupp1
        locb = bfind(tsupp, ltsupp, vsupp1[:, j])
        @inbounds add_to_expression!(vfcoe[locb], β, vcoe1[j])
    end
    for i = 1:length(flt1)
        temp = zeros(UInt8, n)
        temp[i] = 1
        for j = 1:lvsupp1
            if vsupp1[i, j] > 0
                for k = 1:flt1[i]
                    locb = bfind(tsupp, ltsupp, vsupp1[:, j]-temp+fsupp1[i][:, k])
                    vfcoe[locb] -= vsupp1[i, j]*fcoe1[i][k]*vcoe1[j]
                end
            end
        end
    end
    lvsupp2 = size(vsupp2, 2)
    for j = 1:lvsupp2
        locb = bfind(tsupp, ltsupp, vsupp2[:, j])
        @inbounds add_to_expression!(vfcoe[locb], β, vcoe2[j])
    end
    for i = n-length(flt2)+1:n
        temp = zeros(UInt8, n)
        temp[i] = 1
        for j = 1:lvsupp2
            if vsupp2[i, j] > 0
                for k = 1:flt2[i+length(flt2)-n]
                    locb = bfind(tsupp, ltsupp, vsupp2[:, j]-temp+fsupp2[i+length(flt2)-n][:, k])
                    vfcoe[locb] -= vsupp2[i, j]*fcoe2[i+length(flt2)-n][k]*vcoe2[j]
                end
            end
        end
    end
    return vfcoe
end
