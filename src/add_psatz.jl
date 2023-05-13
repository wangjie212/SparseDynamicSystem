using SemialgebraicSets

mutable struct struct_data
    cliques # the clique structrue
    cql # number of cliques
    cliquesize # size of cliques
    basis # monomial basis
    blocks # the block structrue
    cl # number of blocks
    blocksize # size of blocks
    tsupp # total support
end

function add_psatz!(model, nonneg, vars, ineq_cons, eq_cons, order; CS=false, TS="block", SO=1, Groebnerbasis=false, QUIET=false)
    n = length(vars)
    m = length(ineq_cons)
    if ineq_cons != []
        gsupp,gcoe,glt,dg = polys_info(ineq_cons, vars)
    else
        gsupp = Matrix{UInt8}[]
        glt = dg = Int[]
    end
    if eq_cons != []
        hsupp,hcoe,hlt,dh = polys_info(eq_cons, vars)
    else
        hsupp = Matrix{UInt8}[]
        hlt = dh = Int[]
    end
    if Groebnerbasis == true && eq_cons != []
        l = 0
        gb = convert.(Polynomial{true,Float64}, eq_cons)
        nonneg = rem(nonneg, gb)
        SemialgebraicSets.gröbnerbasis!(gb)
        leadm = leadingmonomial.(gb)
        llead = length(leadm)
        lead = zeros(UInt8, n, llead)
        for i = 1:llead, j = 1:n
            @inbounds lead[j,i] = MultivariatePolynomials.degree(leadm[i], vars[j])
        end
    else
        gb = []
        l = length(eq_cons)
    end
    fsupp,fcoe = poly_info(nonneg, vars)
    dmin = ceil(Int, maximum([maxdegree(nonneg); dg; dh])/2)
    order = order < dmin ? dmin : order
    if CS == true
        cliques,cql,cliquesize = clique_decomp(n, m, length(eq_cons), fsupp, gsupp, hsupp)
    else
        cliques,cql,cliquesize = [Vector(1:n)],1,[n]
    end
    I,J = assign_constraint(m, l, gsupp, hsupp, cliques, cql)
    basis = Vector{Vector{Matrix{UInt8}}}(undef, cql)
    for t = 1:cql
        basis[t] = Vector{Matrix{UInt8}}(undef, length(I[t])+length(J[t])+1)
        basis[t][1] = get_basis(n, order, var=cliques[t])
        for s = 1:length(I[t])
            basis[t][s+1] = get_basis(n, order-ceil(Int, dg[I[t][s]]/2), var=cliques[t])
        end
        for s = 1:length(J[t])
            basis[t][s+length(I[t])+1] = get_basis(n, order-ceil(Int, dh[J[t][s]]/2), var=cliques[t])
        end
    end
    blocks,cl,blocksize,sb,numb,status = get_cblocks_mix(n, I, J, m, l, fsupp, gsupp, glt, hsupp, hlt, basis, cliques, cql, tsupp=[], TS=TS, SO=SO, QUIET=QUIET)
    ne = 0
    for t = 1:cql
        ne += sum(numele(blocksize[t][1]))
        if I[t] != []
            ne += sum(glt[I[t][k]]*numele(blocksize[t][k+1]) for k=1:length(I[t]))
        end
        if J[t] != []
            ne += sum(hlt[J[t][k]]*numele(blocksize[t][k+length(I[t])+1]) for k=1:length(J[t]))
        end
    end
    tsupp = zeros(UInt8, n, ne)
    q = 1
    for i = 1:cql
        for j = 1:cl[i][1], k = 1:blocksize[i][1][j], r = k:blocksize[i][1][j]
            @inbounds bi = basis[i][1][:, blocks[i][1][j][k]] + basis[i][1][:, blocks[i][1][j][r]]
            tsupp[:, q] = bi
            q += 1
        end
        for (j, w) in enumerate(I[i]), p = 1:cl[i][j+1], t = 1:blocksize[i][j+1][p], r = t:blocksize[i][j+1][p], s = 1:glt[w]
            ind1 = blocks[i][j+1][p][t]
            ind2 = blocks[i][j+1][p][r]
            @inbounds bi = basis[i][j+1][:, ind1] + basis[i][j+1][:, ind2] + gsupp[w][:, s]
            tsupp[:, q] = bi
            q += 1
        end
        for (j, w) in enumerate(J[i]), p = 1:cl[i][j+length(I[i])+1], t = 1:blocksize[i][j+length(I[i])+1][p], r = t:blocksize[i][j+length(I[i])+1][p], s = 1:hlt[w]
            ind1 = blocks[i][j+length(I[i])+1][p][t]
            ind2 = blocks[i][j+length(I[i])+1][p][r]
            @inbounds bi = basis[i][j+length(I[i])+1][:, ind1] + basis[i][j+length(I[i])+1][:, ind2] + hsupp[w][:, s]
            tsupp[:, q] = bi
            q += 1
        end
    end
    if !isempty(gb)
        tsupp = unique(tsupp, dims=2)
        nsupp = zeros(UInt8, n)
        for col in eachcol(tsupp)
            if divide(col, lead, n, llead)
                temp = reminder(col, vars, gb, n)[2]
                nsupp = [nsupp temp]
            else
                nsupp = [nsupp col]
            end
        end
        tsupp = nsupp
    end
    tsupp = sortslices(tsupp, dims=2)
    tsupp = unique(tsupp, dims=2)
    info = struct_data(cliques,cql,cliquesize,basis,blocks,cl,blocksize,tsupp)
    ltsupp = size(tsupp, 2)
    cons = [AffExpr(0) for i=1:ltsupp]
    for t = 1:cql
        for i = 1:cl[t][1]
            bs = blocksize[t][1][i]
            if bs == 1
               pos = @variable(model, lower_bound=0)
               bi = 2*basis[t][1][:, blocks[t][1][i][1]]
               if !isempty(gb) && divide(bi, lead, n, llead)
                    bi_lm,bi_supp,bi_coe = reminder(bi, vars, gb, n)
                    for z = 1:bi_lm
                        Locb = bfind(tsupp, ltsupp, bi_supp[:,z])
                        @inbounds add_to_expression!(cons[Locb], bi_coe[z], pos)
                    end
               else
                    Locb = bfind(tsupp, ltsupp, bi)
                    @inbounds add_to_expression!(cons[Locb], pos)
               end
            else
               pos = @variable(model, [1:bs, 1:bs], PSD)
               for j = 1:bs, r = j:bs
                   bi = basis[t][1][:, blocks[t][1][i][j]] + basis[t][1][:, blocks[t][1][i][r]]
                   if !isempty(gb) && divide(bi, lead, n, llead)
                        bi_lm,bi_supp,bi_coe = reminder(bi, vars, gb, n)
                        for z = 1:bi_lm
                            Locb = bfind(tsupp, ltsupp, bi_supp[:,z])
                            if j == r
                                @inbounds add_to_expression!(cons[Locb], bi_coe[z], pos[j,r])
                            else
                                @inbounds add_to_expression!(cons[Locb], 2*bi_coe[z], pos[j,r])
                            end
                        end
                   else
                        Locb = bfind(tsupp, ltsupp, bi)
                        if j == r
                            @inbounds add_to_expression!(cons[Locb], pos[j,r])
                        else
                            @inbounds add_to_expression!(cons[Locb], 2, pos[j,r])
                        end
                    end
               end
            end
        end
        for k = 1:length(I[t]), i = 1:length(blocks[t][k+1])
            bs = length(blocks[t][k+1][i])
            if bs == 1
                pos = @variable(model, lower_bound=0)
                for s = 1:glt[I[t][k]]
                    bi = 2*basis[t][k+1][:, blocks[t][k+1][i][1]] + gsupp[I[t][k]][:,s]
                    if !isempty(gb) && divide(bi, lead, n, llead)
                        bi_lm,bi_supp,bi_coe = reminder(bi, vars, gb, n)
                        for z = 1:bi_lm
                            Locb = bfind(tsupp, ltsupp, bi_supp[:,z])
                            @inbounds add_to_expression!(cons[Locb], gcoe[I[t][k]][s]*bi_coe[z], pos)
                        end
                    else
                        Locb = bfind(tsupp, ltsupp, bi)
                        @inbounds add_to_expression!(cons[Locb], gcoe[I[t][k]][s], pos)
                    end
                end
            else
                pos = @variable(model, [1:bs, 1:bs], PSD)
                for j = 1:bs, r = j:bs, s = 1:glt[I[t][k]]
                    bi = basis[t][k+1][:, blocks[t][k+1][i][j]] + basis[t][k+1][:, blocks[t][k+1][i][r]] + gsupp[I[t][k]][:,s]
                    if !isempty(gb) && divide(bi, lead, n, llead)
                        bi_lm,bi_supp,bi_coe = reminder(bi, vars, gb, n)
                        for z = 1:bi_lm
                            Locb = bfind(tsupp, ltsupp, bi_supp[:,z])
                            if j == r
                                @inbounds add_to_expression!(cons[Locb], gcoe[I[t][k]][s]*bi_coe[z], pos[j,r])
                            else
                                @inbounds add_to_expression!(cons[Locb], 2*gcoe[I[t][k]][s]*bi_coe[z], pos[j,r])
                            end
                        end
                   else
                        Locb = bfind(tsupp, ltsupp, bi)
                        if j == r
                            @inbounds add_to_expression!(cons[Locb], gcoe[I[t][k]][s], pos[j,r])
                        else
                            @inbounds add_to_expression!(cons[Locb], 2*gcoe[I[t][k]][s], pos[j,r])
                        end
                    end
                end
            end
        end
        for k = 1:length(J[t]), i = 1:length(blocks[t][k+length(I[t])+1])
            bs = length(blocks[t][k+length(I[t])+1][i])
            if bs == 1
                pos = @variable(model)
                for s = 1:hlt[J[t][k]]
                    bi = 2*basis[t][k+length(I[t])+1][:, blocks[t][k+length(I[t])+1][i][1]] + hsupp[J[t][k]][:,s]
                    Locb = bfind(tsupp, ltsupp, bi)
                    @inbounds add_to_expression!(cons[Locb], hcoe[J[t][k]][s], pos)
                end
            else
                pos = @variable(model, [1:bs, 1:bs], Symmetric)
                for j = 1:bs, r = j:bs, s = 1:hlt[J[t][k]]
                    bi = basis[t][k+length(I[t])+1][:, blocks[t][k+length(I[t])+1][i][j]] + basis[t][k+length(I[t])+1][:, blocks[t][k+length(I[t])+1][i][r]] + hsupp[J[t][k]][:,s]
                    Locb = bfind(tsupp, ltsupp, bi)
                    if j == r
                       @inbounds add_to_expression!(cons[Locb], hcoe[J[t][k]][s], pos[j,r])
                    else
                       @inbounds add_to_expression!(cons[Locb], 2*hcoe[J[t][k]][s], pos[j,r])
                    end
                end
            end
        end
    end
    bc = [AffExpr(0) for i=1:ltsupp]
    for i = 1:size(fsupp, 2)
        Locb = bfind(tsupp, ltsupp, fsupp[:,i])
        if Locb == 0
            @error "The monomial basis is not enough!"
            return model,info
        else
            bc[Locb] = fcoe[i]
        end
    end
    @constraint(model, cons.==bc)
    return model,info
end

function clique_decomp(n, m, l, fsupp, gsupp, hsupp)
    G = SimpleGraph(n)
    for j = 1:size(fsupp, 2)
        add_clique!(G, findall(fsupp[:,j] .!= 0))
    end
    for i = 1:m
        temp = findall(gsupp[i][:,1] .!= 0)
        for j = 2:size(gsupp[i], 2)
            append!(temp, findall(gsupp[i][:,j] .!= 0))
        end
        add_clique!(G, unique(temp))
    end
    for i = 1:l
        temp = findall(hsupp[i][:,1] .!= 0)
        for j = 2:size(hsupp[i], 2)
            append!(temp, findall(hsupp[i][:,j] .!= 0))
        end
        add_clique!(G, unique(temp))
    end
    cliques,cql,cliquesize = chordal_cliques!(G)
    uc = unique(cliquesize)
    sizes=[sum(cliquesize.== i) for i in uc]
    println("-----------------------------------------------------------------------------")
    println("The clique sizes of varibles:\n$uc\n$sizes")
    println("-----------------------------------------------------------------------------")
    return cliques,cql,cliquesize
end

function assign_constraint(m, l, gsupp, hsupp, cliques, cql)
    I = [UInt32[] for i=1:cql]
    J = [UInt32[] for i=1:cql]
    for i = 1:m
        rind = findall(gsupp[i][:,1] .!= 0)
        for j = 2:size(gsupp[i], 2)
            append!(rind, findall(gsupp[i][:,j] .!= 0))
        end
        unique!(rind)
        ind = findfirst(k->issubset(rind, cliques[k]), 1:cql)
        push!(I[ind], i)
    end
    for i = 1:l
        rind = findall(hsupp[i][:,1] .!= 0)
        for j = 2:size(hsupp[i], 2)
            append!(rind, findall(hsupp[i][:,j] .!= 0))
        end
        unique!(rind)
        ind = findfirst(k->issubset(rind, cliques[k]), 1:cql)
        push!(J[ind], i)
    end
    return I,J
end

function get_cblocks_mix(n, I, J, m, l, fsupp, gsupp, glt, hsupp, hlt, basis, cliques, cql; tsupp=[], TS="block", SO=1, QUIET=false)
    blocks = Vector{Vector{Vector{Vector{UInt16}}}}(undef, cql)
    cl = Vector{Vector{Int}}(undef, cql)
    blocksize = Vector{Vector{Vector{Int}}}(undef, cql)
    sb = Vector{Vector{Int}}(undef, cql)
    numb = Vector{Vector{Int}}(undef, cql)
    if tsupp == []
        tsupp = copy(fsupp)
        for i = 1:m
            tsupp = [tsupp gsupp[i]]
        end
        for i = 1:l
            tsupp = [tsupp hsupp[i]]
        end
        tsupp = sortslices(tsupp, dims=2)
        tsupp = unique(tsupp, dims=2)
    end
    status = ones(Int, cql)
    for i = 1:cql
        lc = length(I[i]) + length(J[i])
        ind = [issubset(findall(tsupp[:,j] .!= 0), cliques[i]) for j = 1:size(tsupp, 2)]
        supp = [tsupp[:, ind] UInt8(2)*basis[i][1]]
        supp = sortslices(supp, dims=2)
        supp = unique(supp, dims=2)
        blocks[i] = Vector{Vector{Vector{UInt16}}}(undef, lc+1)
        cl[i] = Vector{Int}(undef, lc+1)
        blocksize[i] = Vector{Vector{Int}}(undef, lc+1)
        sb[i] = Vector{Int}(undef, lc+1)
        numb[i] = Vector{Int}(undef, lc+1)
        blocks[i],cl[i],blocksize[i],sb[i],numb[i],status[i] = get_blocks(n, lc, supp, [gsupp[I[i]]; hsupp[J[i]]], [glt[I[i]]; hlt[J[i]]], basis[i], TS=TS, SO=SO, QUIET=QUIET)
    end
    return blocks,cl,blocksize,sb,numb,maximum(status)
end

function numele(a)
    return Int(sum(Int.(a).^2+a)/2)
end

function divide(a, lead, n, llead)
    return any(j->all(i->lead[i,j]<=a[i], 1:n), 1:llead)
end

function reminder(a, x, gb, n)
    remind = rem(prod(x.^a), gb)
    mon = monomials(remind)
    coe = coefficients(remind)
    lm = length(mon)
    supp = zeros(UInt8,n,lm)
    for i = 1:lm, j = 1:n
        @inbounds supp[j,i]=MultivariatePolynomials.degree(mon[i],x[j])
    end
    return lm,supp,coe
end

function add_poly!(model, vars, degree)
    mon = reverse(monomials(vars, 0:degree))
    coe = @variable(model, [1:length(mon)])
    p = coe'*mon
    return p,coe,mon
end
