function get_basis(n, d; var=Vector(1:n))
    lb = binomial(length(var)+d, d)
    basis = zeros(UInt8, n, lb)
    i = 0
    t = 1
    while i < d+1
        t += 1
        if basis[var[end], t-1] == i
           if i < d
              basis[var[1], t] = i + 1
           end
           i += 1
        else
            j = findfirst(x->basis[var[x], t-1] != 0, 1:n)
            basis[:, t] = basis[:, t-1]
            if j == 1
               basis[var[1], t] -= 1
               basis[var[2], t] += 1
            else
               basis[var[1], t] = basis[var[j], t] - 1
               basis[var[j], t] = 0
               basis[var[j+1], t] += 1
            end
        end
    end
    return basis
end

function bfind(A, l, a)
    low = 1
    high = l
    while low <= high
        mid = Int(ceil(1/2*(low+high)))
        if ndims(A) == 2
            temp = A[:, mid]
        else
            temp = A[mid]
        end
        if temp == a
           return mid
        elseif temp < a
           low = mid + 1
        else
           high = mid - 1
        end
    end
    return 0
end

function get_moment(n, tsupp, lb, ub)
    ltsupp = size(tsupp, 2)
    moment = zeros(ltsupp)
    for i = 1:ltsupp
        moment[i] = prod([(ub[j]^(tsupp[j,i]+1)-lb[j]^(tsupp[j,i]+1))/(tsupp[j,i]+1) for j=1:n])
    end
    return moment
end

function get_tsupp(n, m, gsupp, glt, basis, blocks)
    blocksize = Vector{Vector{UInt16}}(undef, m+1)
    for i = 1:m+1
        blocksize[i] = length.(blocks[i])
    end
    cl = length.(blocksize)
    supp1 = zeros(UInt8, n, Int(sum(Int.(blocksize[1]).^2+blocksize[1])/2))
    k = 1
    for i = 1:cl[1], j = 1:blocksize[1][i], r = j:blocksize[1][i]
        supp1[:,k] = basis[1][:,blocks[1][i][j]] + basis[1][:,blocks[1][i][r]]
        k += 1
    end
    supp2 = zeros(UInt8, n, sum(glt[i]*Int(sum(Int.(blocksize[i+1]).^2+blocksize[i+1])/2) for i=1:m))
    l = 1
    for k = 1:m, i = 1:cl[k+1], j = 1:blocksize[k+1][i], r = j:blocksize[k+1][i], s = 1:glt[k]
        supp2[:,l] = basis[k+1][:,blocks[k+1][i][j]] + basis[k+1][:,blocks[k+1][i][r]] + gsupp[k][:,s]
        l += 1
    end
    tsupp = [supp1 supp2]
    tsupp = sortslices(tsupp,dims=2)
    tsupp = unique(tsupp,dims=2)
    return tsupp
end

function get_Lsupp(n, vsupp, fsupp, flt)
    Lsupp = zeros(UInt8, n, 1)
    for i = 1:length(flt)
        temp = zeros(UInt8, n)
        temp[i] = 1
        for j = 1:size(vsupp, 2)
            if vsupp[i, j] > 0
                for k = 1:flt[i]
                    Lsupp = [Lsupp vsupp[:,j]-temp+fsupp[i][:,k]]
                end
            end
        end
    end
    Lsupp = sortslices(Lsupp, dims=2)
    Lsupp = unique(Lsupp, dims=2)
    return Lsupp
end

function polys_info(p, x)
    m = length(p)
    n = length(x)
    dp = zeros(Int, m)
    pcoe = Vector{Vector{Float64}}(undef, m)
    psupp = Vector{Matrix{UInt8}}(undef, m)
    plt = Vector{Int}(undef, m)
    for i = 1:m
        dp[i] = maxdegree(p[i])
        mon = monomials(p[i])
        pcoe[i] = coefficients(p[i])
        plt[i] = length(mon)
        psupp[i] = zeros(UInt8, n, plt[i])
        for j = 1:plt[i], k = 1:n
            psupp[i][k, j] = MultivariatePolynomials.degree(mon[j], x[k])
        end
    end
    return psupp,pcoe,plt,dp
end

function poly_info(p, x)
    n = length(x)
    mon = monomials(p)
    plt = length(mon)
    pcoe = coefficients(p)
    psupp = zeros(UInt8, n, plt)
    for j = 1:plt, k = 1:n
        psupp[k, j] = MultivariatePolynomials.degree(mon[j], x[k])
    end
    return psupp,pcoe
end

function get_graph(tsupp::Array{UInt8, 2}, basis::Array{UInt8, 2})
    lb = size(basis,2)
    G = SimpleGraph(lb)
    ltsupp = size(tsupp, 2)
    for i = 1:lb, j = i+1:lb
        bi = basis[:,i] + basis[:,j]
        if bfind(tsupp, ltsupp, bi) != 0
           add_edge!(G, i, j)
        end
    end
    return G
end

function get_cgraph(tsupp::Array{UInt8, 2}, gsupp::Array{UInt8, 2}, glt, basis::Array{UInt8, 2})
    lb = size(basis, 2)
    G = SimpleGraph(lb)
    ltsupp = size(tsupp, 2)
    for i = 1:lb, j = i+1:lb
        r = 1
        while r <= glt
            bi = basis[:,i] + basis[:,j] + gsupp[:,r]
            if bfind(tsupp, ltsupp, bi) != 0
               break
            else
                r += 1
            end
        end
        if r <= glt
           add_edge!(G, i, j)
        end
    end
    return G
end

function get_vblocks(n::Int, m::Int, dv, tsupp1, tsupp, vsupp::Array{UInt8, 2}, fsupp, flt, gsupp::Vector{Array{UInt8, 2}}, glt, basis::Vector{Array{UInt8, 2}}; TS="block", SO=1, merge=false, md=3, QUIET=false)
    blocks = Vector{Vector{Vector{UInt16}}}(undef, m+1)
    if TS == false
        for k = 1:m+1
            blocks[k] = [[i for i=1:size(basis[k],2)]]
        end
        status = 1
    else
        status = 1
        blocks[1] = Vector{UInt16}[]
        qvsupp = vsupp
        for i = 1:SO
            G = get_graph(tsupp1, basis[1])
            if TS == "block"
                nblock = connected_components(G)
            else
                nblock = chordal_cliques!(G, method=TS)[1]
                if merge == true
                    nblock = clique_merge!(nblock, QUIET=true, d=md)[1]
                end
            end
            if nblock != blocks[1] || size(vsupp, 2) != size(qvsupp, 2)
                blocks[1] = nblock
                vsupp = qvsupp
                if i < SO
                    blocksize = length.(blocks[1])
                    tsupp = zeros(UInt8, n, Int(sum(Int.(blocksize).^2+blocksize)/2))
                    k = 1
                    for i = 1:length(blocks[1]), j = 1:blocksize[i], r = j:blocksize[i]
                        tsupp[:,k] = basis[1][:,blocks[1][i][j]] + basis[1][:,blocks[1][i][r]]
                        k += 1
                    end
                    tsupp = sortslices(tsupp, dims=2)
                    tsupp = unique(tsupp, dims=2)
                    qvsupp = tsupp[:, [sum(tsupp[:,i])<=dv for i=1:size(tsupp,2)]]
                    tsupp1 = [tsupp get_Lsupp(n, qvsupp, fsupp, flt)]
                    tsupp1 = sortslices(tsupp1, dims=2)
                    tsupp1 = unique(tsupp1, dims=2)
                end
            else
                println("No higher TSSOS hierarchy!")
                status = 0
                break
            end
        end
        if status == 1
            if QUIET == false
                blocksize = sort(length.(blocks[1]), rev=true)
                sb = unique(blocksize)
                numb = [sum(blocksize.== i) for i in sb]
                println("------------------------------------------------------")
                println("The sizes of PSD blocks:\n$sb\n$numb")
                println("------------------------------------------------------")
            end
            for k = 1:m
                G = get_cgraph(tsupp1, gsupp[k], glt[k], basis[k+1])
                blocks[k+1] = connected_components(G)
            end
        end
    end
    return blocks,vsupp,tsupp,status
end

function get_blocks(n::Int, m::Int, tsupp, gsupp::Vector{Array{UInt8, 2}}, glt, basis::Vector{Array{UInt8, 2}}; TS="block", SO=1, merge=false, md=3, QUIET=false)
    blocks = Vector{Vector{Vector{UInt16}}}(undef, m+1)
    blocksize = Vector{Vector{Int}}(undef, m+1)
    cl = Vector{Int}(undef, m+1)
    if TS == false
        for k = 1:m+1
            blocks[k] = [[i for i=1:size(basis[k],2)]]
            blocksize[k] = [size(basis[k],2)]
            cl[k] = 1          
        end
        sb = blocksize[1]
        numb = [1]
        status = 1
    else
        status = 1
        blocks[1] = Vector{UInt16}[]
        for i = 1:SO
            G = get_graph(tsupp, basis[1])
            if TS == "block"
                nblock = connected_components(G)
            else
                nblock = chordal_cliques!(G, method=TS)[1]
                if merge == true
                    nblock = clique_merge!(nblock, QUIET=true, d=md)[1]
                end
            end
            if nblock != blocks[1]
                blocks[1] = nblock
                if i < SO
                    blocksize[1] = length.(blocks[1])
                    tsupp = zeros(UInt8, n, numele(blocksize[1]))
                    k = 1
                    for i = 1:length(blocks[1]), j = 1:blocksize[1][i], r = j:blocksize[1][i]
                        tsupp[:,k] = basis[1][:,blocks[1][i][j]] + basis[1][:,blocks[1][i][r]]
                        k += 1
                    end
                    tsupp = sortslices(tsupp, dims=2)
                    tsupp = unique(tsupp, dims=2)
                end
            else
                if QUIET == false
                    println("No higher TSSOS hierarchy!")
                end
                status = 0
                sb = numb = nothing
                break
            end
        end
        if status == 1
            blocksize[1] = length.(blocks[1])
            cl[1] = length(blocksize[1])
            bz = sort(blocksize[1], rev=true)
            sb = unique(bz)
            numb = [sum(bz.== i) for i in sb]
            if QUIET == false
                println("------------------------------------------------------")
                println("The sizes of PSD blocks:\n$sb\n$numb")
                println("------------------------------------------------------")
            end
            for k = 1:m
                G = get_cgraph(tsupp, gsupp[k], glt[k], basis[k+1])
                blocks[k+1] = connected_components(G)
                blocksize[k+1] = length.(blocks[k+1])
                cl[k+1] = length(blocksize[k+1])
            end
        end
    end
    return blocks,cl,blocksize,sb,numb,status
end

function add_putinar!(model, m, tsupp, gsupp, gcoe, glt, basis, blocks; numeq=0)
    ltsupp = size(tsupp, 2)
    cons = [AffExpr(0) for i=1:ltsupp]
    for i = 1:length(blocks[1])
        bs = length(blocks[1][i])
        if bs == 1
           pos = @variable(model, lower_bound=0)
           bi = basis[1][:,blocks[1][i][1]] + basis[1][:,blocks[1][i][1]]
           Locb = bfind(tsupp, ltsupp, bi)
           @inbounds add_to_expression!(cons[Locb], pos)
        else
           pos = @variable(model, [1:bs, 1:bs], PSD)
           for j = 1:bs, r = j:bs
               bi = basis[1][:,blocks[1][i][j]] + basis[1][:,blocks[1][i][r]]
               Locb = bfind(tsupp, ltsupp, bi)
               if j == r
                  @inbounds add_to_expression!(cons[Locb], pos[j,r])
               else
                  @inbounds add_to_expression!(cons[Locb], 2, pos[j,r])
               end
           end
        end
    end
    for k = 1:m, i = 1:length(blocks[k+1])
        bs = length(blocks[k+1][i])
        if bs == 1
            if k <= m-numeq
                pos = @variable(model, lower_bound=0)
            else
                pos = @variable(model)
            end
            for s = 1:glt[k]
                bi = basis[k+1][:,blocks[k+1][i][1]] + basis[k+1][:,blocks[k+1][i][1]] + gsupp[k][:,s]
                Locb = bfind(tsupp,ltsupp,bi)
                @inbounds add_to_expression!(cons[Locb], gcoe[k][s], pos)
            end
        else
            if k <= m-numeq
                pos = @variable(model, [1:bs, 1:bs], PSD)
            else
                pos = @variable(model, [1:bs, 1:bs], Symmetric)
            end
            for j = 1:bs, r = j:bs, s = 1:glt[k]
                bi = basis[k+1][:,blocks[k+1][i][j]] + basis[k+1][:,blocks[k+1][i][r]] + gsupp[k][:,s]
                Locb = bfind(tsupp, ltsupp, bi)
                if j == r
                   @inbounds add_to_expression!(cons[Locb], gcoe[k][s], pos[j,r])
                else
                   @inbounds add_to_expression!(cons[Locb], 2*gcoe[k][s], pos[j,r])
                end
            end
        end
    end
    return cons
end
