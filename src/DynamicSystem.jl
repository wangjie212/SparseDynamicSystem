function MPI_first(f, g, x, d, lb, ub; TS="block", merge=false, β=1, QUIET=false, solver="Mosek")
    n,m,fsupp,fcoe,flt,mdf,gsupp,gcoe,glt,dg,basis,tsupp=preprocess(f, g, x, d)
    blocks,cl,blocksize,_,_,_=get_cblocks!(m,tsupp,gsupp,glt,basis,TS=TS,merge=merge,QUIET=true)
    tsupp=get_tsupp(n,m,gsupp,gcoe,glt,basis,blocks,cl,blocksize)
    blocks,cl,blocksize,_,_,_=get_cblocks!(m,tsupp,gsupp,glt,basis,TS=TS,merge=merge)
    tsupp=get_tsupp(n,m,gsupp,gcoe,glt,basis,blocks,cl,blocksize)
    vsupp=tsupp[:, [sum(tsupp[:, i])<2d+2-mdf for i=1:size(tsupp,2)]]
    qtsupp=get_qtsupp(n, vsupp, fsupp, flt)
    if size(qtsupp, 2)==size(tsupp, 2)
        qbasis=basis
        qblocks,qcl,qblocksize=blocks,cl,blocksize
    else
        qbasis=Vector{Array{UInt8,2}}(undef, m+1)
        qbasis[1]=get_basis(n, d)
        for i=1:m
            qbasis[i+1]=get_basis(n, d-Int(ceil(dg[i]/2)))
        end
        qblocks,qcl,qblocksize,_,_,_=get_cblocks!(m,qtsupp,gsupp,glt,qbasis,TS=TS,merge=merge)
        qtsupp=get_tsupp(n,m,gsupp,gcoe,glt,qbasis,qblocks,qcl,qblocksize)
    end
    moment=get_moment(n, tsupp, lb, ub)
    opt,wcoe=MPI_SDP(n,m,fsupp,fcoe,flt,gsupp,gcoe,glt,basis,tsupp,qtsupp,vsupp,blocks,cl,blocksize,qbasis,qblocks,qcl,qblocksize,moment,β=β,QUIET=QUIET,solver=solver)
    ind=[abs(wcoe[i])>1e-6 for i=1:size(tsupp, 2)]
    wsupp=[prod(x.^tsupp[:,i]) for i=1:size(tsupp, 2)]
    w=wcoe[ind]'*wsupp[ind]
    return opt,w
end

function preprocess(f, g, x, d)
    n=length(x)
    df=zeros(Int, n)
    fcoe=Vector{Vector{Float64}}(undef, n)
    fsupp=Vector{Array{UInt8,2}}(undef, n)
    flt=Vector{Int}(undef, n)
    for i=1:n
        df[i]=maxdegree(f[i])
        mon=monomials(f[i])
        fcoe[i]=coefficients(f[i])
        flt[i]=length(mon)
        fsupp[i]=zeros(UInt8, n, flt[i])
        for j=1:flt[i], k=1:n
            fsupp[i][k, j]=MultivariatePolynomials.degree(mon[j], x[k])
        end
    end
    mdf=maximum(df)
    m=length(g)
    dg=zeros(Int, m)
    gcoe=Vector{Vector{Float64}}(undef, m)
    gsupp=Vector{Array{UInt8,2}}(undef, m)
    glt=Vector{Int}(undef, m)
    for i=1:m
        dg[i]=maxdegree(g[i])
        mon=monomials(g[i])
        gcoe[i]=coefficients(g[i])
        glt[i]=length(mon)
        gsupp[i]=zeros(UInt8, n, glt[i])
        for j=1:glt[i], k=1:n
            gsupp[i][k, j]=MultivariatePolynomials.degree(mon[j], x[k])
        end
    end
    tsupp=copy(gsupp[1])
    for i=2:m
        tsupp=[tsupp gsupp[i]]
    end
    tsupp=sortslices(tsupp, dims=2)
    tsupp=unique(tsupp, dims=2)
    ltsupp=size(tsupp, 2)
    vsupp=[prod(x.^tsupp[:,i]) for i=1:ltsupp]
    v=rand(ltsupp)'*vsupp
    vf=sum([differentiate(v, x[i])*f[i] for i=1:n])
    mon=monomials(vf)
    vfsupp=zeros(UInt8, n, length(mon))
    for j=1:length(mon), k=1:n
        vfsupp[k, j]=MultivariatePolynomials.degree(mon[j], x[k])
    end
    tsupp=[tsupp vfsupp]
    basis=Vector{Array{UInt8,2}}(undef, m+1)
    basis[1]=get_basis(n, d)
    for i=1:m
        basis[i+1]=get_basis(n, d-Int(ceil(dg[i]/2)))
    end
    # tsupp=[tsupp UInt8(2)*basis[1]]
    tsupp=sortslices(tsupp, dims=2)
    tsupp=unique(tsupp, dims=2)
    return n,m,fsupp,fcoe,flt,mdf,gsupp,gcoe,glt,dg,basis,tsupp
end

function get_tsupp(n,m,gsupp,gcoe,glt,basis,blocks,cl,blocksize)
    supp1=zeros(UInt8, n, Int(sum(Int.(blocksize[1]).^2+blocksize[1])/2))
    k=1
    for i=1:cl[1], j=1:blocksize[1][i], r=j:blocksize[1][i]
        supp1[:,k]=basis[1][:,blocks[1][i][j]]+basis[1][:,blocks[1][i][r]]
        k+=1
    end
    supp2=zeros(UInt8, n, sum(glt[i]*Int(sum(blocksize[i+1].^2+blocksize[i+1])/2) for i=1:m))
    l=1
    for k=1:m, i=1:cl[k+1], j=1:blocksize[k+1][i], r=j:blocksize[k+1][i], s=1:glt[k]
        supp2[:,l]=basis[k+1][:,blocks[k+1][i][j]]+basis[k+1][:,blocks[k+1][i][r]]+gsupp[k][:,s]
        l+=1
    end
    tsupp=[supp1 supp2]
    tsupp=sortslices(tsupp,dims=2)
    tsupp=unique(tsupp,dims=2)
    return tsupp
end

function get_qtsupp(n, vsupp, fsupp, flt)
    qtsupp=copy(vsupp)
    for i=1:n
        temp=zeros(UInt8, n)
        temp[i]=1
        for j=1:size(vsupp, 2)
            if vsupp[i, j]>0
                for k=1:flt[i]
                    qtsupp=[qtsupp vsupp[:, j]-temp+fsupp[i][:, k]]
                end
            end
        end
    end
    qtsupp=sortslices(qtsupp,dims=2)
    qtsupp=unique(qtsupp,dims=2)
    return qtsupp
end

function MPI_SDP(n, m, fsupp, fcoe, flt, gsupp, gcoe, glt, basis, tsupp, qtsupp, vsupp, blocks, cl, blocksize, qbasis, qblocks, qcl, qblocksize, moment; β=1, QUIET=false, solver="Mosek")
    ltsupp=size(tsupp,2)
    lqtsupp=size(qtsupp,2)
    if solver=="COSMO"
        model=Model(optimizer_with_attributes(COSMO.Optimizer))
    else
        model=Model(optimizer_with_attributes(Mosek.Optimizer))
    end
    set_optimizer_attribute(model, MOI.Silent(), QUIET)
    coeff1,pos1,gpos1=add_putinar!(model,m,qtsupp,gsupp,gcoe,glt,qbasis,qblocks,qcl,qblocksize)
    coeff2,pos2,gpos2=add_putinar!(model,m,tsupp,gsupp,gcoe,glt,basis,blocks,cl,blocksize)
    coeff3,pos3,gpos3=add_putinar!(model,m,tsupp,gsupp,gcoe,glt,basis,blocks,cl,blocksize)
    vcoe=@variable(model, [1:size(vsupp,2)])
    vfcoe=dvf(n, qtsupp, vsupp, vcoe, fsupp, fcoe, flt, β=β)
    @constraint(model, coeff1.==vfcoe)
    coeff4=copy(coeff2)
    coeff4[1]-=1
    for i=1:size(vsupp, 2)
        Locb=bfind(tsupp,ltsupp,vsupp[:,i])
        coeff4[Locb]-=vcoe[i]
    end
    @constraint(model, coeff4.==coeff3)
    ind=[abs(moment[i])>1e-8 for i=1:ltsupp]
    @objective(model, Min, moment[ind]'*coeff2[ind])
    optimize!(model)
    status=termination_status(model)
    if status!=MOI.OPTIMAL
       println("termination status: $status")
       status=primal_status(model)
       println("solution status: $status")
    end
    wcoe=value.(coeff2)
    objv=objective_value(model)
    return objv,wcoe
end

function add_putinar!(model,m,tsupp,gsupp,gcoe,glt,basis,blocks,cl,blocksize)
    ltsupp=size(tsupp, 2)
    cons=[AffExpr(0) for i=1:ltsupp]
    pos=Vector{Union{VariableRef,Symmetric{VariableRef}}}(undef, cl[1])
    for i=1:cl[1]
        bs=blocksize[1][i]
        if bs==1
           pos[i]=@variable(model, lower_bound=0)
           bi=basis[1][:,blocks[1][i][1]]+basis[1][:,blocks[1][i][1]]
           Locb=bfind(tsupp,ltsupp,bi)
           @inbounds add_to_expression!(cons[Locb], pos[i])
        else
           pos[i]=@variable(model, [1:bs, 1:bs], PSD)
           for j=1:bs, r=j:bs
               bi=basis[1][:,blocks[1][i][j]]+basis[1][:,blocks[1][i][r]]
               Locb=bfind(tsupp,ltsupp,bi)
               if j==r
                  @inbounds add_to_expression!(cons[Locb], pos[i][j,r])
               else
                  @inbounds add_to_expression!(cons[Locb], 2, pos[i][j,r])
               end
           end
        end
    end
    gpos=Vector{Vector{Union{VariableRef,Symmetric{VariableRef}}}}(undef, m)
    for k=1:m
        gpos[k]=Vector{Union{VariableRef,Symmetric{VariableRef}}}(undef, cl[k+1])
        for i=1:cl[k+1]
            bs=blocksize[k+1][i]
            if bs==1
                gpos[k][i]=@variable(model, lower_bound=0)
                for s=1:glt[k]
                    bi=basis[k+1][:,blocks[k+1][i][1]]+basis[k+1][:,blocks[k+1][i][1]]+gsupp[k][:,s]
                    Locb=bfind(tsupp,ltsupp,bi)
                    @inbounds add_to_expression!(cons[Locb], gcoe[k][s], gpos[k][i])
                end
            else
                gpos[k][i]=@variable(model, [1:bs, 1:bs], PSD)
                for j=1:bs, r=j:bs, s=1:glt[k]
                    bi=basis[k+1][:,blocks[k+1][i][j]]+basis[k+1][:,blocks[k+1][i][r]]+gsupp[k][:,s]
                    Locb=bfind(tsupp,ltsupp,bi)
                    if j==r
                       @inbounds add_to_expression!(cons[Locb], gcoe[k][s], gpos[k][i][j,r])
                    else
                       @inbounds add_to_expression!(cons[Locb], 2*gcoe[k][s], gpos[k][i][j,r])
                    end
                end
            end
        end
    end
    return cons,pos,gpos
end

function dvf(n, qtsupp, vsupp, vcoe, fsupp, fcoe, flt; β=1)
    lqtsupp=size(qtsupp, 2)
    vfcoe=[AffExpr(0) for i=1:lqtsupp]
    lvsupp=size(vsupp, 2)
    for j=1:lvsupp
        locb=bfind(qtsupp, lqtsupp, vsupp[:, j])
        @inbounds add_to_expression!(vfcoe[locb], β, vcoe[j])
    end
    for i=1:n
        temp=zeros(UInt8, n)
        temp[i]=1
        for j=1:lvsupp
            if vsupp[i, j]>0
                for k=1:flt[i]
                    locb=bfind(qtsupp, lqtsupp, vsupp[:, j]-temp+fsupp[i][:, k])
                    vfcoe[locb]-=vsupp[i, j]*fcoe[i][k]*vcoe[j]
                    # @inbounds add_to_expression!(vfcoe[locb], -vsupp[i, j]*fcoe[i][k], vcoe[j])
                end
            end
        end
    end
    return vfcoe
end

function get_basis(n, d)
    lb=binomial(n+d,d)
    basis=zeros(UInt8,n,lb)
    i=0
    t=1
    while i<d+1
        t+=1
        if basis[n,t-1]==i
           if i<d
              basis[1,t]=i+1
           end
           i+=1
        else
            j=findfirst(x->basis[x,t-1]!=0, 1:n)
            basis[:,t]=basis[:,t-1]
            if j==1
               basis[1,t]-=1
               basis[2,t]+=1
            else
               basis[1,t]=basis[j,t]-1
               basis[j,t]=0
               basis[j+1,t]+=1
            end
        end
    end
    return basis
end

function bfind(A, l, a)
    low=1
    high=l
    while low<=high
        mid=Int(ceil(1/2*(low+high)))
        temp=A[:, mid]
        if temp==a
           return mid
        elseif temp<a
           low=mid+1
        else
           high=mid-1
        end
    end
    return 0
end

function get_moment(n, tsupp, lb, ub)
    ltsupp=size(tsupp, 2)
    moment=zeros(ltsupp)
    for i=1:ltsupp
        moment[i]=prod([(ub[j]^(tsupp[j,i]+1)-lb[j]^(tsupp[j,i]+1))/(tsupp[j,i]+1) for j=1:n])
    end
    return moment
end

function get_graph(tsupp::Array{UInt8, 2},basis::Array{UInt8, 2})
    lb=size(basis,2)
    G=SimpleGraph(lb)
    ltsupp=size(tsupp,2)
    for i = 1:lb, j = i+1:lb
        bi=basis[:,i]+basis[:,j]
        if bfind(tsupp,ltsupp,bi)!=0
           add_edge!(G,i,j)
        end
    end
    return G
end

function get_cgraph(tsupp::Array{UInt8, 2},gsupp::Array{UInt8, 2},glt,basis::Array{UInt8, 2})
    lb=size(basis,2)
    G=SimpleGraph(lb)
    ltsupp=size(tsupp,2)
    for i = 1:lb, j = i+1:lb
        r=1
        while r<=glt
            bi=basis[:,i]+basis[:,j]+gsupp[:,r]
            if bfind(tsupp,ltsupp,bi)!=0
               break
            else
                r+=1
            end
        end
        if r<=glt
           add_edge!(G,i,j)
        end
    end
    return G
end

function get_cblocks!(m::Int,tsupp::Array{UInt8, 2},gsupp::Vector{Array{UInt8, 2}},glt,basis::Vector{Array{UInt8, 2}};ub=[],sizes=[],TS="block",merge=false,QUIET=false)
    blocks=Vector{Vector{Vector{UInt16}}}(undef, m+1)
    blocksize=Vector{Vector{UInt16}}(undef, m+1)
    cl=Vector{UInt16}(undef, m+1)
    if TS==false
        blocks[1]=[[i for i=1:size(basis[1],2)]]
        blocksize[1]=[size(basis[1],2)]
        cl[1]=1
        for k=1:m
            blocks[k+1]=[[i for i=1:size(basis[k+1],2)]]
            blocksize[k+1]=[size(basis[k+1],2)]
            cl[k+1]=1
        end
        status=1
        nub=Int.(blocksize[1])
        nsizes=[1]
        if QUIET==false
            println("------------------------------------------------------")
            println("The sizes of PSD blocks:\n$nub\n$nsizes")
            println("------------------------------------------------------")
        end
    else
        G=get_graph(tsupp, basis[1])
        if TS=="block"
            blocks[1]=connected_components(G)
            blocksize[1]=length.(blocks[1])
            cl[1]=length(blocksize[1])
        else
            blocks[1],cl[1],blocksize[1]=chordal_cliques!(G, method=TS)
            if merge==true
                blocks[1],cl[1],blocksize[1]=clique_merge!(blocks[1],cl[1],QUIET=true)
            end
        end
        nub=Int.(unique(blocksize[1]))
        nsizes=[sum(blocksize[1].== i) for i in nub]
        if isempty(ub)||nub!=ub||nsizes!=sizes
            status=1
            if QUIET==false
                println("------------------------------------------------------")
                println("The sizes of PSD blocks:\n$nub\n$nsizes")
                println("------------------------------------------------------")
            end
            for k=1:m
                G=get_cgraph(tsupp,gsupp[k],glt[k],basis[k+1])
                if TS=="block"
                    blocks[k+1]=connected_components(G)
                    blocksize[k+1]=length.(blocks[k+1])
                    cl[k+1]=length(blocksize[k+1])
                else
                    blocks[k+1],cl[k+1],blocksize[k+1]=chordal_cliques!(G, method=TS)
                    if merge==true
                        blocks[k+1],cl[k+1],blocksize[k+1]=clique_merge!(blocks[k+1],cl[k+1],QUIET=true)
                    end
                end
            end
        else
            status=0
            if QUIET==false
                println("No higher TSSOS hierarchy!")
            end
        end
    end
    return blocks,cl,blocksize,nub,nsizes,status
end
