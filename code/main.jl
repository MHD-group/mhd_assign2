#=
	Created On  : 2023-04-03 00:24
    Copyright © 2023 Junyi Xu <jyxu@mail.ustc.edu.cn>

    Distributed under terms of the MIT license.
=#

using PyCall
using LaTeXStrings

@pyimport matplotlib.pyplot as plt


# %%
function init1(x::AbstractVector, u::Vector)
	@. u[ x < -0.4] = 0.0
	@. u[-0.4 <= x < -0.2] = 1.0 - abs(x[-0.4 <= x < -0.2]+0.3) / 0.1
	@. u[-0.2 <= x < -0.1] = 0.0
	@. u[-0.1 <= x < -0.0] = 1.0
	@. u[ x >= 0.0 ] = 0.0
end

struct Cells
	x::AbstractVector{Float64}
	u::Vector{Float64} # u^n
	up::Vector{Float64} # u^(n+1) ; u plus
	function Cells(b::Float64=-1.0, e::Float64=2.0; step::Float64=0.01, f::Function=init1)
		x = range(b, e, step=step)
		u=similar(x)
		init1(x, u)
		up=similar(x)
		new(x, u , up)
	end
end

next(c::Cells, flg::Bool)::Vector = flg ? c.up : c.u
current(c::Cells, flg::Bool)::Vector = flg ? c.u : c.up

C=1 # C = Δt/Δx
function update(c::Cells, flg::Bool)
	up=next(c, flg) # u^(n+1)
	u=current(c, flg) # u^n
	up .= u - C * [u[1]-u[end], diff(u)...] # u_j^{n+1} = u_j^n - Δt/Δx * ( u_j^n - u_{j-1}^n )
	return !flg
end
# %%


function main()
	c=Cells()
	flg=Bool(1) # flag
	for i = 1:100
		flg=update(c, flg)
	end

	plt.plot(c.x, c.u)
	plt.show()
end

main()
