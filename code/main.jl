#=
	Created On  : 2023-04-03 00:24
    Copyright © 2023 Junyi Xu <jyxu@mail.ustc.edu.cn>

    Distributed under terms of the MIT license.
=#

using PyCall
using LaTeXStrings

@pyimport matplotlib.pyplot as plt

Δx=0.01

# %%
function init1(x::AbstractVector, u::Vector)
	@. u[ x < -0.4] = 0.0
	@. u[-0.4 <= x < -0.2] = 1.0 - abs(x[-0.4 <= x < -0.2]+0.3) / 0.1
	@. u[-0.2 <= x < -0.1] = 0.0
	@. u[-0.1 <= x < -0.0] = 1.0
	@. u[ x >= 0.0 ] = 0.0
end

function init2(x::AbstractVector, u::Vector) # Burgers 方程 初始化
	@. u[ x < -0.8] = 1.8
	@. u[-0.8 <= x < -0.3] = 1.4 + 0.4*cos(2π*(x[-0.8 <= x < -0.3]+0.8))
	@. u[-0.3 <= x < 0.0] = 1.0
	@. u[ x >= 0.0 ] = 1.8
end

struct Cells
	x::AbstractVector{Float64}
	u::Vector{Float64} # u^n
	up::Vector{Float64} # u^(n+1) ; u plus
	function Cells(b::Float64=-1.0, e::Float64=2.0; step::Float64=Δx, init::Function=init1)
		x = range(b, e, step=step)
		u=similar(x)
		init(x, u)
		up=similar(x)
		new(x, u , up)
	end
end
Cells(Δ::Float64)=Cells(-1.0, 2.0, step=Δ)
Cells(init::Function)=Cells(-1.0, 2.0, init=init)
Cells(b::Float64, e::Float64, Δ::Float64)=Cells(b, e, step=Δ)

next(c::Cells, flg::Bool)::Vector = flg ? c.up : c.u
current(c::Cells, flg::Bool)::Vector = flg ? c.u : c.up

# %%
C = 1.0
# C = Δt/Δx
Δt = Δx * C


function upwind(up::Vector, u::Vector)
	up .= u - C * [u[1]-u[end], diff(u)...] # u_j^{n+1} = u_j^n - Δt/Δx * ( u_j^n - u_{j-1}^n )
end


function lax_wendroff(up::Vector, u::Vector)
	for j = 2:length(u)-1
		up[j] = u[j] - 0.5 * C * ( u[j+1] - u[j-1] ) + 0.5 * C^2 * ( u[j+1] - 2u[j] + u[j-1] )
	end
	up[1] = u[1] - 0.5 * C * ( u[2] - u[end] ) + 0.5 * C^2 * ( u[2] - 2u[1] + u[end] )
	up[end] = u[end] - 0.5 * C * ( u[1] - u[end-1] ) + 0.5 * C^2 * ( u[1] - 2u[end] + u[end-1] )
end


function update(c::Cells, flg::Bool, f::Function)
	up=next(c, flg) # u^(n+1)
	u=current(c, flg) # u^n
	f(up, u)
	return !flg
end
# %%

t=0.5

function main()
	c=Cells(init1)
	plt.plot(c.x, c.u, "-.k", linewidth=0.2, label="init")

	f = upwind
	flg=true # flag
	for _ = 1:round(Int, t/Δt)
		flg=update(c, flg, f)
	end

	plt.title("time = "*string(t)*", "*string(f))
	plt.plot(c.x, c.up, "--.", label="up")
	plt.show()
	# plt.savefig("../figures/problem1_"*string(f)*string(C)*".pdf", bbox_inches="tight")

end

if abspath(PROGRAM_FILE) == @__FILE__
    main()
end
