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
	function Cells(b::Float64=-1.0, e::Float64=2.0; step::Float64=Δx, f::Function=init1)
		x = range(b, e, step=step)
		u=similar(x)
		f(x, u)
		up=similar(x)
		new(x, u , up)
	end
end

next(c::Cells, flg::Bool)::Vector = flg ? c.up : c.u
current(c::Cells, flg::Bool)::Vector = flg ? c.u : c.up

C = 0.05
# C = Δt/Δx
Δt = Δx * C

function upwind(u::Vector)
	u - C * [u[1]-u[end], diff(u)...] # u_j^{n+1} = u_j^n - Δt/Δx * ( u_j^n - u_{j-1}^n )
end

function update(c::Cells, flg::Bool)
	up=next(c, flg) # u^(n+1)
	u=current(c, flg) # u^n
	up .= upwind(u)
	return !flg
end
# %%

t=0.5

function main()
	init = init1
	c=Cells(f=init)

	# plt.subplot(211)
	plt.plot(c.x, c.u, "-.k", linewidth=0.2, label="init")
	# plt.plot(c.x, current(c,!flg), "-.k", linewidth=0.2, label="init")

	flg=Bool(1) # flag
	for _ = 1:round(Int, t/Δt)
		flg=update(c, flg)
	end

	# plt.subplot(212)
	plt.plot(c.x, c.up, label="up")
	plt.show()

end

main()
