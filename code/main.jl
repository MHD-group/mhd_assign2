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
	function Cells(b::Float64=-1.0, e::Float64=2.0; step::Float64=0.01, init::Function=init1)
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

function update!(c::Cells, flg::Bool, f::Function)
	up=next(c, flg) # u^(n+1)
	u=current(c, flg) # u^n
	f(up, u)
	return !flg
end

# %%

function minmod(a::AbstractFloat, b::AbstractFloat)::AbstractFloat
	if sign(a) * sign(b) > 0
		if abs(a) < abs(b)
			return a
		end
		return b
	end
	return 0
end


# %%
C = 0.2
Δx= 0.007
# C = Δt/Δx
Δt = Δx * C


function upwind(up::Vector, u::Vector)
	for i = 2:length(u)
		up[i] = u[i] - C * (u[i] -u[i-1])  # u_j^{n+1} = u_j^n - Δt/Δx * ( u_j^n - u_{j-1}^n )
	end
	up[1] = u[1] - C * (u[1] -u[end])
end


function upwind2(up::Vector, u::Vector)
	for i = 2:length(u)
		up[i] = u[i] - 0.5C * (u[i]^2 -u[i-1]^2)  # u_j^{n+1} = u_j^n - Δt/Δx * ( u_j^n - u_{j-1}^n )
	end
	up[1] = u[1] - 0.5C * (u[1]^2 -u[end]^2)
end



function lax_wendroff(up::Vector, u::Vector)
	for j = 2:length(u)-1
		up[j] = u[j] - 0.5 * C * ( u[j+1] - u[j-1] ) + 0.5 * C^2 * ( u[j+1] - 2u[j] + u[j-1] )
	end
	up[1] = u[1] - 0.5 * C * ( u[2] - u[end] ) + 0.5 * C^2 * ( u[2] - 2u[1] + u[end] )
	up[end] = u[end] - 0.5 * C * ( u[1] - u[end-1] ) + 0.5 * C^2 * ( u[1] - 2u[end] + u[end-1] )
end

function limiter(up::Vector, u::Vector)
	for i = 3:length(u)-1
		up[i] = u[i] - C * (u[i] - u[i-1]) - 0.5 * C * (1 - C) *
			( minmod(u[i]-u[i-1], u[i+1]-u[i]) - minmod(u[i-1]-u[i-2], u[i]-u[i-1]) )
	end
	up[1] = u[1] - C * (u[1] - u[end]) - 0.5 * C * (1 - C) *
		( minmod(u[1]-u[end], u[2]-u[1]) - minmod(u[end]-u[end-1], u[1]-u[end]) )
	up[2] = u[2] - C * (u[2] - u[1]) - 0.5 * C * (1 - C) *
		( minmod(u[2]-u[1], u[3]-u[2]) - minmod(u[1]-u[end], u[2]-u[1]) )
	up[end] = u[end] - C * (u[end] - u[end-1]) - 0.5 * C * (1 - C) *
		( minmod(u[end]-u[end-1], u[1]-u[end]) - minmod(u[end-1]-u[end-2], u[end]-u[end-1]) )
end

function limiter2(up::Vector, un::Vector)
	u = @. 0.5un^2
	for i = 3:length(u)-1
		up[i] = un[i] - C * (u[i] - u[i-1]) - 0.5 * C * (1 - C) *
			( minmod(u[i]-u[i-1], u[i+1]-u[i]) - minmod(u[i-1]-u[i-2], u[i]-u[i-1]) )
	end
	up[1] = un[1] - C * (u[1] - u[end]) - 0.5 * C * (1 - C) *
		( minmod(u[1]-u[end], u[2]-u[1]) - minmod(u[end]-u[end-1], u[1]-u[end]) )
	up[2] = un[2] - C * (u[2] - u[1]) - 0.5 * C * (1 - C) *
		( minmod(u[2]-u[1], u[3]-u[2]) - minmod(u[1]-u[end], u[2]-u[1]) )
	up[end] = un[end] - C * (u[end] - u[end-1]) - 0.5 * C * (1 - C) *
		( minmod(u[end]-u[end-1], u[1]-u[end]) - minmod(u[end-1]-u[end-2], u[end]-u[end-1]) )
end



# %%
t=1

function main()
	f = limiter2
	plt.figure(figsize=(10,3))
	c=Cells(step=Δx, init=init2)
	# plt.subplot(length(functions),1,i)

	# plt.rcParams["font.size"]=30
	# plt.rcParams["lines.color"]="r"
	plt.plot(c.x, c.u, "-.k", linewidth=0.2, label="init")

	flg=true # flag
	for _ = 1:round(Int, t/Δt)-3
		flg=update!(c, flg, f)
	end

	plt.title("time = "*string(t)*", "*"Minmod")
	# plt.plot(c.x, c.up, marker="o", markeredgewidth=0.4, markersize=4,  markerfacecolor="none", label="up")
	plt.plot(c.x, c.up, color="navy", marker="o", markeredgecolor="purple", markeredgewidth=0.4, markersize=4,  markerfacecolor="none", label="up")
	# plt.savefig("../figures/problem1_"*string(f)*string(C)*".pdf", bbox_inches="tight")
	plt.savefig("../figures/problem2_"*string(f)*string(t)*".pdf", bbox_inches="tight")
	plt.show()
end
main()
