#=
	Created On  : 2023-04-03 00:24
    Copyright © 2023 Junyi Xu <jyxu@mail.ustc.edu.cn>

    Distributed under terms of the MIT license.
=#

using CircularArrays
using PyCall
using LaTeXStrings

@pyimport matplotlib.pyplot as plt
@pyimport matplotlib
# https://stackoverflow.com/questions/3899980/how-to-change-the-font-size-on-a-matplotlib-plot
matplotlib.rc("font", size=20)

# %%
function init1(x::AbstractVector, u::AbstractVector)
	@. u[ x < -0.4] = 0.0
	@. u[-0.4 <= x < -0.2] = 1.0 - abs(x[-0.4 <= x < -0.2]+0.3) / 0.1
	@. u[-0.2 <= x < -0.1] = 0.0
	@. u[-0.1 <= x < -0.0] = 1.0
	@. u[ x >= 0.0 ] = 0.0
end

function init2(x::AbstractVector, u::AbstractVector) # Burgers 方程 初始化
	@. u[ x < -0.8] = 1.8
	@. u[-0.8 <= x < -0.3] = 1.4 + 0.4*cos(2π*(x[-0.8 <= x < -0.3]+0.8))
	@. u[-0.3 <= x < 0.0] = 1.0
	@. u[ x >= 0.0 ] = 1.8
end

struct Cells
	x::AbstractVector{Float64}
	u::CircularVector{Float64} # u^n
	up::CircularVector{Float64} # u^(n+1) ; u plus
	function Cells(b::Float64=-1.0, e::Float64=2.0; step::Float64=0.01, init::Function=init1)
		x = range(b, e, step=step)
		u=CircularVector(0.0, length(x))
		init(x, u)
		up=similar(u)
		new(x, u , up)
	end
end
Cells(Δ::Float64)=Cells(-1.0, 2.0, step=Δ)
Cells(init::Function)=Cells(-1.0, 2.0, init=init)
Cells(b::Float64, e::Float64, Δ::Float64)=Cells(b, e, step=Δ)

next(c::Cells, flg::Bool)::CircularVector = flg ? c.up : c.u
current(c::Cells, flg::Bool)::CircularVector = flg ? c.u : c.up

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


function upwind(up::CircularVector, u::CircularVector)
	for i = 1:length(u)
		up[i] = u[i] - C * (u[i] -u[i-1])  # u_j^{n+1} = u_j^n - Δt/Δx * ( u_j^n - u_{j-1}^n )
	end
end


function upwind2(up::CircularVector, u::CircularVector)
	for i = 1:length(u)
		up[i] = u[i] - 0.5C * (u[i]^2 -u[i-1]^2)  # u_j^{n+1} = u_j^n - Δt/Δx * ( u_j^n - u_{j-1}^n )
	end
end

function lax_wendroff(up::CircularVector, u::CircularVector)
	for j = 1:length(u)
		up[j] = u[j] - 0.5 * C * ( u[j+1] - u[j-1] ) + 0.5 * C^2 * ( u[j+1] - 2u[j] + u[j-1] )
	end
end

function limiter(up::CircularVector, u::CircularVector)
	for i = 1:length(u)
		up[i] = u[i] - C * (u[i] - u[i-1]) - 0.5 * C * (1 - C) *
			( minmod(u[i]-u[i-1], u[i+1]-u[i]) - minmod(u[i-1]-u[i-2], u[i]-u[i-1]) )
	end
end

function limiter2(up::CircularVector, un::CircularVector)
	u = @. 0.5un^2
	for i = 1:length(u)
		up[i] = un[i] - C * (u[i] - u[i-1]) - 0.5 * C * (1 - C) *
			( minmod(u[i]-u[i-1], u[i+1]-u[i]) - minmod(u[i-1]-u[i-2], u[i]-u[i-1]) )
	end
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
