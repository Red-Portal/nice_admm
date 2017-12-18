
function f(x)
    result = x[1].^2 + x[2].^2 + x[3].^2 + x[4].^2 - 2 * x[1] - 3 * x[4]
    return result
end

function d_f(x)
    dx1 = 2 * x[1] - 2.0 
    dx2 = 2 * x[2]
    dx3 = 2 * x[3]
    dx4 = 2 * x[4] - 3.0
    return [dx1 dx2 dx3 dx4]::Matrix{Float64}
end

function tangent_subspace(N::Matrix{Float64})
    t_N = transpose(N)
    return inv(N * N') * N
end

function active_constraints(x)
    active_set = Matrix{Float64}(0, 4)
    g = Vector{Float64}(0)

    g1 = -2 * x[1] - x[2] - x[3] - 4 * x[4] + 7 
    g1_grad = [-2.0 -1.0 -1.0 -4]
    if(g1 >= 0)
        active_set = vcat(g1_grad, active_set)
        push!(g, g1)
    end

    g2 = -x[1] - x[2] - x[3] .^2 - x[4] + 5.1
    g2_grad = [-1.0 -1.0 -2.*x[3] -1]
    if(g2 >= 0)
        active_set = vcat(g2_grad, active_set)
        push!(g, g2)
    end

    g3 = -x[1]
    g3_grad = [-1.0 0.0 0.0 0.0]
    if(g3 >= 0)
        active_set = vcat(g3_grad, active_set)
        push!(g, g3)
    end

    g4 = -x[2]
    g4_grad = [0.0 -1.0 0.0 0.0]
    if(g4 >= 0)
        active_set = vcat(g4_grad, active_set)
        push!(g, g4)
    end

    g5 = -x[3]
    g5_grad = [0.0 0.0 -1.0 0.0]
    if(g5 >= 0)
        active_set = vcat(g5_grad, active_set)
        push!(g, g5)
    end

    g6 = -x[4]
    g6_grad = [0.0 0.0 0.0 -1.0]
    if(g6 >= 0)
        active_set = vcat(g6_grad, active_set)
        push!(g, g6)
    end

    return active_set, g
end

function compute_alpha(rate, projection, objective_value, gradient)
    return (-rate * objective_value) / dot(projection, gradient)
end

function is_parallel(x, y)
    return dot(x, x) * dot(y, y) - dot(x, y).^2
end

function main()
    starting_point = [2.0 2.0 1.0 0.0]

    max_iter = 10
    iteration = 0
    rate = 0.1
    point = starting_point
    objective = typemax(Float32)
    delta = typemax(Float32)

    for i = 1:max_iter
        grad = d_f(point)
        delta = -rate .* grad
        objective = f(point)

        if(norm(delta, 2) < eps(Float32))
            break
        end
        println("iteration: ", iteration, " objective: ", objective)

        point = point + delta

        N, value = active_constraints(point)

        if(length(N) != 0)
            grad = d_f(point)
            delta = -rate .* grad
            tangent = tangent_subspace(N)
            println("N: ", N)

            dim = size(N, 2)
            id = eye(dim, dim)::Matrix{Float64}
            projection_matrix = id - N' * tangent

            s = delta * projection_matrix'
            println("s: ", s)
            a = compute_alpha(rate, s, objective, grad)
            println("a: ", a)
            projection_move = a .* s

            restoration = -value' * tangent
            println("rest: ", restoration)

            update = projection_move + restoration

            if(is_parallel(update, delta) < eps(Float64))
                break
            end

            #println("kkt: ", check_parallel(update, delta))

            #println("point before: ", point)
            point = point + update
            
            println("update: ", update)
            println("point after: ", point)
        end
        iteration += 1
    end

    println("iterations: ", iteration)
    println("objective: ", objective)
    println("best: ", point)
end

main()

