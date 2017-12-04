
function f(x)
    result = (x[1] - 1).^2 +
        (x[2] - 2).^2  +
        (x[3] - 3).^2  +
        (x[4] - 4).^2

    return result
end

function d_f(x)
    dx1 = 2 * x[1] - 2 
    dx2 = 2 * x[2] - 4
    dx3 = 2 * x[3] - 6
    dx4 = 2 * x[4] - 8
    return [dx1 dx2 dx3 dx4]
end

function projection_direction(N::Matrix{Float64})
    t_N = transpose(N)
    dim = size(N, 2)
    id = eye(dim, dim)::Matrix{Float64}
    return id - N' * inv(N * N') * N
end


function max_mu(A, b, x, projection)
    mu = (- A*x' + b) ./ (A * projection')

    active_mu = filter(elem -> elem > 0, mu)

    if(length(active_mu) == 0)
        return typemax(Float32)
    elseif(length(active_mu) == 1)
        return active_mu[1] - eps(Float32)
    else
        return minimum(active_mu) - eps(Float32)
    end
end

function best_mu(x, s)
    x1, x2, x3, x4 = x
    s1, s2, s3, s4 = s
    
    up = s1 * (1 - x1) + s2 * (2 - x2) + s3 * (3 - x3) + s4 * (4 - x4) 
    down = s1.^2 + s2.^2 + s3.^2 + s4.^2
    result = up / down
    if(result >= 0)
        return result
    else
        return typemax(Float32)
    end
end

function active_constraints(A, b, x)
    status = A * transpose(x) - b

    result = Matrix{Float64}(0, 4)
    idx = 1
    for it in status
        if(it > 0)
            result = vcat(result, transpose(A[idx,:]))::Matrix{Float64}
        end
        idx += 1
    end
    return result::Matrix{Float64}
end

function main()
    starting_point = [1 1 1 1]
    #starting_point = [1 1 1.5 0.7]
    #starting_point = [0.5 1.0 1.5 2.0]

    const N = [1 1 1 1;
               3 3 2 1;
               -1 0 0 0;
               0 -1 0 0;
               0 0 -1 0;
               0 0 0 -1]
    const b = [5;
               10;
               0;
               0;
               0;
               0] 

    max_iter = 500
    iteration = 0
    rate = 0.05
    point = starting_point
    objective = typemax(Float32)
    delta = typemax(Float32)

    for i = 1:max_iter
        delta = -rate .* d_f(point)
        objective = f(point)

        if(norm(delta, 2) < 1e-10)
            break
        end
        
        println("objective: ", objective)

        next_point = point + delta
        tight = active_constraints(N, b, next_point)

        #println("point: ", point)
        #println("next point: ", next_point)
        #println("gradient: ", delta)

        if(length(tight) == 0)
            point = next_point
        else
            s = delta * projection_direction(tight)'
            #println("s: ", s)
            #println("x: ", point)

            #println(s)
            if(norm(s, 2) < 1e-10)
                #println("here!")
                max_rescale = max_mu(N, b, point, delta)
                if(max_rescale < 1e-10)
                    break
                else
                    point = point + max_rescale * delta 
                    continue
                end
            end

            best_mu_value = best_mu(point, s)
            max_mu_value = max_mu(N, b, point, s)

            mu = min(best_mu_value, max_mu_value)

            #println(update)
            if(mu < 1e-10)
                #println("mu! ", mu)
                break
            end

            #println("best: ", best_mu_value)
            #println("max: ", max_mu_value)

            point = point + mu * s
        end
        iteration += 1
    end

    println("iterations: ", iteration)
    println("objective: ", objective)
    println("best: ", point)
end

@time main()
#@time main()
