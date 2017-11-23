
function f(input)
    x1, x2, x3, x4 = input
    result = x1.^2 + x2.^2 + x3.^2 + x4.^2 - 2 * x1 - 3 * x4
    return result
end

function d_f(input)
    x1, x2, x3, x4 = input
    dx1 = 2 * x1 - 2
    dx2 = 2 * x2
    dx3 = 2 * x3
    dx4 = 2 * x4 - 3 
    return [dx1 dx2 dx3 dx4]
end

function main()
    @time begin
        starting_point = [0 0 0 0]

        constraints = [2 1 1 4;
                       1 1 2 1;
                       -1 0 0 0;
                       0 -1 0 0;
                       0 0 -1 0;
                       0 0 0 -1]

        iteration = 0
        rate = 0.1
        point = starting_point
        loss = typemax(Float32)
        delta = typemax(Float32)
        while mean(abs.(delta)) > 0.0001
            delta = d_f(point)
            loss = f(point)
            point = point - rate .* delta 
            iteration += 1
        end
    end

    println("iterations: ", iteration)
    println("loss: ", loss)
    println("best: ", point)
end

@time main()
@time main()
