@testset "solveQP" begin
    Q = @SMatrix [2.0 0.0; 0.0 2.0]

    @testset "unconstrained" begin
        d = SVector(2.0, 4.0)
        A = SMatrix{2,0,Float64}()
        b = SVector{0,Float64}()

        sol = LevelSetShapes.solveQP(Q, d, A, b)

        @test sol == @SVector [1.0, 2.0]
    end

    @testset "inequalities" begin
        d = SVector(2.0, 2.0)
        A = -@SMatrix [1.0 -1.0 0.0 0.0; 0.0 0.0 1.0 -1.0]
        b = SVector(-0.5, -0.5, -0.5, -0.5)

        sol = LevelSetShapes.solveQP(Q, d, A, b)

        @test sol ≈ @SVector [0.5, 0.5]
    end

    @testset "inconsistent inequalities" begin
        d = SVector(0.0, 0.0)
        A = @SMatrix [1.0 -1.0; 0.0 0.0]
        b = SVector(1.0, 1.0)

        @test_throws ErrorException LevelSetShapes.solveQP(Q, d, A, b)
    end

    @testset "type restrictions" begin
        A = -@SMatrix [1.0 -1.0 0.0 0.0; 0.0 0.0 1.0 -1.0]
        b = SVector(-0.5, -0.5, -0.5, -0.5)

        @test_throws MethodError LevelSetShapes.solveQP(Matrix(Q), Vector(SVector(2.0, 2.0)), Matrix(A), Vector(b))
        @test LevelSetShapes.solveQP(Q, SVector(2, 2), A, b) ≈ @SVector [0.5, 0.5]
    end

    @testset "ForwardDiff" begin
        A = -@SMatrix [1.0 -1.0 0.0 0.0; 0.0 0.0 1.0 -1.0]
        b = SVector(-0.5, -0.5, -0.5, -0.5)

        distance_to_box(x) = begin
            sol = LevelSetShapes.solveQP(Q, 2x, A, b)
            return norm(x - sol)
        end

        grad = ForwardDiff.gradient(distance_to_box, SVector(1.0, 1.0))

        @test grad ≈ @SVector [inv(sqrt(2.0)), inv(sqrt(2.0))]

        nested(v) = sum(ForwardDiff.gradient(distance_to_box, SVector(v...)))
        nested_grad = ForwardDiff.gradient(nested, [1.0, 1.0])

        @test all(isfinite, nested_grad)
    end
end
