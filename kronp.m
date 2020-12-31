% Takes the Kronecker power of a matrix
function B = kronp(A, k)
    B = 1;
    for i = 1:k
        B = kron(B, A);
    end
end