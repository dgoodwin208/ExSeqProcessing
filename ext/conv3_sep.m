function Y = conv3_sep(A, b)

n = length(b);
a = b;
Y = convn(b, A);
b = a';
Y = convn(b, Y);
b = reshape(a, [1 1 n]);
Y = convn(b, Y);
%b = reshape(a, [1 1 1 n]);
%Y = convn(b, Y);

end
