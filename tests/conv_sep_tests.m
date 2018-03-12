%normalized gaussian kernel
a=single([.0003 .1065 .7866 .1065 .0003]); 
n=length(a);

%make a 3D kernel
H3 = fspecial3('gaussian');
a2 = H3(3, :, 3);

H4 = fspecial('gaussian',5);
[s,v,d] = svd(H4);
a3 = abs(s(:,1));

b=a;
b2 = a2;
b3 = a3;
for d=2:3
    b=kron(b,a');
    b2=kron(b2,a2');
    b3=kron(b3,a3');
end
H=single(reshape(b,n*ones(1,3)));
H2=reshape(b2,n*ones(1,3));
H4=reshape(b3,n*ones(1,3));
%norm(H3(:) - H2(:))
%norm(H2(:))

%make a test matrix
A = single(randn(2048,2048,141));

%%compare execution times
options = {};
options.Power2Flag = false;
tic
B = convnfft(A,H, 'same', [], options);
toc

chunks = [500, 750, 1000];
for chunk=chunks
    tic
    B2 = convnsep({a,a,a}, A, 'same', chunk);
    toc
end

%compare error to size of the result of convn
norm(B(:)-B2(:))/  norm(B(:))
