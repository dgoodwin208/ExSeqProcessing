
fv = rand(80,3);
vect = rand(3,1);

f = @() fv * vect;
f2 = @() mtimesx(fv, vect);
f3 = @() mmx('mult', fv, vect);

timeit(f)
%timeit(f2,1)
timeit(f3,1)
