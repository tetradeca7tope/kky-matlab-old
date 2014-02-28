% A set of tests for gradientDescent.m

% Test 1: 1D example
f = @(x) x^2 + 2*x + 1;
gradf = @(x) 2*x + 2;
init_pt = -10;
params.num_iters = 10;
[fmin, xmin] = gradientDescent(f, gradf, init_pt, params);
fprintf('f: %s, fmin: %0.4f, \nxmin: %0.4f\n\n', func2str(f), fmin, xmin);

% Test 2: Taken from B&V
f = @(x) 0.5* (x(1)^2 + 10*x(2)^2);
gradf = @(x) [x(1); 10*x(2)];
init_pt = -10*[1; 1];
params.num_iters = 100;
[fmin, xmin] = gradientDescent(f, gradf, init_pt, params);
fprintf('f: %s, fmin: %0.4f, \nxmin: %s\n\n', func2str(f), fmin, mat2str(xmin));

% Test 3: Taken from B&V
f = @(x) exp(x(1) + 3*x(2) - 0.1) + exp(x(1) - 3*x(2) - 0.1) + exp(-x(1) - 0.1);
gradf = @(x) ...
  [exp(x(1) + 3*x(2) - 0.1) + exp(x(1) - 3*x(2) - 0.1) - exp(-x(1) - 0.1); ...
  3*exp(x(1) + 3*x(2) - 0.1) - 3*exp(x(1) - 3*x(2) - 0.1)];
init_pt = 10*rand(2,1);
params.num_iters = 100;
[fmin, xmin] = gradientDescent(f, gradf, init_pt, params);
fprintf('f: %s, fmin: %0.4f, \nxmin: %s\n\n', func2str(f), fmin, mat2str(xmin));

% Test 4: Random 100D Convex Quadratic
U = orth(randn(100));
P = U * diag ( 2 + 3*rand(100, 1) ) * U'; % ensure P is well-conditioned
q = 5*rand(100, 1);
r = 4*rand(1, 1);
f = @(x) 0.5* x'*P*x + q'*x + r;
gradf = @(x) P*x + q;
init_pt = zeros(100, 1);
params.num_iters = 100;
[fmin, xmin] = gradientDescent(f, gradf, init_pt, params);
fprintf('f: %s, fmin: %0.4f, \n', func2str(f), fmin);
fprintf('Analytical Soln: fmin: %0.4f\n\n', -0.5*q'*(P\q) + r);

