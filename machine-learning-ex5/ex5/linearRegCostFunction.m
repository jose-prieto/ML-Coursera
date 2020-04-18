function [J, grad] = linearRegCostFunction(X, y, theta, lambda)
%LINEARREGCOSTFUNCTION Compute cost and gradient for regularized linear 
%regression with multiple variables
%   [J, grad] = LINEARREGCOSTFUNCTION(X, y, theta, lambda) computes the 
%   cost of using theta as the parameter for linear regression to fit the 
%   data points in X and y. Returns the cost in J and the gradient in grad

% Initialize some useful values
m = length(y); % number of training examples

% You need to return the following variables correctly 
J = 0;
grad = zeros(size(theta));

% ====================== YOUR CODE HERE ======================
% Instructions: Compute the cost and gradient of regularized linear 
%               regression for a particular choice of theta.
%
%               You should set J to the cost and grad to the gradient.
%

%lamba for reg
    n = size(X,2);
    thetareg = theta(2:n);

%hypothesis
    h = X * theta;

%regularized cost function
    cost = sum((h - y) .^ 2) / (2 * m);
    reg = lambda * sum(thetareg .^ 2) / (2 * m);
    J = cost + reg;

%regularized gradient
    grad = X' * (h - y) / m;
    grad(2:n) += lambda * theta(2:n) / m;

% =========================================================================

grad = grad(:);

end
