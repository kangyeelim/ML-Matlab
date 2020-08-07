function [J, grad] = costFunctionReg(theta, X, y, lambda)
%COSTFUNCTIONREG Compute cost and gradient for logistic regression with regularization
%   J = COSTFUNCTIONREG(theta, X, y, lambda) computes the cost of using
%   theta as the parameter for regularized logistic regression and the
%   gradient of the cost w.r.t. to the parameters. 

% Initialize some useful values
m = length(y); % number of training examples

% You need to return the following variables correctly 
J = 0;
grad = zeros(size(theta));

% ====================== YOUR CODE HERE ======================
% Instructions: Compute the cost of a particular choice of theta.
%               You should set J to the cost.
%               Compute the partial derivatives and set grad to the partial
%               derivatives of the cost w.r.t. each parameter in theta


n = size(X, 2);
z = X * theta;
h_theta = sigmoid(z); % m * n matrix
total = sum(-y' * log(h_theta) - (1 - y')*log(1 - h_theta));
reg = (lambda/ (2 *m)) * sum(theta(2: n) .* theta(2: n)); 
J = (1 / m) * total + reg;

y_mat = repmat(y, 1, n);
grad = (1 / m) * sum(X .* (h_theta - y_mat)); % 1 * n matrix
reg_grad = (lambda / m) * theta(2:n); % (n - 1) * (n - 1) matrix
grad(:, 2:n) = grad(:, 2:n) + reg_grad(:, 1)';


% =============================================================

end
