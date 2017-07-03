function [X_norm, mu, sigma] = featureNormalize(X, mu, sigma)
%FEATURENORMALIZE Normalizes the features in X 
%   FEATURENORMALIZE(X, mu, sigma) returns a normalized version of X where
%   the mean value of each feature is 0 and the standard deviation
%   is 1. This is often a good preprocessing step to do when
%   working with learning algorithms. (precalculated mu and sigma are optional)

% You need to set these values correctly
if ~exist('mu', 'var') || ~exist('sigma', 'var')
	mu = mean(X);
	sigma = std(X);
end

%Alternative way (more efficient)
X_norm = bsxfun(@minus, X, mu);
X_norm = bsxfun(@rdivide, X_norm, sigma);

%Alternative way
% X_norm = (X - mu)./sigma;

X_norm(isnan(X_norm) | isinf(X_norm)) = 0;%Nan values are present because there are no variation hence set them as 0

% ============================================================

end
