function [J grad] = nnCostFunction(nn_params, ...
                                   input_layer_size, ...
								   num_hidden_layers, ...
                                   hidden_layer_size, ...
                                   num_labels, ...
                                   X, Y, lambda)
	%NNCOSTFUNCTION Implements the neural network cost function for a specified number of layer
	%neural network which performs classification
	%   [J grad] = NNCOSTFUNCTON(nn_params, input_layer_size, num_hidden_layers, hidden_layer_size, num_labels, ...
	%   X, y, lambda) computes the cost and gradient of the neural network. The
	%   parameters for the neural network are "unrolled" into the vector
	%   nn_params and need to be converted back into the weight matrices. 
	% 
	%   The returned parameter grad should be a "unrolled" vector of the
	%   partial derivatives of the neural network.
	%

	% Reshape nn_params back into the parameters Theta1, Theta2, Theta3,..., ThetaN in a array cell of thetas the weight matrices
	% for our N layer neural network

	% You need to return the following variables correctly 
	J = 0;
	grad = [];

	% Setup some useful variables
	regularization_term = 0;
	m = size(X, 1);
	ThetasCell = reshapeNNWeitghtsToThetasCell(nn_params, input_layer_size, num_hidden_layers, hidden_layer_size, num_labels);
	AsCell = {};
	ZsCell = {};
	SigmasCell = {};
	DeltasCell = {};
									   
	ZsCell(1) = X*ThetasCell{1}';
	AsCell(1) = [ones(m,1),sigmoid(ZsCell{1})];

	for i=2:num_hidden_layers
		ZsCell(end+1) = AsCell{end}*ThetasCell{i}';
		AsCell(end+1) = [ones(m,1),sigmoid(ZsCell{end})];
	end

	h = sigmoid(AsCell{end}*ThetasCell{end}');

	for i=1:numel(ThetasCell)
		regularization_term += sum(sum(ThetasCell{i}(:,2:end).^2));
	end

	regularization_term *= lambda/(2*m);

	J = -sum(sum(sum(log(h(Y==1))) + sum(log(1-h(Y==0))),2))/m;%Cost function safe from 0 * log(0) produces a NaN because of 0 * Inf
	J += regularization_term;

	SigmasCell(end+1) = h - Y;

	for i=numel(ThetasCell):-1:2
		SigmasCell = [{(SigmasCell{1}*ThetasCell{i}.*sigmoidGradient([ones(size(ZsCell{i-1}, 1), 1) ZsCell{i-1}]))(:, 2:end)} SigmasCell];
	end
	
	DeltasCell(1) = SigmasCell{1}'*X;%Note: First A matrix is X

	for i=2:numel(SigmasCell)
		DeltasCell(i) = SigmasCell{i}'*AsCell{i-1};
	end

	for i=1:numel(ThetasCell)
		grad = [grad ; (DeltasCell{i}./m + (lambda/m)*[zeros(size(ThetasCell{i},1), 1) ThetasCell{i}(:, 2:end)])(:)];
	end
end
