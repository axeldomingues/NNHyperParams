function ThetasCell = reshapeNNWeitghtsToThetasCell(nn_params, input_layer_size, num_hidden_layers, hidden_layer_size, num_labels)
  %reshapeNNWeitghtsToThetasCell reshapes a list of nn params (weights) 
  %   to a list of Theta matrixes according to the NN architecture.
  %   ThetasCell = reshapeNNWeitghtsToThetasCell(nn_params, input_layer_size, num_hidden_layers, hidden_layer_size, num_labels)
  %   outputs the a list (cells) of Theta matrixes according to the NN architecture which is specified by the input.

	% Useful values
	previousLayerFinalIdx = hidden_layer_size * (input_layer_size + 1);
	ThetasCell = {reshape(nn_params(1:previousLayerFinalIdx), hidden_layer_size, (input_layer_size + 1))};% adding the first Theta matrix to the cell array

	for i=2:num_hidden_layers
		% iterates through the second hidden layer until the last hidden layer
		thisLayerFinalIdx = previousLayerFinalIdx+(hidden_layer_size * (hidden_layer_size + 1));
		ThetasCell(end+1) = reshape(nn_params(previousLayerFinalIdx+1:thisLayerFinalIdx), ...
												   hidden_layer_size, (hidden_layer_size + 1));%Adding this layer theta to the array cell of thetas
		previousLayerFinalIdx = thisLayerFinalIdx;
	end

	ThetasCell(end+1) = reshape(nn_params(previousLayerFinalIdx+1:end), ...
											   num_labels, (hidden_layer_size + 1));%Adding the last layer theta to the array cell of thetas

  % =========================================================================
end