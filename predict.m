function p = predict(nn_params, num_hidden_layers, hidden_layer_size, num_labels, X, threshold, indexThreshold, invertThreshold)
  %PREDICT Predict the label of an input given a trained neural network
  %   p = PREDICT(nn_params, num_hidden_layers, hidden_layer_size, num_labels, X, threshold, indexThreshold)
  %   outputs the predicted label of X given the
  %   trained weights of a neural network (nn_params)

  % Useful values
  m = size(X, 1);
  previousComputedInput = X;
  input_layer_size = size(X, 2);
  ThetasCell = reshapeNNWeitghtsToThetasCell(nn_params, input_layer_size, num_hidden_layers, hidden_layer_size, num_labels);

  % You need to return the following variables correctly 
  p = zeros(size(X, 1), 1);
  
  for i=1:numel(ThetasCell)
    previousComputedInput = sigmoid([ones(m, 1) previousComputedInput] * ThetasCell{i}');
  end
  
  [h, p] = max(previousComputedInput, [], 2);
  
  if threshold > 0
    % the threshold will be considered to ignore the relevant indexThreshold that fall below the threshold
    isBelowThreshold = @(value) logical(value<threshold);% created an anonymous function to evalute if a value is below the specified threshold
	
	if ~invertThreshold
		logicalRelevantIndexes = p == indexThreshold;% Creating a logical indexing array (1 0 0 0 1 0..) of the relevant preditions
	else
		logicalRelevantIndexes = p != indexThreshold;% Creating a logical indexing array (1 0 0 0 1 0..) of the relevant preditions
	end
	
    compoundCondInd = logicalRelevantIndexes & isBelowThreshold(h);% this combines the two logical indexations from p with h. (Ps that are relevant ie that are relevant class, & Hs of the relevant class that falls below the specified threshold)
    
    previousComputedInput(compoundCondInd & bsxfun(@eq, previousComputedInput, h)) = -Inf; %replace the relevant max(s) by -Inf (it replaces for each relevant row the max h with minimum value -Inf)
    
    [h, p] = max(previousComputedInput, [], 2); % Recalculating the preditions ignoring the relevant classes that fell by the threshold
  end

  % =========================================================================
end