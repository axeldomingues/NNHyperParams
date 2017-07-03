function [nn_params, cost, best_cv_fscores] = trainNNModel(X, y, num_labels, num_hidden_layers, hidden_layer_size, lambda, maxIterations, numTraningSteps, initial_nn_params, thresholds, train_indexes, cv_indexes, file, mu, sigma, label_affected_by_threshold, invertThreshold, U)%Because of lack of mem we load the data several times (TMP)
	% Some useful variables
	m = size(X, 1);
	input_layer_size = size(X, 2);
	cost = [];
	aux_cost = [];
	best_cv_fscores = [];

	% Add ones to the X data matrix
	X = [ones(m, 1) X];

	% extrapolate the values into Y binary vectors, and thus the result is the binary matrix Y (classes)
	I = eye(num_labels);
	Y = zeros(m,num_labels);
	for i=1:m
	  Y(i,:) = I(y(i),:);
	end

	fprintf('\nStart training NN model. Number of hidden layers: (%s), Hidden layer size: (%s), Lambda: (%s)...\n', num2str(num_hidden_layers), num2str(hidden_layer_size),...
		num2str(lambda));
		
	if(isempty(initial_nn_params))
		%the initial neural network parameters are empty thus it means that a training from scratch is required
		initial_nn_params = [randInitializeWeights(input_layer_size, hidden_layer_size)(:)];% first theta (input theta)

		for i=2:num_hidden_layers
			% iterates through the second hidden layer until the last hidden layer
			initial_nn_params = [initial_nn_params ; randInitializeWeights(hidden_layer_size, hidden_layer_size)(:)];
		end

		initial_nn_params = [initial_nn_params ; randInitializeWeights(hidden_layer_size, num_labels)(:)];% last theta (output theta)
	end

	% Set options for fminunc
	options = optimset('GradObj', 'on', 'MaxIter', maxIterations);

	% Create "short hand" for the cost function to be minimized
	costFunction = @(p) nnCostFunction(p, ...
									   input_layer_size, ...
									   num_hidden_layers, ...
									   hidden_layer_size, ...
									   num_labels, X, Y, lambda);
	

	% Now, costFunction is a function that takes in only one argument (the
	% neural network parameters)
	for i=1:numTraningSteps
		current_best_cv_fscore = -1;
		[nn_params, aux_cost] = fmincg(costFunction, initial_nn_params, options);
		initial_nn_params = nn_params;
		cost = [cost ; aux_cost];
		
		%loading cv data
		clear X; clear y;
		load(file);

		X=X(cv_indexes, :);	% selecting our cross validation data
		y=y(cv_indexes, :);  % selecting out cross validation data

		X = featureNormalize(X, mu, sigma);%Normalizing the features
		
		if (~isempty(U))		
			% Reduce our data to input_layer_size dims
			X = projectData(X, U, input_layer_size);
		end
		
		for aThreshold = thresholds
			%Evaluating this specific model against the CV data
			print = aThreshold == 0;%Print the evaluation at each step only for the threshold == 0
			
			[fscore confusionMatrix] = evaluateModel(nn_params, num_hidden_layers,...
				hidden_layer_size, lambda, num_labels, X, y,...
				aThreshold, label_affected_by_threshold, invertThreshold, print);
			
			if(fscore > current_best_cv_fscore)
				current_best_cv_fscore = fscore;
			end
		end
		
		best_cv_fscores = [best_cv_fscores ; current_best_cv_fscore];
		
		%reloading train data
		clear X; clear y;
		load(file);

		X=X(train_indexes, :);	% selecting our cross validation data

		X = featureNormalize(X, mu, sigma);%Normalizing the features
		
		if (~isempty(U))		
			% Reduce our data to input_layer_size dims
			X = projectData(X, U, input_layer_size);
		end
	end
	
	disp(best_cv_fscores);%Aux display to see the generalization optimization
end
