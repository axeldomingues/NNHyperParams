function [modelFscore confusionMatrix] = evaluateModel(nn_params, num_hidden_layers, hidden_layer_size, lambda, num_labels, X, y, threshold, classThreshold, invertThreshold, print)
  %EVALUATEMODEL evaluates a given NN model against a given data set X and y
  %   p = EVALUATEMODEL(nn_params, num_hidden_layers, hidden_layer_size, lambda, num_labels, X, y, threshold, classThreshold)
  %   outputs evaluation metric of this NN model fscore & the confusion matrix
  
  if ~exist('print', 'var')
	print = 0;
  end
  
  if print
	fprintf('Start evaluation of NN model. Number of hidden layers: (%s), Hidden layer size: (%s), Lambda: (%s), Threshold: (%s)...\n', num2str(num_hidden_layers),...
		num2str(hidden_layer_size), num2str(lambda), num2str(threshold));
  end

  %declaration of useful vars
  p = predict(nn_params, num_hidden_layers, hidden_layer_size, num_labels, X, threshold, classThreshold, invertThreshold);
  confusionMatrix = confusionmat(p,y);
  modelFscore = 0;
  classesFscores = [];
  
  warning('off','Octave:divide-by-zero');% Here is often the case that division by-zero happens
  
  for i=1:num_labels
    % Here we will calculate the fscore for each class so that we can avg all the fscores to evaluate our model
    truePositives = confusionMatrix(i,i);
    falsePositives = sum(confusionMatrix(i,:)) - truePositives;
    falseNegatives = sum(confusionMatrix(:,i)) - truePositives;
	
	precision = truePositives/(truePositives+falsePositives);
    recall = truePositives/(truePositives+falseNegatives);
	thisClassFscore = 2*((precision*recall)/(precision+recall));
	
	if print
		fprintf('Class: (%s), Precision: (%s), Recall: (%s), ClassFscore: (%s)...\n', num2str(i), num2str(precision), num2str(recall), num2str(thisClassFscore));
	end
    
    classesFscores(end+1) = thisClassFscore;
  end
  
  modelFscore = num_labels / sum(1./classesFscores);% Fscoring all the fscores
  
  warning('on','Octave:divide-by-zero');% Here is often the case that division by-zero happens
  
  if print
	disp(confusionMatrix);
	fprintf('Above is the result confusion matrix, the rows represent the predicted classes, and the columns the actual classes...\n');
	fprintf('ModelFscore: (%s)...\n', num2str(modelFscore));
	%pause (5);% Pauses the script for 5 secs before the progress continues (helps human reading of the output)
	fflush(stdout);% This forces the flush the buffer from disp
	fflush(stderr);
  end

  % =========================================================================
end