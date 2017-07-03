function modelSelection(savedModelsFile, traningDataFile, invertThreshold, varianceRetained)
%% Machine Learning Playground - Model selection Script - selects the best model for a given list of NN models - selects the model for chappie ;)

if ~exist('savedModelsFile', 'var') || isempty(savedModelsFile) || ~exist('traningDataFile', 'var') || isempty(traningDataFile)|| ~exist('invertThreshold', 'var') || isempty(invertThreshold) || ~exist('varianceRetained', 'var') || isempty(varianceRetained)
	%% Initialization
	clear ; close all; clc
	savedModelsFile = 'step13_100-20.96_profitableEntryPoints_savedModels_with_U.mat';
	traningDataFile = 'step13_100-20.96_profitableEntryPoints_traningData.mat';
	invertThreshold = 0;
	varianceRetained = 1.0;
end

%We need to convert the invertThreshold, because it is a logical value and not a numeric
invertThreshold = logical(invertThreshold);

%We need to define if our data will be recuded in terms of its dimentions or not
dimentionalityReduction = varianceRetained < 1.0;

%% Constants definition
TRAINING_DATA_SHARE = 0.6;
CROSS_DATA_SHARE = 0.2;
TEST_DATA_SHARE = 0.2;
TRAIN_DATA_FILE_PATH = 'spread_15_NonOverwritten_complete_training_data.mat';
TEST_DATA_FILE_PATH = 'complete_test_data.mat';
SAVED_MODELS_FLE_PATH = savedModelsFile;

% Load Training Data
fprintf('Loading data set...\n')
load(traningDataFile);% Loading the traning set

if(any(any(isnan(X) | isinf(X))))
  fprintf('There are NaN values in the X input...\n');
  fprintf('Terminating...\n');
  return
end

%[X mu sigma] = featureNormalize(X);%Normalizing the features

%tmp lack of memory
mu = mean(X);%tmp
sigma = std(X);%tmp

if(any(any(isnan(X) | isinf(X))))
  fprintf('There are NaN values in the normalized X input...\n');
  fprintf('Terminating...\n');
  return
end

%Variables definition
m = size(X, 1);
num_labels = max(y);
label_affected_by_threshold = 1;					% the label (class) that will be affected by the thresholds
input_layer_size  = size(X, 2);						% number of features
training_data_count = floor(TRAINING_DATA_SHARE*m);
cross_data_count = floor(CROSS_DATA_SHARE*m);
test_data_count = floor(TEST_DATA_SHARE*m);
savedModels = struct([]);							% The var that will hold the previously trained models and its info
rand_data_indices = randperm(m);					% returns an array with random indexes from 1 to m in order to shuffle our data sets
train_indexes = [];									% the array of the random indexes that belong to the train set
cv_indexes = [];									% the array of the random indexes that belong to the cv set
test_indexes = [];									% the array of the random indexes that belong to the test set
U = [];												% The eigen vectors of our projected data
max_iterations = 100;								% the maximun amount that our optimizer function will iterate to minimize our models cost
num_traning_steps = 10;								% the maximun amount that our optimizer function will iterate to minimize our models cost
continue_training = 1;								% tells us if we should pick up previously calculated nn params and continue to train

%loading previous info
if exist(SAVED_MODELS_FLE_PATH, 'file')
	%Load previously saved trained models in order to reuse past computation and to ease the models comparison
	load(SAVED_MODELS_FLE_PATH, 'savedModels', 'rand_data_indices');% load this info into savedModels vector & random indexes
end

%Selecting our training data
train_indexes = rand_data_indices(1:training_data_count);									% the array of the random indexes that belong to the train set
cv_indexes = rand_data_indices(training_data_count+1:training_data_count+cross_data_count);	% the array of the random indexes that belong to the cv set
test_indexes = rand_data_indices(end-test_data_count:end);	

X=X(train_indexes, :);	% selecting our training data
y=y(train_indexes, :);   % selecting out training data

X = featureNormalize(X, mu, sigma);%Normalizing the features TMP: lack of memory

if dimentionalityReduction
	%It is expected that we reduce our the dimentionality of our data
	%  Run PCA over our training data
	[U, S] = pca(X);
	%updating our input size
	input_layer_size = minDimsRetainVariance(S, varianceRetained);
	% Reduce our data to retain the variance specified
	X = projectData(X, U, input_layer_size);
	
	fprintf('\nThe dimentionality of our data was reduced to %d dimentions and retained %f%% from the original variance... \n', input_layer_size, varianceRetained*100);
end

%% Setup the model parameters
min_hidden_layers = 2;										% The lower limit of hidden layers for our selection
max_hidden_layers = 2;										% The upper limit of hidden layers for our selection
hidden_layer_size = [100];									% The number of neurons in each hidden layer
lambdas = [5.12];
thresholds = [0 0.6 0.7 0.8 0.9 0.925 0.95 0.975];	% the different thresholds that the predominant class will be ignored if the predition H falls below the threshold value

modelsToTrain = repmat( 
					struct( 'InputLayerSize', [],
							'NumberOfHiddenLayers', [], 
							'HiddenLayerSize', [], 
							'OutputLayerSize', [], 
							'Lambda', [], 
							'Threshold', [],
							'VarianceRetained', [],			% This field will inform how variance retained data has its data due to dimentionalityReduction
							'U', [],						% The eigen vectors associated to the dimentionality reduction aplied to this model
							'TraningDataCount', [],			% This field will contain how many m examples were used in the training data to train this model
							'LearnedThetas', [],			% The learned thetas for this model in a unrolled fashion, we can deduct each theta matrix position given the NN size
							'CostList', [],					% The field wich will contain the J cost for each gradient iteration (Note: we can discover how many iterations used with this list)
							'BestCVFScoreTrainStepList', [],% The field which will contain the CVFscore for each training set to detect overfiting points
							'TrainFScore', [],
							'CVFScore', 0,
							'ConfusionMatrix', [],			% The confusion matrix calculated against the CV data set
							'TrainingTimeInSeconds', []), (max_hidden_layers-min_hidden_layers+1)*numel(hidden_layer_size)*numel(lambdas)*numel(thresholds), 1);

i = 1; % The index to iterate over the matrix of models definitions
for num_hidden_layers = min_hidden_layers:max_hidden_layers
	for aHidden_layer_size = hidden_layer_size
		for aLambda = lambdas
			for aThreshold = thresholds
				matchingSavedModelIdx = find([savedModels.InputLayerSize] == input_layer_size
										   & [savedModels.NumberOfHiddenLayers] == num_hidden_layers 
										   & [savedModels.HiddenLayerSize] == aHidden_layer_size
										   & [savedModels.Lambda] == aLambda
										   & [savedModels.Threshold] == aThreshold										   
										   & [savedModels.VarianceRetained] == varianceRetained
										   & arrayfun(@(x) isequal(x.U, U), savedModels)
										   & [savedModels.TraningDataCount] == training_data_count
										   %& arrayfun(@(x) numel(x.CostList) == max_iterations, savedModels) % no longer this matters because now we can futher train models
										   & arrayfun(@(x) ~isempty(x.LearnedThetas), savedModels));% attempting to get the index of previouly matching saved model that matches this NN arquitecture (By matching i mean not only the same NN architecture but as well the same parameters and data)

				if isempty(matchingSavedModelIdx)
					% There is not previouly computed model for these specific model parameters, and hence preparing to train it
					modelsToTrain(i).InputLayerSize = input_layer_size;
					modelsToTrain(i).NumberOfHiddenLayers = num_hidden_layers;
					modelsToTrain(i).HiddenLayerSize = aHidden_layer_size;
					modelsToTrain(i).OutputLayerSize = num_labels;
					modelsToTrain(i).Lambda = aLambda;
					modelsToTrain(i).Threshold = aThreshold;
					modelsToTrain(i).VarianceRetained = varianceRetained;
					modelsToTrain(i).U = U;
					modelsToTrain(i).TraningDataCount = training_data_count;
				else
					%copying over the previously obtain results for this specific model
					modelsToTrain(i) = savedModels(matchingSavedModelIdx(1));% The mathing saved model index must have only one element					
				end
				i = i + 1;
			end
		end
	end
end

fprintf('Initialization complete.\n');
fprintf('\nEvaluating sigmoid gradient...\n')

%Evaluating our sigmoidGradient

g = sigmoidGradient([1 -0.5 0 0.5 1]);
fprintf('Sigmoid gradient evaluated at [1 -0.5 0 0.5 1]:\n  ');
fprintf('%f ', g);
fprintf('\n\n');

%Evaluating our Backpropagation implementation without regularization

fprintf('\nChecking Backpropagation... \n');

%  Check gradients by running checkNNGradients
checkNNGradients;

%Evaluating our Backpropagation implementation with regularization

fprintf('\nChecking Backpropagation (w/ Regularization) ... \n')

%  Check gradients by running checkNNGradients
lambda = 3;
checkNNGradients(lambda);

fprintf('\nStart Training Neural Network Models... \n');

for num_hidden_layers = min_hidden_layers:max_hidden_layers
	for aHidden_layer_size = hidden_layer_size
		for aLambda = lambdas
			% iterates over all the models to train each one of them relevant for the setup parameters, the other that might be previouly saved are ignored
			matchingModelsIndexes = find([modelsToTrain.InputLayerSize] == input_layer_size
							& [modelsToTrain.NumberOfHiddenLayers] == num_hidden_layers 										
							& [modelsToTrain.HiddenLayerSize] == aHidden_layer_size
							& [modelsToTrain.Lambda] == aLambda
							& [modelsToTrain.VarianceRetained] == varianceRetained
							& arrayfun(@(x) isequal(x.U, U), modelsToTrain)');
						
			if  isempty(modelsToTrain(matchingModelsIndexes(1)).LearnedThetas) || continue_training
				% This set of models that match this NN architecture were not yet trained, or requires further training, thus we will train it
				tStart = tic;
				[nn_params, cost, best_cv_fscores] = trainNNModel(X, y, num_labels, modelsToTrain(matchingModelsIndexes(1)).NumberOfHiddenLayers,...
					modelsToTrain(matchingModelsIndexes(1)).HiddenLayerSize, modelsToTrain(matchingModelsIndexes(1)).Lambda,...
					max_iterations, num_traning_steps, modelsToTrain(matchingModelsIndexes(1)).LearnedThetas, thresholds,...
					train_indexes, cv_indexes, traningDataFile, mu, sigma, label_affected_by_threshold,...
					invertThreshold, modelsToTrain(matchingModelsIndexes(1)).U);
				elapsedTimeInSec = toc(tStart);
				
				for i = matchingModelsIndexes
					%saving the execution data to the model struct
					modelsToTrain(i).LearnedThetas = nn_params;
					modelsToTrain(i).CostList = [modelsToTrain(i).CostList ; cost];
					modelsToTrain(i).BestCVFScoreTrainStepList = [modelsToTrain(i).BestCVFScoreTrainStepList ; best_cv_fscores];
					modelsToTrain(i).TrainingTimeInSeconds = elapsedTimeInSec;
				end
				
				fprintf('Elapsed time traing this specific model is %d seconds... \n', elapsedTimeInSec);
				
				for i = matchingModelsIndexes
					%Evaluating this specific model with a specific threshold
					[fscore confusionMatrix] = evaluateModel(modelsToTrain(i).LearnedThetas, modelsToTrain(i).NumberOfHiddenLayers, modelsToTrain(i).HiddenLayerSize,...
						modelsToTrain(i).Lambda, num_labels, X, y, modelsToTrain(i).Threshold, label_affected_by_threshold, invertThreshold, 1);
					modelsToTrain(i).TrainFScore = fscore;% setting the fscore performed against the training set
					
					%Finding the matching index in the saved models array in order to save this data
					matchingSavedModelIdx = find([savedModels.InputLayerSize] == modelsToTrain(i).InputLayerSize
											   & [savedModels.NumberOfHiddenLayers] == modelsToTrain(i).NumberOfHiddenLayers 
											   & [savedModels.HiddenLayerSize] == modelsToTrain(i).HiddenLayerSize
											   & [savedModels.Lambda] == modelsToTrain(i).Lambda
											   & [savedModels.Threshold] == modelsToTrain(i).Threshold
											   & [savedModels.VarianceRetained] == modelsToTrain(i).VarianceRetained
											   & arrayfun(@(x) isequal(x.U, modelsToTrain(i).U), savedModels)
											   & [savedModels.TraningDataCount] == training_data_count);% attempting to get the index of previouly matching saved model that matches this NN arquitecture (By matching i mean not only the same NN architecture but as well the same parameters and data)
											   
					if isempty(matchingSavedModelIdx)
						%this model is not saved yet, thus we need to add it to the structures to save it
						matchingSavedModelIdx = numel(savedModels) + 1;% end + 1
					end					
					
					savedModels(matchingSavedModelIdx) = modelsToTrain(i);% Adding this model computed data to be saved
				end
			
				save(SAVED_MODELS_FLE_PATH,'savedModels','rand_data_indices','mu','sigma','-v6');% overwrite the newly processed models into the saved models array				
			end
		end
	end
end

fprintf('\nStart cross validating the models... \n');
load(traningDataFile);% Loading the data set again to load the cross validation data

X=X(cv_indexes, :);	% selecting our cross validation data
y=y(cv_indexes, :);  % selecting out cross validation data

X = featureNormalize(X, mu, sigma);%Normalizing the features

if dimentionalityReduction
	%It is expected that we reduce our the dimentionality of our data
	% Reduce our data to retain the variance specified
	X = projectData(X, U, input_layer_size);	
end

for num_hidden_layers = min_hidden_layers:max_hidden_layers
	for aHidden_layer_size = hidden_layer_size
		for aLambda = lambdas
			for aThreshold = thresholds
				matchingTrainedModelIdx = find([modelsToTrain.InputLayerSize] == input_layer_size
											& [modelsToTrain.NumberOfHiddenLayers] == num_hidden_layers 										
											& [modelsToTrain.HiddenLayerSize] == aHidden_layer_size
											& [modelsToTrain.Lambda] == aLambda
											& [modelsToTrain.Threshold] == aThreshold
											& [modelsToTrain.VarianceRetained] == varianceRetained
											& arrayfun(@(x) isequal(x.U, U), modelsToTrain)')(1);% There must be one and only one trained model respecting these set of params
				
				if(~isempty(modelsToTrain(matchingTrainedModelIdx).ConfusionMatrix) && continue_training)
					%We retrained this model and thus we will display the previous CM
					disp(modelsToTrain(matchingTrainedModelIdx).ConfusionMatrix);
					fprintf('Above is the previouly trained confusion matrix...\n');
				end
				
				%Evaluating this specific model against the CV data
				[fscore confusionMatrix] = evaluateModel(modelsToTrain(matchingTrainedModelIdx).LearnedThetas, modelsToTrain(matchingTrainedModelIdx).NumberOfHiddenLayers,...
					modelsToTrain(matchingTrainedModelIdx).HiddenLayerSize, modelsToTrain(matchingTrainedModelIdx).Lambda, num_labels, X, y,...
					modelsToTrain(matchingTrainedModelIdx).Threshold, label_affected_by_threshold, invertThreshold, 1);
				
				modelsToTrain(matchingTrainedModelIdx).CVFScore = fscore;% setting the fscore performed against the cv data set
				modelsToTrain(matchingTrainedModelIdx).ConfusionMatrix = confusionMatrix;% setting the confusion matrix against the cv data set
				
				%Finding the matching index in the saved models array in order to save this data
				matchingSavedModelIdx = find([savedModels.InputLayerSize] == input_layer_size
										   & [savedModels.NumberOfHiddenLayers] == num_hidden_layers 
										   & [savedModels.HiddenLayerSize] == aHidden_layer_size
										   & [savedModels.Lambda] == aLambda
										   & [savedModels.Threshold] == aThreshold
										   & [savedModels.VarianceRetained] == modelsToTrain(i).VarianceRetained
										   & arrayfun(@(x) isequal(x.U, modelsToTrain(i).U), savedModels)
										   & [savedModels.TraningDataCount] == training_data_count										   
										   & arrayfun(@(x) ~isempty(x.LearnedThetas), savedModels));% attempting to get the index of previouly matching saved model that matches this NN arquitecture (By matching i mean not only the same NN architecture but as well the same parameters and data)
				
				savedModels(matchingSavedModelIdx) = modelsToTrain(matchingTrainedModelIdx);% Adding this model computed data to be saved
			end
		end
	end
end

save(SAVED_MODELS_FLE_PATH,'savedModels','rand_data_indices','mu','sigma','-v6');% overwrite the newly processed models into the saved models array

fprintf('Selecting the best model against the cross validating data... \n');
[bestCVFscore bestModelIdx] =  max([modelsToTrain.CVFScore]);

fprintf('\nSelected best model against CV data, FScore(%s). Number of hidden layers: (%s), Hidden layer size: (%s), Lambda: (%s), Threshold: (%s), TrainFScore: (%s)...\n',...
	num2str(bestCVFscore), num2str(modelsToTrain(bestModelIdx).NumberOfHiddenLayers), num2str(modelsToTrain(bestModelIdx).HiddenLayerSize), num2str(modelsToTrain(bestModelIdx).Lambda),...
	num2str(modelsToTrain(bestModelIdx).Threshold), num2str(modelsToTrain(bestModelIdx).TrainFScore));

disp(modelsToTrain(bestModelIdx).ConfusionMatrix);
fprintf('Above is the result CV confusion matrix, the rows represent the predicted classes, and the columns the actual classes...\n');
fprintf('CV ModelFscore: (%s)...\n', num2str(modelsToTrain(bestModelIdx).CVFScore));
	
fprintf('\nTest the selected model to see its generalization against the test data... \n');
load(traningDataFile);% Loading the data set again to load the test data

X=X(test_indexes, :);	% selecting our test data
y=y(test_indexes, :); % selecting out test data

X = featureNormalize(X, mu, sigma);%Normalizing the features

if dimentionalityReduction
	%It is expected that we reduce our the dimentionality of our data
	% Reduce our data to retain the variance specified
	X = projectData(X, U, input_layer_size);	
end

%Evaluating the selected model against the test data
[fscore confusionMatrix] = evaluateModel(modelsToTrain(bestModelIdx).LearnedThetas, modelsToTrain(bestModelIdx).NumberOfHiddenLayers,...
	modelsToTrain(bestModelIdx).HiddenLayerSize, modelsToTrain(bestModelIdx).Lambda, num_labels, X, y,...
	modelsToTrain(bestModelIdx).Threshold, label_affected_by_threshold, invertThreshold, 1);
%TMP Commented out
% fprintf('\nTest the selected model with unseen markets to see its generalization against the unseen test data... \n');
% load(TEST_DATA_FILE_PATH);% Loading the data set again to load the test data
% X = featureNormalize(X, mu, sigma);%Normalizing the features

% %Evaluating the selected model against the test data
% [fscore confusionMatrix] = evaluateModel(modelsToTrain(bestModelIdx).LearnedThetas, modelsToTrain(bestModelIdx).NumberOfHiddenLayers,...
	% modelsToTrain(bestModelIdx).HiddenLayerSize, modelsToTrain(bestModelIdx).Lambda, num_labels, X, y,...
	% modelsToTrain(bestModelIdx).Threshold, label_affected_by_threshold, invertThreshold, 1);