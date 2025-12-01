cd data;
dataTable = readtable('tpehgdb.smr', 'Delimiter', '|', 'HeaderLines', 2, ...
    'ReadVariableNames', false, 'FileType', 'delimitedtext');
cd ..;

variableNames = {'Record', 'Gestation', 'RecTime', 'Group', 'Premature', 'Early'};
dataTable.Properties.VariableNames = variableNames;

% Randomly extract one record name of each group for evaluation. The records must be chosed by the 
% group variable, whose possible values are '>=26-PRE', '<26-PRE', '>=26-TERM', '<26-TERM'.
preTermBefore26 = dataTable(strcmp(dataTable.Group, '<26-PRE'), :);
preTermBefore26 = preTermBefore26(randi(height(preTermBefore26)), :);
preTermBefore26 = preTermBefore26.Record{1};
% preTermBefore26 = 'tpehg1526';


preTermAfter26 = dataTable(strcmp(dataTable.Group, '>=26-PRE'), :);
preTermAfter26 = preTermAfter26(randi(height(preTermAfter26)), :);
preTermAfter26 = preTermAfter26.Record{1};
% preTermAfter26 = 'tpehg1720';

termBefore26 = dataTable(strcmp(dataTable.Group, '<26-TERM'), :);
termBefore26 = termBefore26(randi(height(termBefore26)), :);
termBefore26 = termBefore26.Record{1};
% termBefore26 = 'tpehg949';

termAfter26 = dataTable(strcmp(dataTable.Group, '>=26-TERM'), :);
termAfter26 = termAfter26(randi(height(termAfter26)), :);
termAfter26 = termAfter26.Record{1};
% termAfter26 = 'tpehg725';

fprintf('The randomly selected records are:\n');
fprintf('\t<26-PRE: %s\n', preTermBefore26);
fprintf('\t>=26-PRE: %s\n', preTermAfter26);
fprintf('\t<26-TERM: %s\n', termBefore26);
fprintf('\t>=26-TERM: %s\n', termAfter26);

% Run the estimator.m function for each record.
estimator(preTermBefore26);
estimator(preTermAfter26);
estimator(termBefore26);
estimator(termAfter26);
