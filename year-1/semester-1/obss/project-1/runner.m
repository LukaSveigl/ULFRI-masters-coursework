% This matlab script is used to run the yang.m detector on all of the records in the
% ./data/set-p/ directory.


for i = 100:200
    recordName = sprintf('%d', i);

    % Check if the record exists
    if ~exist(sprintf('./data/set-p/%s.dat', recordName), 'file')
        continue;
    end

    % Run the detector
    yang(recordName);

    % For each record, the detector outputs .qrs, .wabp and .asc files. The .asc files need 
    % to be converted into .qrs files via wrann, and evaluated via bxb comparing them to the
    % respective .atr files.

    cd ./data/set-p/;

    % Convert the .asc files to .qrs files
    command = sprintf('wrann -r %s -a qrs < %s.asc', recordName, recordName);
    system(command);

    % Evaluate the .qrs files
    system(sprintf('bxb -r %s -a atr qrs -f 0 -l eval1.txt eval2.txt', recordName));

    cd ../../;
end

cd ./data/set-p/;

% Summarize the results
system('sumstats eval1.txt eval2.txt > results.txt');

cd ../../;