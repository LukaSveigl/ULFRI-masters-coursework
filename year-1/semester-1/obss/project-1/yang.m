function []=yang(recordName)
    [fullPath] = init(recordName);

    % Get the ECG and BP indices in the signal.
    fprintf('Extracting ECG and BP indices:\n');
    [ecgIndex, bpIndex] = getIndices(recordName);
    fprintf('\tECG index: %d\n', ecgIndex);
    fprintf('\tBP index: %d\n', bpIndex);

    % Calculate the QRS and BP complexes.
    fprintf('Performing QRS detection.\n');
    gqrs(recordName, [], [], ecgIndex);
    fprintf('Performing BP detection.\n');
    wabp(recordName, [], [], [], bpIndex);

    evaluatePeaks(recordName, fullPath);

    cleanup();
end

% Evaluates the QRS and BP complexes and corrects/predicts the correct positions of 
% the QRS complexes.
function evaluatePeaks(recordName, fullPath)
    qrsPoints = rdann(recordName, 'qrs');
    bpPoints = rdann(recordName, 'wabp');

    slidingWindowSize = 10;
    slidingWindow = [];
    correctedSignal = [];

    for i = 2 : length(bpPoints)
        qrsBetweenBp = qrsPoints(qrsPoints > bpPoints(i - 1) & qrsPoints < bpPoints(i));

        if (length(qrsBetweenBp) == 1)
            % Case 1: There is exactly one QRS complex between two BP complexes. This case is part
            % of the 'clean' part of the signal, so we update the moving average.

            % Make sure there are always at most `slidingWindowSize` elements in the sliding window.
            if (length(slidingWindow) < slidingWindowSize)
                slidingWindow(end + 1) = qrsBetweenBp(1);
            else
                slidingWindow = slidingWindow(2 : end);
                slidingWindow(end + 1) = qrsBetweenBp(1);
            end

            % If the sliding window is empty, skip this iteration as that means we are at the
            % beginning of the signal. If the sliding window is not empty, check if the current
            % QRS complex is away from the next BP signal for at most the moving average. If it is add
            % the QRS complex to the corrected signal. If it is not, subtract the moving average from
            % the next BP complex and add the result to the corrected signal.
            if (~isempty(slidingWindow))
                if (abs(bpPoints(i) - qrsBetweenBp(1)) <= mean(slidingWindow))
                    correctedSignal(end + 1) = qrsBetweenBp(1);
                else
                    correctedSignal(end + 1) = bpPoints(i) - mean(slidingWindow);
                end
            end
        elseif (length(qrsBetweenBp) > 1)
            % Case 2: There are more than one QRS complex between two BP complexes. This case is part of the 'noisy' part of the signal,
            % for which the authors of the paper propose a solution using a "simple horizontal line check". The approach, however, is 
            % not very clear. Instead, we propose the use of a temporal analysis approach. The assumption is that the QRS complexes 
            % should be evenly spaced, while allowing for some small deviation. The approach is as follows:
            % 1. Calculate the expected interval between QRS complexes from the sliding window.
            % 2. Calculate the deviation between the expected interval and the actual interval between QRS complexes.
            % 3. If the deviation is smaller than a threshold, the QRS complexes are considered to be true QRS complexes. Otherwise, they are
            %    considered to be noise.
            
            % Calculate the expected interval between QRS complexes.
            expectedInterval = mean(diff(slidingWindow));
            threshold = 0.2;

            % Identify true QRS complexes.
            deviations = abs(diff(qrsBetweenBp) - expectedInterval);
            trueDetections = deviations <= threshold;

            % Extract the true QRS complexes using logical indexing
            qrsBetweenBp = qrsBetweenBp(trueDetections);

            for j = 1 : length(qrsBetweenBp)
                correctedSignal(end + 1) = qrsBetweenBp(j);
            end

            % Predict the QRS complex based on the moving average. This is used in the `basic` implementation.
            %predictedQrs = bpPoints(i) - mean(slidingWindow);
            %correctedSignal(end + 1) = predictedQrs;
        else
            % Case 3: There is no QRS complex between two BP complexes. If not at the beginning of
            % the signal, subtract the moving average from the next BP complex and add the result to
            % the corrected signa.
            if (~isempty(slidingWindow))
                correctedSignal(end + 1) = bpPoints(i) - mean(slidingWindow);
            end
        end
    end

    % Write the corrected signal to a file.
    asciName = sprintf('%s.asc', recordName);
    fid = fopen(asciName, 'wt');
    for i = 1 : size(correctedSignal, 2)
        fprintf(fid,'0:00:00.00 %d N 0 0 0\n', correctedSignal(1, i));
    end
    fclose(fid);
end

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Helper functions
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%


% Initializes the script by moving to the data directory and returning the full path to the
% record.
function [fullPath]=init(recordName)
    fprintf('Running yang.m with arguments:\n');
    fprintf('\trecordName: %s\n', recordName);

    fprintf('Moving to data directory.\n');
    cd data/set-p/;
    fullPath = fullfile(pwd, recordName);
end

% Cleans up the script by moving back to the src directory.
function cleanup()
    fprintf('Moving back to src directory.\n');
    cd ../..;
end

% Returns the indices of the ECG and BP signals in the given record.
function [ecgIndex, bpIndex]=getIndices(recordName)
    siginfo = wfdbdesc(recordName);
    description = squeeze(struct2cell(siginfo));
    % Extract the signal name from the description.
    description = description(9, :);
    ecgIndex = getIndex(description, 'ECG');
    bpIndex = getIndex(description, 'BP');
end

% Returns the index of the given pattern in the given signal information.
% Taken from the sample entry of the CinC Challenge 2014.
function ind=getIndex(siginfo, pattern)
    ind = [];
    tmp_ind = strfind(siginfo, pattern);
    for m = 1 : length(siginfo)
        if (~isempty(tmp_ind{m}))
            ind(end + 1) = m;
        end
    end
end