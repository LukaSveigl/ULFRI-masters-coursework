function []=estimator(recordName)
    init(recordName);

    % Extract the signals from the record.
    [signals, Fss, tms] = extractSignals(recordName);

    % Preprocess the signals.
    [signals, Fss, tms] = preprocessSignals(signals, Fss, tms);

    % Compute the spectrograms.
    [firstS, secondS, thirdS] = computeSpectrograms(signals, Fss, tms);

    % Compute the estimators.
    [medianEstimators, peakEstimators] = computeEstimators(firstS, secondS, thirdS, recordName);

    cleanup();
end


% Extracts the first, second and third channel, filtered using a 4-pole band-pass Butterworth
% filter with cutoff frequencies of 0.3 and 4 Hz, from the given record. The structure of the
% record is described on the following page: https://physionet.org/content/tpehgdb/1.0.1/
function [signals, Fss, tms]=extractSignals(recordName)
    signals = [];
    Fss = [];
    tms = [];

    [signal, Fs, tm] = rdsamp(recordName, 4);
    signals = [signals, signal];
    Fss = [Fss, Fs];
    tms = [tms, tm];

    [signal, Fs, tm] = rdsamp(recordName, 8);
    signals = [signals, signal];
    Fss = [Fss, Fs];
    tms = [tms, tm];

    [signal, Fs, tm] = rdsamp(recordName, 12);
    signals = [signals, signal];
    Fss = [Fss, Fs];
    tms = [tms, tm];
end


% Extracts the metadata for the given record from the tpehgdb.smr file. The structure of the
% file is described on the following page: https://physionet.org/content/tpehgdb/1.0.1/
function [gestation, recTime, group, premature, early]=extractMetadata(recordName)
    cd ..;
    dataTable = readtable('tpehgdb.smr', 'Delimiter', '|', 'HeaderLines', 2, ...
        'ReadVariableNames', false, 'FileType', 'delimitedtext');
    cd tpehgdb;

    variableNames = {'Record', 'Gestation', 'RecTime', 'Group', 'Premature', 'Early'};
    dataTable.Properties.VariableNames = variableNames;

    % Find the row corresponding to the given record.
    recordIndex = strcmp(dataTable.Record, recordName);

    recordData = dataTable(recordIndex, :);
    gestation = recordData.Gestation;
    recTime = recordData.RecTime;
    group = recordData.Group{1};
    premature = recordData.Premature{1};
    early = recordData.Early{1};
end


% As per the instructions on the page https://physionet.org/content/tpehgdb/1.0.1/, the
% first and last 180 seconds of each record are removed since these intervals contain
% transient effects from filters.
function [signals, Fss, tms]=preprocessSignals(oldSignals, oldFss, oldTms)
    fprintf('Preprocessing signals.\n');

    signals = [];
    Fss = [];
    tms = [];

    for sigIndex = 1 : size(oldSignals, 2)
        signal = oldSignals(:, sigIndex);
        Fs = oldFss(sigIndex);
        tm = oldTms(:, sigIndex);

        % Remove the first and last 180 seconds.
        signal = signal(181 * int32(Fs) : end - 180 * int32(Fs));
        tm = tm(181 * int32(Fs) : end - 180 * int32(Fs));

        signals = [signals, signal];
        Fss = [Fss, Fs];
        tms = [tms, tm];
    end
end


% Computes the spectrograms using the short-time Fourier transform for the given signals and displays them.
% The result of the function is a list of 3 spectrograms, one for each channel.
function [firstS, secondS, thirdS]=computeSpectrograms(signals, Fss, tms)    
    % segmentLength = 256;
    % overlap = 128;
    % window = hamming(segmentLength);

    segmentLength = 256;
    overlap = 255;
    window = hamming(segmentLength);

    % segmentLength = 512;
    % overlap = 256;
    % window = hamming(segmentLength);

    % segmentLength = 256;
    % overlap = 128;
    % window = hann(segmentLength);

    % segmentLength = 256;
    % overlap = 128;
    % window = blackman(segmentLength);

    % segmentLength = 256;
    % overlap = 255;
    % window = blackman(segmentLength);

    fprintf('Computing spectrograms with arguments:\n');
    fprintf('\tsegmentLength: %d\n', segmentLength);
    fprintf('\toverlap: %d\n', overlap);
    fprintf('\twindow: hamming\n');

    [spec1, freq1, time1] = spectrogram(signals(:, 1), window, overlap, segmentLength, Fss(1), 'yaxis');
    [spec2, freq2, time2] = spectrogram(signals(:, 2), window, overlap, segmentLength, Fss(2), 'yaxis');
    [spec3, freq3, time3] = spectrogram(signals(:, 3), window, overlap, segmentLength, Fss(3), 'yaxis');

    firstS = cell(1, 3);
    secondS = cell(1, 3);
    thirdS = cell(1, 3);

    firstS{1} = spec1;
    firstS{2} = freq1;
    firstS{3} = time1;
    firstS{4} = segmentLength;
    firstS{5} = Fss(1);

    secondS{1} = spec2;
    secondS{2} = freq2;
    secondS{3} = time2;
    secondS{4} = segmentLength;
    secondS{5} = Fss(2);

    thirdS{1} = spec3;
    thirdS{2} = freq3;
    thirdS{3} = time3;
    thirdS{4} = segmentLength;
    thirdS{5} = Fss(3);
end


% Computes the estimators for all three signals of the given record and plots them on the
% corresponding spectrograms. The estimators are the median frequency and the peak frequency.
% The result of the function is a list of 3 median estimators and a list of 3 peak estimators,
% one for each channel. Additionally, the function exports the spectrograms with the estimators
% plotted on them.
function [medianEstimators, peakEstimators]=computeEstimators(firstS, secondS, thirdS, recordName)
    fprintf('Computing estimators.\n');

    medianEstimators = cell(1, 3);
    peakEstimators = cell(1, 3);

    % First spectrogram.
    fprintf('Computing estimators for signal 1.\n')

    spec1 = firstS{1};
    freq1 = firstS{2};
    time1 = firstS{3};
    segl1 = firstS{4};
    Fs1 = firstS{5};

    [peakIndices, peakFrequencies] = estimatePeaks(spec1, freq1, time1, segl1, Fs1);
    [medianIndices, medianFrequencies] = estimateMedian(spec1, freq1, time1, segl1, Fs1);

    medianEstimators{1} = medianFrequencies;
    peakEstimators{1} = peakFrequencies;

    lineWidth = 1.5;

    % Plot the time course of each estimator along the spectrogram.
    figure;
    title(recordName);
    set(gcf, 'Position',  [100, 100, 1280, 800])

    subplot(3, 1, 1);
    imagesc(time1, freq1, 10 * log10(abs(spec1)));
    title('Spectrogram - Signal 1');
    xlabel('Time (s)');
    ylabel('Frequency (Hz)');
    colorbar;

    ylim([0 5]);

    hold on;
    plot(time1, medianFrequencies, 'r', 'LineWidth', lineWidth);
    plot(time1, peakFrequencies, 'b', 'LineWidth', lineWidth);
    legend('Median Frequency', 'Peak Frequency');
    hold off;

    % Second spectrogram.
    fprintf('Computing estimators for signal 2.\n')

    spec2 = secondS{1};
    freq2 = secondS{2};
    time2 = secondS{3};
    segl2 = secondS{4};
    Fs2 = secondS{5};

    [peakIndices, peakFrequencies] = estimatePeaks(spec2, freq2, time2, segl2, Fs2);
    [medianIndices, medianFrequencies] = estimateMedian(spec2, freq2, time2, segl2, Fs2);

    medianEstimators{2} = medianFrequencies;
    peakEstimators{2} = peakFrequencies;

    subplot(3, 1, 2);
    imagesc(time2, freq2, 10 * log10(abs(spec2)));
    title('Spectrogram - Signal 2');
    xlabel('Time (s)');
    ylabel('Frequency (Hz)');
    colorbar;

    ylim([0 5]);

    hold on;
    plot(time2, medianFrequencies, 'r', 'LineWidth', lineWidth);
    plot(time2, peakFrequencies, 'b', 'LineWidth', lineWidth);
    legend('Median Frequency', 'Peak Frequency');
    hold off;

    % Third spectrogram.
    fprintf('Computing estimators for signal 3.\n')

    spec3 = thirdS{1};
    freq3 = thirdS{2};
    time3 = thirdS{3};
    segl3 = thirdS{4};
    Fs3 = thirdS{5};

    [peakIndices, peakFrequencies] = estimatePeaks(spec3, freq3, time3, segl3, Fs3);
    [medianIndices, medianFrequencies] = estimateMedian(spec3, freq3, time3, segl3, Fs3);

    medianEstimators{3} = medianFrequencies;
    peakEstimators{3} = peakFrequencies;

    subplot(3, 1, 3);
    imagesc(time3, freq3, 10 * log10(abs(spec3)));
    title('Spectrogram - Signal 3');
    xlabel('Time (s)');
    ylabel('Frequency (Hz)');
    colorbar;

    ylim([0 5]);

    hold on;
    plot(time3, medianFrequencies, 'r', 'LineWidth', lineWidth);
    plot(time3, peakFrequencies, 'b', 'LineWidth', lineWidth);
    legend('Median Frequency', 'Peak Frequency');
    hold off;

    % Export the figures.
    fprintf('Exporting figures.\n');
    figureName = strcat(recordName, '.png');
    saveas(gcf, figureName);
end


% Estimates the median frequency using the following formula: f_med = i_m * F_s / N, where
% sum_{i=i_low}^{i=i_m} P[i] approximately equals sum_{i=i_m + 1}^{i=i_high} P[i]. The median 
% frequency is thus defined as the frequency where the sums of the parts above and below in the 
% frequency power spectrum, P[i], are approximately equal.
function [medianIndices, medianFrequencies]=estimateMedian(spec, freq, time, segmentLength, Fs)
    medianIndices = [];
    medianFrequencies = [];
    
    for i = 1:length(time)
        cumSum = cumsum(abs(spec(:, i)));
        totalSum = cumSum(end);

        % Find the index of the frequency with the closest cumulative sum to halfSum.
        [~, idx] = min(abs(cumSum - 0.5 * totalSum));

        fMed = idx * Fs / segmentLength;
        
        medianFrequencies = [medianFrequencies, fMed];
    end

end


% Estimates the peak frequency using the following formula: f_max = F_s / N * arg(max_{i=i_low}^{i=i_high} P[i]}).
function [peakIndices, peakFrequencies]=estimatePeaks(spec, freq, time, segmentLength, Fs)
    peakIndices = [];
    peakFrequencies = [];

    % Loop through the spectrogram and compute the i_min and i_max indices for each time segment.
    % for i = 1 : segmentLength : size(spec, 2)
    %     i_low = i;
    %     i_high = i + segmentLength;
% 
    %     if (i_high > size(spec, 2))
    %         i_high = size(spec, 2);
    %     end
% 
    %     % Find the index with the maximum amplitude in the current segment.
    %     [~, maxIdx] = max(abs(spec(:, i_low : i_high)), [], 1);
% 
    %     fMax = freq(maxIdx);
% 
    %     peakIndices = [peakIndices, maxIdx];
    %     peakFrequencies = [peakFrequencies, fMax];
    % end

    for i = 1:length(time)
        [~, maxIdx] = max(abs(spec(:, i)), [], 1);

        fMax = freq(maxIdx);

        peakIndices = [peakIndices, maxIdx];
        peakFrequencies = [peakFrequencies, fMax];
    end
end


%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Helper functions
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%


% Initializes the script by moving to the data directory and returning the full path to the
% record.
function [fullPath]=init(recordName)
    fprintf('Running estimator.m with arguments:\n');
    fprintf('\trecordName: %s\n',recordName);

    fprintf('Moving to data directory.\n');
    cd data/tpehgdb;
    fullPath = fullfile(pwd, recordName);

    % Verify that the record exists.
    if (~exist(strcat(fullPath, '.dat'), 'file'))
        fprintf('Record %s does not exist.\n', recordName);
        cleanup();
        return;
    end
end


% Cleans up the script by moving back to the src directory.
function cleanup()
    fprintf('Moving back to src directory.\n');
    cd ../../;
end
