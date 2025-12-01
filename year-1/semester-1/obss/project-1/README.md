# OBSS-1

A repository for the first seminary work of the "Processing of biomedical signals and images" masters course.

## Description

This project is an implementation of a QRS detector supplemented by the use of Blood pressure signals described
in this article: [https://ucilnica.fri.uni-lj.si/pluginfile.php/78415/mod_assign/intro/Yang_ecg_bp.pdf](https://ucilnica.fri.uni-lj.si/pluginfile.php/78415/mod_assign/intro/Yang_ecg_bp.pdf). The program tries to detect the peaks of a heartbeat by combing trough the peaks detected from
the ECG signals, sandwiched between peaks detected from the Blood pressure signals.

## Data

Data used in this project has been gathered from the CinC Challenge 2014. It uses the files present in the `set-p`directory. All of the .dat, .hea and .atr files are used, but only the ECG and BP signals are extracted.

## Requirements

To run this project, 2 components are necessary:
1. The WFDB itself. To install the WFDB, go to their website and follow their instructions for your specific operating system. For Windows, the recommended method is the use of precompiled binaries. If you do not wish to go through the Windows installation process, the binaries are provided in the `wfdb` directory.
2. The WFDB toolbox. The toolbox is also provided here, but to be safe you can go to the WFDB toolbox website and run the matlab installation commands. Whatever you do, do not modify the WFDB system variable, as that will override the toolbox's default value and the programs will not search for the records in the current directory.

## How to run

To run this project, the appropriate project structure must be assembled. The data must be present in the `data` directory, in the `set-p` subdirectory. For a record, all of it's data, header and attribute files must be present.

To run the detector on a single record, run the `yang.m` file from either Octave or MATLAB's consoles, and specify the record name, e.g.: `yang 100`. If you wish to change the training set where the program looks for the record, simply modify the `parseargs` function.

To run the detector on all records in a specific dataset, simply run the `runner.m` file. The program will generate a `results.txt` file which contains the results of the evaluation.

In both cases, the program will generate the appropriate `.qrs` and `.wabp` files, calculate the peaks, and write them in an `.asc` file.

## Future work

In the future, the detection could be augmented by using supplemental data in the form of, for example, EMG signals.