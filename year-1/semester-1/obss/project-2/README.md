# OBSS-2
A repository for the second seminary work of the "Processing of biomedical signals and images" masters course.

## Description

This project is an implementation of a program that allows the user to estimate the median and peak frequencies for a record from the Term-Preterm EHG Database. The program estimates the frequencies on 3 of the signals from the record, filtered using a 4-pole band-pass Butterworth filter from 0.3Hz to
4Hz. The 1. and 3. signals are measured on horizontally positioned electrodes (the 3. signal being closer to the cervix) while the 2. signal is meaasured on vertically positioned electrodes.

## Data

Data used in this project has been gathered from the [Term-Preterm EHG Database](https://physionet.org/content/tpehgdb/1.0.1/).

## Requirements

To run this project, 2 components are necessary:

- The WFDB itself. To install the WFDB, go to their website and follow their instructions for your specific operating system. For Windows, the recommended method is the use of precompiled binaries. It is best practice to add the binaries directory to your system path.
- The WFDB toolbox. The toolbox is also provided here, but to be safe you can go to the WFDB toolbox website and run the matlab installation commands. Whatever you do, do not modify the WFDB system variable, as that will override the toolbox's default value and the programs will not search for the records in the current directory.

## How to run

To run this project, the appropriate project structure must be assembled. The data must be present in the `data` directory, in the `tpehgdb` subdirectory. For a record, all of it's data, header and attribute files must be present.

To run the estimators on a single record, run the estimator.m file from either Octave or MATLAB's consoles, and specify the record name, e.g.: estimator tpehg725. 

To run the detector on either 4 random records, each from it's own distinct group or on pre-specified records, simply run the runner.m file. The program will generate Matlab figures, which will additionally be saved in a `.png` file with the record name.
