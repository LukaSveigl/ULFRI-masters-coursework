# DL-Project

A repository for the project of the Deep learning course.

## Installation and running

### Local setup

To set up this project on a local machine, a CUDA capable GPU is required. If you wish to set up the project locally, follow the steps below:

- Clone this repository into your desired location.
- Install the required packages (available in the requirements.txt file in the src directory)
- Run one of the files in the src directory.

### HPC setup

To set up this project on the Arnes HPC, follow the steps below:

- Clone this repository into your desired location.
- In the src directory create a directory called containers.
- Move into the containers directory and run the following command: singularity build ./container-torch.sif docker://pytorch/pytorch:2.2.2-cuda12.1-cudnn8-devel
    - Install the required packages (available in the requirements.txt file in the src directory)
    - The packages must be installed using the following command `singularity exec ./containers/container-torch.sif pip install <package-name>`
    - Not all packages from the `requirements.txt` file must be installed. If installing by hand, simply install sentence-transformers, scikit-learn and numpy.
- Run `sbatch sbatch_run.sh`

