# This file contains the setup functionality for the project. Simply
# run it to setup the correct output directories and run the truth_generator.py
# and detector.py, which will generate the output images used by the LBP recognizer.

import os, subprocess

if __name__ == '__main__':
    print('Running setup.py...')
    print('Creating output directories.')
    
    os.mkdir('out')
    os.mkdir('out/computed')
    os.mkdir('out/cr_computed')
    os.mkdir('out/truths')
    os.mkdir('out/cr_truths')

    print('Running python scripts. This might take a while.')
    subprocess.run(['python', 'src/truth_generator.py'])
    subprocess.run(['python', 'src/detector.py'])

    print('Done.')
