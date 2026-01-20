

#%%
if __name__ == '__main__':
    
    import os
    import subprocess

    '''
    Python script to execute the ECICE model with specified input file list. Input file list in dir: /home/waynedj/Projects/ecice/processing
    NB: Ensure that the ECICE executable is compiled and available in the current directory.
    1. Input file list: 'input_filelist_v002.txt'
    2. Number of input ice types: 3
    3. Number of frequncy channels: 4
    4. Distribution files for different ice types.
    5. Maximum number of iterations: 1000
    6. Input prompts:
    	Apply SEALION Weather Correction? (y/n)
	    Apply Open Water Filter? (y/n)
	    Substitute low 85 or 89 GHz data with 37 GHz data? (y/n)
	    Apply Cloud Filter? (y/n)
    '''
   
    # Absolute path to the ECICE processing directory
    ecice_dir = '/home/waynedj/Projects/ecice/processing'

    # Build the full path to the executable
    ecice_exec = os.path.join(ecice_dir, 'ecice')

    # Make sure the executable exists
    if not os.path.isfile(ecice_exec):
        raise FileNotFoundError(f"ECICE executable not found: {ecice_exec}")

    # Command to run
    cmd = [
        ecice_exec,
        'input_filelist_v001.txt',
        '3',
        '4',
        'Distributions/AMSR2-Dist-OW-AQ2_07_08.txt',
        'Distributions/AMSR2-Dist-FYI-AQ2_04_06.txt',
        'Distributions/AMSR2-Dist-YI-WRSplnnew.txt',
        'Distributions/AMSR2-Dist-MYI-AQ2_01_03.txt',
        '1000'
    ]

    # Redirect standard input from the prompts file
    input_file = os.path.join(ecice_dir, 'input_prompts_v001.txt')

    # Execute ECICE in its directory
    subprocess.run(cmd, cwd=ecice_dir, stdin=open(input_file, 'r'))
    
# %%
