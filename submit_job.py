#!/usr/bin/env python

import subprocess
import pandas as pd

parameter_file_full_path = "/home/students/yiqunc/stat572/code/fig_1_exp.csv"

exp_id_df = pd.read_csv(parameter_file_full_path,index_col=0)
exp_id_df = exp_id_df[8:9]
# df_temp = exp_id_df[20:25]

for i, job in exp_id_df.iterrows():
    #job = row
        #job = job[0]
    job = [str(x) for x in job]
    
    qsub_command = """qsub -q w-bigmem.q -v "time={0},rate={1}" /home/students/yiqunc/stat572/code/ell_0_spike.sh""".format(*job)           
     #print qsub_command # Uncomment this line when testing to view the qsub command
     # Comment the following 3 lines when testing to prevent jobs from being submitted
    exit_status = subprocess.call(qsub_command, shell=True)
    if exit_status is 1:  # Check to make sure the job submitted
        print("Job {0} failed to submit".format(qsub_command))
print("Done submitting jobs!")



