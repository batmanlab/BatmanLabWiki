#!/usr/bin/env python
from __future__ import print_function


import sys
import os
import time
from os import listdir
from os.path import isfile, join
from os import walk
import subprocess
#from basis.utilities import execute
from execUtils import execute


import tempfile
import shutil

import argparse
from argparse import RawTextHelpFormatter



def tic():
    #Homemade version of matlab tic and toc functions
    import time
    global startTime_for_tictoc
    startTime_for_tictoc = time.time()

def toc():
    import time
    if 'startTime_for_tictoc' in globals():
        print( "Elapsed time is " + str(time.time() - startTime_for_tictoc) + " seconds." )
    else:
        print( "Toc: start time not set" )




def submit_PSCBridge(cmd, 
                 stdOut,
                 stdErr,
                 jobName,
                 slurmOptions=[],
                 MATLABVersion=None,
                 simulate=True):
  """
   This is a helper function to submit slurm task to the c3ddb cluster
   
  """
  def getMCRDir(matVer):
    if matVer=='v901':
      mcrDir='/pylon2/ms4s88p/batmangh/installed/MATLAB_Compiler_Runtime/v901/'
    else:
      raise ValueError('install this version, I don''t know where the engine is!')
    return mcrDir  

  slurmLauncher = '/pylon2/ms4s88p/batmangh/Projects/LungProject/src/scripts/slurmLauncher_bridge.sh'
  matlabExecHelper='/pylon2/ms4s88p/batmangh/Projects/LungProject/build/bin/runMatlabExec'
  
  cmdLine = ['sbatch']
  cmdLine = cmdLine +   ['-o' , stdOut, '-e', stdErr,  '--job-name', jobName]  
  cmdLine = cmdLine + slurmOptions 
  

  cmdLine = cmdLine + [slurmLauncher]
  if not(MATLABVersion==None):
    MCR_DIR = getMCRDir(MATLABVersion)
    cmdLine = cmdLine + [matlabExecHelper, MCR_DIR]

  cmdLine = cmdLine + cmd

  cmdLineString = ''
  for c in cmdLine:
      cmdLineString = cmdLineString + ' ' + str(c)

  if simulate:
    print( 'Simulate : ', cmdLineString )
  else:
    print( 'Running : ', cmdLineString )
    # make sure all entries of the cmdLine is string
    cmdLine = [str(c)  for c in cmdLine ]
    e,o = execute(cmdLine,simulate=False, stdout=True)
    jobID = o.split()[-1]
    return jobID
    #p = subprocess.Popen(cmdLine, shell=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    #for line in p.stdout:
    #  print( line )
    #p.wait()





def submit_c3ddb(cmd, 
                 stdOut,
                 stdErr,
                 jobName,
                 slurmOptions=[],
                 MATLABVersion=None,
                 simulate=True):
  """
   This is a helper function to submit slurm task to the c3ddb cluster
   
  """
  def getMCRDir(matVer):
    if matVer=='v83':
      mcrDir='/home/batmanghelich/installed/MATLAB_Compiler_Runtime/v83/'
    else:
      raise ValueError('install this version, I don''t know where the engine is!')
    return mcrDir  

  slurmLauncher = 'slurmLauncher_c3ddb.sh'
  matlabExecHelper='runMatlabExec_c3ddb'
  
  cmdLine = ['sbatch']
  cmdLine = cmdLine +   ['-o' , stdOut, '-e', stdErr,  '--job-name', jobName]  
  cmdLine = cmdLine + slurmOptions 
  

  cmdLine = cmdLine + [slurmLauncher]
  if not(MATLABVersion==None):
    MCR_DIR = getMCRDir(MATLABVersion)
    cmdLine = cmdLine + [matlabExecHelper, MCR_DIR]

  cmdLine = cmdLine + cmd

  cmdLineString = ''
  for c in cmdLine:
      cmdLineString = cmdLineString + ' ' + str(c)

  if simulate:
    print( 'Simulate : ', cmdLineString )
  else:
    print( 'Running : ', cmdLineString )
    # make sure all entries of the cmdLine is string
    cmdLine = [str(c)  for c in cmdLine ]
    e,o = execute(cmdLine,simulate=False, stdout=True)
    jobID = o.split()[-1]
    return jobID
    #p = subprocess.Popen(cmdLine, shell=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    #for line in p.stdout:
    #  print( line )
    #p.wait()




def   checkJobRunningOnC3ddbCluster(jobIDs, sleepTime=2):
      finishedFlag = [False]*len(jobIDs) 

      while  not(all(finishedFlag)):
        print( "checking ..." )
        for cnt in range(len(jobIDs)):
            j = jobIDs[cnt]
            flag = finishedFlag[cnt]

            if flag:
                continue

            e,o = execute('squeue',stdout=True,quiet=True) 
            if o.find(j)>0:
               print( j + " is still running .... " )
            else:
               flag = True

            if flag:
                print( "it seems  ", j , "is finished !!" )
                finishedFlag[cnt] = True           

            time.sleep(sleepTime)        
    



if __name__ == '__main__':
    #parse input arguments 
    parser = argparse.ArgumentParser(description="""This program performs some of the modules defined in the utils. 
      Here is a list of available options : 
          submitToc3ddb  : to submit a job to the c3ddb cluster
                          Here is somes examples:
                          Ex1 : ./utils.py submitToc3ddb  --jobName  testJob  --simulate    --isMATLAB   --slurmOpt "--time 1:00:00"   --cmd test.sh "--opt 1  --opt 3  --opt 5"
        
                                                                                                            
                                                    
                                                    .""", formatter_class=RawTextHelpFormatter )
                                                       
    #parser.add_argument('--action', help='Specify the action', required=True,  type=str, choices=['extractFeaturesFromSuperVoxels'])
    subparsers = parser.add_subparsers(dest='action')


    #  subparser for submit jobs to C3ddb cluster
    parser_submit2C3ddb = subparsers.add_parser('submitToc3ddb')
    # add a required argument
    parser_submit2C3ddb.add_argument('--cmd', nargs='+', help="Command line to run on the cluster", required=True)
    parser_submit2C3ddb.add_argument('--slurmOpt', nargs='+', help='set of options for Slurm', required=False, default=[])
    parser_submit2C3ddb.add_argument('--isMATLAB', action='store_true', help='Is this a MATLAB code ? ', default=False)
    parser_submit2C3ddb.add_argument('--matVersion', type=str, help='MATLAB version', default='v83', required=False)
    parser_submit2C3ddb.add_argument('--jobName', type=str, help='name of the job', required=True)
    parser_submit2C3ddb.add_argument('--logFileRoot', type=str, help='Root of the log files', required=False, default='/scratch/users/batmanghelich/COPDGene/work/kayhan/logFiles')
    parser_submit2C3ddb.add_argument('--simulate', action='store_true', help='dry-run', default=False, required=False)

    #  subparser for submit jobs to bridge cluster
    parser_submit2Bridge = subparsers.add_parser('submitToBridge')
    # add a required argument
    parser_submit2Bridge.add_argument('--cmd', nargs='+', help="Command line to run on the cluster", required=True)
    parser_submit2Bridge.add_argument('--slurmOpt', nargs='+', help='set of options for Slurm', required=False, default=['-p RM-shared'])
    parser_submit2Bridge.add_argument('--isMATLAB', action='store_true', help='Is this a MATLAB code ? ', default=False)
    parser_submit2Bridge.add_argument('--matVersion', type=str, help='MATLAB version', default='v83', required=False)
    parser_submit2Bridge.add_argument('--jobName', type=str, help='name of the job', required=True)
    parser_submit2Bridge.add_argument('--logFileRoot', type=str, help='Root of the log files', required=False, default='/home/batmangh/MyPylon1Space/logFiles/COPDGene')
    parser_submit2Bridge.add_argument('--simulate', action='store_true', help='dry-run', default=False, required=False)



    args = vars(parser.parse_args())
    print( args )

    if args['action']=='submitToc3ddb':
       cmd = reduce(list.__add__, map(str.split,args['cmd'] )  )
       print( cmd )
       if not(args['slurmOpt']==[]):
          slurmOpt = reduce(list.__add__, map(str.split,args['slurmOpt'] )  )
          print(slurmOpt)
       else:
          slurmOpt = [] 
       jobName = args['jobName']
       simulate = args['simulate']
       stdOut = args['logFileRoot'] + '/' + jobName + '.stdout'
       stdErr = args['logFileRoot'] + '/' + jobName + '.stderr'
       if args['isMATLAB']:
          matVersion = args['matVersion']
       else:
          matVersion = None

       submit_c3ddb(cmd, 
                 stdOut,
                 stdErr,
                 jobName,
                 slurmOptions=slurmOpt,
                 MATLABVersion=matVersion,
                 simulate=simulate)
 

    if args['action']=='submitToBridge':
       cmd = reduce(list.__add__, map(str.split,args['cmd'] )  )
       print( cmd )
       if not(args['slurmOpt']==[]):
          slurmOpt = reduce(list.__add__, map(str.split,args['slurmOpt'] )  )
          print(slurmOpt)
       else:
          slurmOpt = [] 
       jobName = args['jobName']
       simulate = args['simulate']
       stdOut = args['logFileRoot'] + '/' + jobName + '.stdout'
       stdErr = args['logFileRoot'] + '/' + jobName + '.stderr'
       if args['isMATLAB']:
          matVersion = args['matVersion']
       else:
          matVersion = None

       submit_PSCBridge(cmd, 
                 stdOut,
                 stdErr,
                 jobName,
                 slurmOptions=slurmOpt,
                 MATLABVersion=matVersion,
                 simulate=simulate)


