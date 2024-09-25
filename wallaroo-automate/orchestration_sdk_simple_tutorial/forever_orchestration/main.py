import subprocess
from time import sleep

subprocess.run(["find", "."]) 
subprocess.run(["find", "/usr"]) 
subprocess.run(["find", "/"]) 

sleep(30000000)
