#Executable  = /usr/bin/bash
#arguments = one

Executable = ./one

request_cpus = 100

Log         = one.log
Output      = one.out
Error       = one.err

getenv = True

# as long as jobs run at DESY Zeuthen batch, please disable file transfer feature
should_transfer_files = no
# request 2GB RAM
request_memory = 8GB

Queue 1

