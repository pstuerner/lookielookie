#!/bin/bash

scriptPath=$(dirname "$(readlink -f "$0")")
source "${scriptPath}/.env.sh"

# the docker-compose variables should be available here
lookielookie timeseries