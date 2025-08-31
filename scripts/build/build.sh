#!/bin/bash
# Get postbuild.sh funcs
source "$(dirname "$0")/prebuild.sh"
buildsubModule
docker build --rm  $@ -t  rltoolkit -f "$(dirname "$0")/../../docker/rltoolkit.Dockerfile" "$(dirname "$0")/../.."


