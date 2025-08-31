#!/bin/bash
docker build --rm  $@ -t  rltoolkit -f "$(dirname "$0")/../../docker/rltoolkit.Dockerfile" "$(dirname "$0")/../.."