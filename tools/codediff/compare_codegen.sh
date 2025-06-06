#!/bin/bash
# SPDX-FileCopyrightText: Copyright (c) 2023-present NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause
#
# compare_codegen.sh - compare generated CUDA kernels between git commits
#
# This script is made to compare generated kernels when making changes to
# codegen. Invoking it without any arguments will checkout the reference commit
# as well as this commit. The default reference commit is the merge-base equal
# to `git merge-base origin/main HEAD`. For each, it will build the project in
# release mode then invoke all binary and python tests, saving the generated
# cuda kernels to .cu files. It will then diff these files and report which
# ones changed. The exit code is 1 if there are differences.
#
# The -r option controls the git ref to compare against. The -o option lets you
# specify the output directory.
#
# If the -- option is given then instead of running all tests, we will use the
# rest of the command line as the test to run. For example, to compare the
# generated code for a single binary test, you could use:
#
#   tools/compare_codegen.sh -- build/test_nvfuser --gtest_filter='*TestFoo*'
#
# or to run all benchmarks you can use:
#
#   tools/compare_codegen.sh -- build/nvfuser_bench \
#       --benchmark_filter=NvFuserScheduler \
#       --benchmark_repetitions=1 \
#       --benchmark_min_time=0
#
# In those cases, the outputs will be placed in a subdirectory of the output
# directory for each commit labelled "custom_command_$LAUNCHTIME" where
# $LAUNCHTIME is a string representing the time this script was launched.
#
# The -t option applies only if a custom command is provided, and specifies the
# command type passed to run_command.sh. Should be one of GOOGLEBENCH,
# GOOGLETEST, PYTEST, or UNKNOWN.
#
# By default, `pip install -v --no-build-isolation python/` is used to rebuild
# the project. You can also set environment variable CUSTOM_BUILD_COMMAND if
# your build is different.

set -e
set -o pipefail

comparetoref=$(git merge-base origin/main HEAD)

usage() {
  echo "Usage: $0 [-h] [-q] [-r ${comparetoref}] [-o codegen_comparison] [-t command_type] [-- custom command to run]"
  echo -n "If given, the custom command should only run a single executable. "
  echo "If multiple executables are run, kernel files may be overwritten."
}

# top-level directory of nvfuser repo
nvfuserdir="$(dirname "$(dirname "$(dirname "$(readlink -f "${BASH_SOURCE[0]}")")")")"

outdir=$nvfuserdir/codegen_comparison

while getopts "r:o:t:hq" arg
do
  case $arg in
    r)
      comparetoref=$OPTARG
      ;;
    o)
      outdir=$OPTARG
      ;;
    t)
      commandtype=$(echo "$OPTARG" | tr '[:lower:]' '[:upper:]')
      ;;
    q)
      quiet=1
      ;;
    h | ?)
      usage
      exit 1
      ;;
  esac
done
# getopts stops parsing if it sees "--". We can detect that case and record command
while [[ $# -gt 0 ]]
do
  if [[ "$1" == "--" ]]
  then
    hascustomcommand=1
    shift
    break
  fi
  shift
done
customcommand=$*

if [[ $(git status --porcelain --untracked-files=no) ]]
then
    echo "Must use git checkout in order to compare. Commit changes before running this script."
    exit 1
fi

# These braces are important since they will force bash to read the entire file
# into memory before execution. Otherwise, if the script changes during
# execution we might run the updated code instead of the original. See the
# later note about $scriptdir which addresses this problem for other scripts
# called by this one.
{

# save current commit and current head so we can switch back to branch
currentcommit=$(git describe --always --long)
origcommit=$currentcommit
# When head is detached, symbolic-ref will fail. Then we just use the commit SHA.
orighead=$(git symbolic-ref --short HEAD || git rev-parse HEAD)

comparecommit=$(git describe --always --long "$comparetoref")

# record launch time to name custom command directories consistently
launchtime=$(date +%Y%m%d_%H%M%S)

# We will be modifying the git repository. Since this might change the scripts
# we need to run, we first copy the codediff directory to a temp dir, then we
# can safely run the copied scripts from there.
scriptdir=$(mktemp -d -t codediffXXXXXX)
cp -r "$nvfuserdir/tools/codediff/"* "$scriptdir/"

movecudafiles() {
    find . -maxdepth 1 \( -name '__tmp_*.cu' -o -name '__tmp_*.ptx' \) -exec mv '{}' "$1" \;
}

cleanup() {
    numkernels=$(find . -maxdepth 1 -name '__tmp_*.cu' -o -name '__tmp_*.ptx' | wc -l)

    if (( numkernels > 0 ))
    then
        backupdir=$outdir/${currentcommit}-interrupted
        echo "Interrupted. Backing up $numkernels .cu and .ptx files to $backupdir"
        mkdir -p "$backupdir"
        movecudafiles "$backupdir"
    fi

    git -c advice.detachedHead=false checkout "$orighead"
    git submodule update --init --recursive

    rm -rf "$scriptdir"
}

trap "cleanup" EXIT

collect_kernels() {
    outdir=$1
    commit=$2

    # Make sure we are doing a clean rebuild. Otherwise we might get linking error.
    rm -rf $nvfuserdir/python/build

    git -c advice.detachedHead=false checkout "$commit"
    git submodule update --init --recursive
    currentcommit=$commit

    customcmddir=$outdir/$commit/custom_command_$launchtime

    binarytestdir=$outdir/$commit/binary_tests
    pyfrontenddir=$outdir/$commit/python_frontend_tests
    pyopsdir=$outdir/$commit/python_ops_tests
    pyschedopsdir=$outdir/$commit/python_schedule_ops_tests

    # Test for output directories and return early if they exist. This
    # avoids rebuilds when we are changing code and comparing repeatedly to
    # the same earlier commit.
    if [[ $hascustomcommand ]]
    then
      if [[ -d "$customcmddir/cuda" ]]
      then
          return
      fi
    else
      if [[ -d "$binarytestdir/cuda" && -d "$pyfrontenddir/cuda" &&
          -d "$pyopsdir/cuda" && -d "$pyschedopsdir/cuda" ]]
      then
          return
      fi
    fi

    # Build in Release mode
    (
        cd "$nvfuserdir"
        CUSTOM_BUILD_COMMAND="${CUSTOM_BUILD_COMMAND:-pip install -v --no-build-isolation python/}"
        bash -c "${CUSTOM_BUILD_COMMAND}"
    )

    # Make tests reproducible
    export NVFUSER_TEST_RANDOM_SEED=0
    export NVFUSER_DISABLE=parallel_compile
    export DEBUG_SERDE=disable
    # run tests and benchmarks with cuda_to_file and dump output to files

    mkdir -p "$outdir/$commit"

    # Name each run with commit abbrev, tell it where nvfuser lives since the
    # run_command script was copied to $scriptdir before git operations.
    bashcmd=("bash" "$scriptdir/run_command.sh" "-f" "$nvfuserdir" "-n" "$commit")

    if [[ -n $quiet ]]
    then
        bashcmd+=("-q")
    fi

    if [[ $hascustomcommand ]]
    then
        if [[ -n $commandtype ]]
        then
            "${bashcmd[@]}" -t "$commandtype" -o "$customcmddir" -- "${customcommand[@]}"
        else
            "${bashcmd[@]}" -o "$customcmddir" -- "${customcommand[@]}"
        fi
    else
        # python tests
        # Using -s to disable capturing stdout. This is important as it will let us see which tests creates each .cu file
        "${bashcmd[@]}" -o "$pyopsdir" -- \
            python -m pytest "$nvfuserdir/python_tests/test_ops.py" -n 0 -v -s --color=yes
        "${bashcmd[@]}" -o "$pyschedopsdir" -- \
            python -m pytest "$nvfuserdir/python_tests/test_schedule_ops.py" -n 0 -v -s --color=yes
        "${bashcmd[@]}" -o "$pyfrontenddir" -- \
            python -m pytest "$nvfuserdir/python_tests/test_python_frontend.py" -n 0 -v -s --color=yes

        # binary tests
        "${bashcmd[@]}" -o "$binarytestdir" -- \
            "$nvfuserdir/build/test_nvfuser" --gtest_color=yes
    fi
}

collect_kernels "$outdir" "$origcommit"
collect_kernels "$outdir" "$comparecommit"

cleanup

# Diff produced files and produce report
set +e  # exit status of diff is 1 if there are any mismatches
found_diffs=0
for d in "$outdir/$origcommit"/*
do
    b=$(basename "$d")
    # Note we use nvfuserdir here instead of scriptdir since we should be back on
    # original commit by now
    if ! python "$nvfuserdir/tools/codediff/diff_report.py" \
        "$outdir/$comparecommit/$b" "$outdir/$origcommit/$b" \
        -o "$outdir/codediff_${comparecommit}_${origcommit}_${b}.html" \
        --html --hide-diffs;
    then
        found_diffs=1
    fi
done
exit "$found_diffs"
}
