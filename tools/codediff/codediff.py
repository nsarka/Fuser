# SPDX-FileCopyrightText: Copyright (c) 2023-present NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause
"""
Find corresponding .cu files for matching tests, even when new tests are
introduced between two commits. Diffs are displayed and the return value is the
number of mismatched corresponding tests.

Tests are skipped if they produce different numbers of .cu files, or if they
exist in only one of the given runs.

Example usage:
    python tools/diff_codegen_nvfuser_tests.py \
            codegen_comparison/{$commit1,$commit2}/binary_tests
"""

from dataclasses import asdict, dataclass, field, InitVar
import difflib
from enum import Enum
import os
import re
import subprocess
import sys

from dataclasses_json import dataclass_json


@dataclass_json
@dataclass
class GitRev:
    full_hash: str
    diff: str | None = None
    abbrev: str = field(init=False)
    title: str = field(init=False)
    author_name: str = field(init=False)
    author_email: str = field(init=False)
    author_time: str = field(init=False)
    commit_time: str = field(init=False)

    def __post_init__(self):
        self.abbrev = (
            subprocess.run(
                ["git", "rev-parse", "--short", self.full_hash], capture_output=True
            )
            .stdout.strip()
            .decode("utf-8")
        )
        for line in (
            subprocess.run(
                ["git", "branch", "--quiet", "--color=never", self.full_hash],
                capture_output=True,
            )
            .stdout.strip()
            .decode("utf-8")
            .splitlines()
        ):
            # Possible output:
            #
            #     main
            #     * scalar_seg_edges
            #
            # In this case, we have checked out the HEAD of the
            # scalar_seg_edges branch. Here we just strip the *.
            if line[0] == "*":
                line = line[2:]
                in_branches.append(line)

        def git_show(fmt) -> str:
            return (
                subprocess.run(
                    [
                        "git",
                        "show",
                        "--no-patch",
                        f"--format={fmt}",
                        self.full_hash,
                    ],
                    capture_output=True,
                )
                .stdout.strip()
                .decode("utf-8")
            )

        self.title = git_show("%s")
        self.author_name = git_show("%an")
        self.author_email = git_show("%ae")
        self.author_time = git_show("%ad")
        self.commit_time = git_show("%cd")


@dataclass_json
@dataclass
class LaunchParams:
    blockDim: tuple[int]
    gridDim: tuple[int]
    dynamic_smem_bytes: int


@dataclass_json
@dataclass
class CompiledKernel:
    filename: str
    code: str | None = None
    ptx: str | None = None
    ptxas_info: str | None = None
    launch_params_str: str | None = None
    launch_params: LaunchParams | None = None
    gmem_bytes: int = 0
    smem_bytes: int = 0
    cmem_bank_bytes: list[int] | None = None
    registers: int | None = None
    stack_frame_bytes: int = 0
    spill_store_bytes: int = 0
    spill_load_bytes: int = 0
    mangled_name: str | None = None
    arch: str | None = None
    index_type: str | None = None

    def __post_init__(self):
        self.parse_ptxas()
        self.parse_launch_params()

    def parse_ptxas(self):
        # Example input:
        #
        #   ptxas info    : 307 bytes gmem
        #   ptxas info    : Compiling entry function
        #   '_ZN76_GLOBAL__N__00000000_37___tmp_kernel_pointwise_f0_c1_r0_g0_cu_8995cef2_3255329nvfuser_pointwise_f0_c1_r0_g0ENS_6TensorIfLi2ELi2EEES1_S1_'
        #   for 'sm_86'
        #   ptxas info    : Function properties for
        #   _ZN76_GLOBAL__N__00000000_37___tmp_kernel_pointwise_f0_c1_r0_g0_cu_8995cef2_3255329nvfuser_pointwise_f0_c1_r0_g0ENS_6TensorIfLi2ELi2EEES1_S1_
        #   ptxas         .     0 bytes stack frame, 0 bytes spill stores, 0 bytes spill loads
        #   ptxas info    : Used 203 registers, 16 bytes smem, 472 bytes cmem[0], 8 bytes cmem[2]
        #
        # Here we parse this into the fields presented, and we replace the
        # mangled kernel name since it includes the kernel number and is
        # useless for the purposes of diffing since the kernel signature is
        # already included.
        if self.ptxas_info is None:
            return

        m = re.search(r"Compiling entry function '(.*)' for '(.*)'", self.ptxas_info)
        if m is not None:
            self.mangled_name, self.arch = m.groups()

        def find_unique_int(pattern) -> int | None:
            assert self.ptxas_info is not None
            m = re.search(pattern, self.ptxas_info)
            return 0 if m is None else int(m.groups()[0])

        self.stack_frame_bytes = find_unique_int(r"(\d+) bytes stack frame")
        self.spill_store_bytes = find_unique_int(r"(\d+) bytes spill stores")
        self.spill_load_bytes = find_unique_int(r"(\d+) bytes spill loads")
        self.registers = find_unique_int(r"(\d+) registers")
        self.gmem_bytes = find_unique_int(r"(\d+) bytes gmem")
        self.smem_bytes = find_unique_int(r"(\d+) bytes smem")

        self.cmem_bank_bytes = []
        cmem_banks = 0
        for m in re.finditer(r"(\d+) bytes cmem\[(\d+)\]", self.ptxas_info):
            nbytes_str, bank_str = m.groups()
            bank = int(bank_str)
            if len(self.cmem_bank_bytes) <= bank:
                self.cmem_bank_bytes += [0] * (bank + 1 - len(self.cmem_bank_bytes))
            self.cmem_bank_bytes[bank] = int(nbytes_str)
            cmem_banks += 1

    def parse_launch_params(self):
        # If NVFUSER_DUMP=launch_param is given we will get a line like this for every launch:
        #   Launch Parameters: BlockDim.x = 32, BlockDim.y = 2, BlockDim.z = 2, GridDim.x = 8, GridDim.y = 8, GridDim.z = -1, Smem Size = 49152
        # This is not done by default since we might have hundreds of thousands of these lines.
        # Still, if we recognize it, we will parse this info. If there are
        # multiple lines, we just check that they are all equal and if not then
        # we keep the first version and print a warning.
        if self.launch_params_str is None:
            return

        for line in self.launch_params_str.splitlines():
            m = re.search(
                r"Launch Parameters: BlockDim.x = (.*), BlockDim.y = (.*), BlockDim.z = (.*), "
                r"GridDim.x = (.*), GridDim.y = (.*), GridDim.z = (.*), Smem Size = (.*)$",
                line,
            )
            bx, by, bz, gx, gy, gz, s = m.groups()
            lp = LaunchParams((bx, by, bz), (gx, gy, gz), s)
            if self.launch_params is None:
                self.launch_params = lp
            else:
                if lp != self.launch_params:
                    # Found multiple mismatched launch params for one kernel. Only using first
                    return


@dataclass_json
@dataclass
class BenchmarkResult:
    gpu_time: float
    gpu_time_unit: str
    cpu_time: float
    cpu_time_unit: float
    iterations: int | None = None


@dataclass_json
@dataclass
class CompiledTest:
    """One grouping of kernels. A run holds multiple of these"""

    name: str
    kernels: list[CompiledKernel]
    passed: bool = True
    benchmark_result: BenchmarkResult | None = None


class CommandType(str, Enum):
    """Denotes what type of command was run"""

    UNKNOWN = "UNKNOWN"
    GOOGLETEST = "GOOGLETEST"
    GOOGLEBENCH = "GOOGLEBENCH"
    PYTEST = "PYTEST"

    def __str__(self):
        return self.name

    @classmethod
    def from_string(cls, type_str: str):
        l = type_str.lower()
        if l[:3] == "unk":
            # Specified unknown. Don't print warning
            return cls.UNKNOWN
        elif l == "gtest" or l == "googletest":
            return cls.GOOGLETEST
        elif l == "gbench" or l == "googlebench":
            return cls.GOOGLEBENCH
        elif l == "pytest":
            return cls.PYTEST
        else:
            print(
                f"WARNING: Unrecognized command type '{type_str}'. Parsing as UNKNOWN.",
                file=sys.stderr,
            )
            return cls.UNKNOWN


class LogParser:
    """General parser for STDOUT of NVFuser commands

    This parser does not group into individual tests, but rather places all
    kernels into a single CompiledTest whose name is "Ungrouped Kernels".
    """

    def __init__(self, log_file: str):
        self.compile_regex()

        self.kernel_map: dict[str, CompiledTest] = {}

        self.reset_test_state()

        self.parse(log_file)

    def compile_regex(self):
        # regex for stripping ANSI color codes
        self.ansi_re = re.compile(r"(\x9B|\x1B\[)[0-?]*[ -\/]*[@-~]")

    def reset_kernel_state(self):
        self.current_file = None
        self.ptxas_info = ""
        self.launch_params_str = ""

    def reset_test_state(self):
        """Initialize temporary variables used during parsing pass"""
        self.reset_kernel_state()
        self.current_test = None
        self.kernels = []

    def parse(self, log_file: str):
        for line in open(log_file, "r").readlines():
            line = self.ansi_re.sub("", line.rstrip())
            self.parse_line(line)
        self.finalize()

    def finalize_kernel(self):
        if self.current_file is not None:
            k = CompiledKernel(
                self.current_file,
                ptxas_info=self.ptxas_info,
                launch_params_str=self.launch_params_str,
            )
            self.kernels.append(k)
        self.reset_kernel_state()

    def finalize_test(self, passed: bool):
        assert self.current_test is not None
        self.finalize_kernel()
        new_test = CompiledTest(self.current_test, self.kernels, passed)
        self.kernel_map[self.current_test] = new_test
        self.reset_test_state()
        return new_test

    def finalize(self):
        if len(self.kernels) > 0:
            group_name = "Ungrouped Kernels"
            self.kernel_map[group_name] = CompiledTest(group_name, self.kernels)

    def parse_line(self, line):
        """Parse a line of log. Return True if consumed"""
        if line[:10] == "PRINTING: ":
            if line[-3:] == ".cu":
                self.finalize_kernel()
                # This avoids comparing the .ptx files that are created then
                # removed by the MemoryTest.LoadCache tests
                self.current_file = line[10:]
        elif line[:6] == "ptxas ":
            # NVFUSER_DUMP=ptxas_verbose corresponds to nvcc --ptxas-options=-v
            # or --resources-usage. This always prints after printing the cuda
            # filename
            if self.current_file is None:
                print("WARNING: Cannot associate ptxas info with CUDA kernel")
                return False
            self.ptxas_info += line + "\n"
        elif line[:19] == "Launch Parameters: ":
            if self.current_file is None:
                print("WARNING: Cannot associate launch params with CUDA kernel")
                return False
            self.launch_params_str += line + "\n"
        else:
            return False
        return True


class LogParserGTest(LogParser):
    """Parse output of googletest binaries like test_nvfuser"""

    def parse_line(self, line):
        if super().parse_line(line):
            return True

        if line[:13] == "[ RUN      ] ":
            self.current_test = line[13:]
        elif line[:13] == "[       OK ] ":
            self.finalize_test(True)
        elif line[:13] == "[  FAILED  ] ":
            if self.current_test is not None and self.current_file is not None:
                # Avoid the summary of failed tests, such as
                #   [  FAILED  ] 1 test, listed below:
                #   [  FAILED  ] NVFuserTest.FusionTuringMatmulSplitK_CUDA
                self.finalize_test(False)
        else:
            return False
        return True


class LogParserGBench(LogParser):
    """Parse output of google benchmark binaries like nvfuser_bench"""

    def compile_regex(self):
        super().compile_regex()

        # Example line:
        #   benchmark_name   34.0 us      1.53 ms   2007  /Launch_Parameters[block(2/2/32)/grid(32/2/2)/49664]
        # This is the only kind of line we match for benchmarks. Note that this is printed at the end of each benchmark
        self.result_re = re.compile(
            r"^(?P<testname>\S+)\s+(?P<gputime>[-+\.\d]+)\s+(?P<gputimeunit>\S+)\s+(?P<cputime>[-+\.\d]+)\s+(?P<cputimeunit>\S+)\s+(?P<iterations>\d+).*$"
        )

    def parse_line(self, line):
        if super().parse_line(line):
            return True

        m = re.match(self.result_re, line)
        if m is not None:
            d = m.groupdict()
            self.current_test = d["testname"]
            time = d["gputime"]
            time_unit = d["gputimeunit"]
            cpu = d["cputime"]
            cpu_unit = d["cputimeunit"]
            iterations = d["iterations"]
            # Skip metadata which for nvfuser_bench sometimes includes LaunchParams
            # meta = m.groups()[6]
            new_test = self.finalize_test(True)
            new_test.benchmark_result = BenchmarkResult(
                time, time_unit, cpu, cpu_unit, iterations
            )
            return True
        return False


class LogParserPyTest(LogParser):
    """Parse output of pytest tests.

    Note that the tests must be run with both the -v and -s options
    """

    def compile_regex(self):
        super().compile_regex()

        self.itemlist_re = re.compile(r"Running \d+ items in this shard: (.*)$")

        # match lines like these:
        #  [2024-10-23 02:00:20] tests/python/test_python_frontend.py::TestNvFuserFrontend::test_nanogpt_split_mha_linears
        #  tests/python/test_python_frontend.py::TestNvFuserFrontend::test_nanogpt_split_mha_linears
        #  tests/python/test_python_frontend.py::TestNvFuserFrontend::test_nanogpt_split_mha_linears PASSED

        self.wildcard_testname_re = re.compile(
            r"^((?P<timestamp>\[[\d\-: ]+\]) )?(?P<testname>\S+\.py::\S+)\s?(?P<line>.*)$"
        )

        self.extra_wildcard_testname_re = re.compile(
            r".*?(?P<testname>\S+::\S+) (?P<line>.*)$"
        )

        self.all_test_names: list[str] | None = None

    def parse_line(self, line):
        if self.all_test_names is None:
            m = re.match(self.itemlist_re, line)
            if m is not None:
                # grab the test list
                self.all_test_names = m.groups()[0].split(", ")
                return True

        m = re.match(self.wildcard_testname_re, line)
        if m is not None:
            d = m.groupdict()
            self.current_test = d["testname"]
            line = d["line"]

        if line == "PASSED":
            self.finalize_test(True)
        elif line == "FAILED" and self.current_test is not None:
            self.finalize_test(False)

        if super().parse_line(line):
            return True

        return False


@dataclass_json
@dataclass
class TestRun:
    """A single process that might contain many kernels, grouped into tests"""

    directory: str
    git: GitRev = field(init=False)
    name: str = field(init=False)
    command: str = field(init=False)
    command_type: CommandType = field(init=False)
    exit_code: int = field(init=False)
    env: str = field(init=False)
    gpu_names: str = field(init=False)
    nvcc_version: str = field(init=False)
    # map from name of test to list of kernel base filenames
    kernel_map: dict[str, CompiledTest] = field(default_factory=dict)
    # collecting the preamble lets us skip it when diffing, and lets us compare
    # only the preamble between runs
    preamble: str = field(init=False)
    # The following lets us skip preamble when loading kernels. Note that the
    # preamble can change length due to differing index types, so we can't rely
    # on f.seek()
    preamble_size_lines: int = field(init=False)

    def __post_init__(self):
        if not os.path.isdir(self.directory):
            print(f"ERROR: {self.directory} does not name a directory")
            sys.exit(1)

        try:
            self.name = (
                open(os.path.join(self.directory, "run_name"), "r").read().rstrip()
            )
        except FileNotFoundError:
            self.name = os.path.basename(self.directory)

        # get description of this git rev
        gitdiff = None
        try:
            gitdiff = open(os.path.join(self.directory, "git_diff"), "r").read()
        except FileNotFoundError:
            pass
        git_hash = open(os.path.join(self.directory, "git_hash"), "r").read().rstrip()
        self.git = GitRev(git_hash, diff=gitdiff)

        self.command = open(os.path.join(self.directory, "command"), "r").read()

        try:
            self.command_type = CommandType.from_string(
                open(os.path.join(self.directory, "command_type"), "r").read().rstrip()
            )
        except FileNotFoundError:
            print(
                f"WARNING: Could not find {os.path.join(self.directory, 'command_type')}. "
                "Parsing as UNKNOWN command type means kernels will be ungrouped.",
                file=sys.stderr,
            )
            self.command_type = CommandType.UNKNOWN

        try:
            self.env = ""
            for line in open(os.path.join(self.directory, "env"), "r").readlines():
                # remove $testdir which is set by compare_codegen.sh
                # NOTE: compare_codegen.sh should have already removed these lines
                if re.search(r"^testdir=", line) is None:
                    self.env += line
        except FileNotFoundError:
            self.env = None

        try:
            self.nvcc_version = open(
                os.path.join(self.directory, "nvcc_version"), "r"
            ).read()
        except FileNotFoundError:
            self.nvcc_version = None

        try:
            self.gpu_names = list(
                open(os.path.join(self.directory, "gpu_names"), "r").readlines()
            )
        except FileNotFoundError:
            self.gpu_names = None

        self.exit_code = int(open(os.path.join(self.directory, "exitcode"), "r").read())

        self.compute_kernel_map()

        self.find_preamble()

    def compute_kernel_map(self):
        """
        Compute a map from test name to list of cuda filenames
        """
        logfile = os.path.join(self.directory, "stdout")
        if not os.path.isfile(logfile):
            raise RuntimeError(
                f"Input directory {self.directory} contains no file named 'stdout'"
            )

        if self.command_type == CommandType.GOOGLETEST:
            parser = LogParserGTest(logfile)
        elif self.command_type == CommandType.GOOGLEBENCH:
            parser = LogParserGBench(logfile)
        elif self.command_type == CommandType.PYTEST:
            parser = LogParserPyTest(logfile)
        else:
            # The base class provides a parser that groups everything into a
            # single "test" called "Ungrouped Kernels"
            parser = LogParser(logfile)

        self.kernel_map = parser.kernel_map

    def find_preamble(self):
        """Look for common preamble in collected kernels"""
        preamble_lines = []
        first = True
        files_processed = 0  # limit how many files to check
        for cufile in os.listdir(os.path.join(self.directory, "cuda")):
            cufile_full = os.path.join(self.directory, "cuda", cufile)
            with open(cufile_full, "r") as f:
                for i, line in enumerate(f.readlines()):
                    line = line.rstrip()
                    # we set nvfuser_index_t in the preamble. We ignore that change for the purposes of this diff
                    if line[:8] == "typedef " and line[-17:] == " nvfuser_index_t;":
                        line = "typedef int nvfuser_index_t; // NOTE: index type hard-coded as int for display only"
                    if re.search(r"void (nvfuser|kernel)_?\d+\b", line) is not None:
                        # we arrived at the kernel definition
                        break
                    if first:
                        preamble_lines.append(line)
                    elif i >= len(preamble_lines) or preamble_lines[i] != line:
                        break
                preamble_lines = preamble_lines[:i]
            if len(preamble_lines) == 0:
                # early return if preamble is determined to be empty
                break
            first = False
            files_processed += 1
            if files_processed >= 50:
                break
        self.preamble_size_lines = len(preamble_lines)
        self.preamble = "\n".join(preamble_lines)

    def get_kernel(
        self, test_name, kernel_number, strip_preamble=True
    ) -> CompiledKernel:
        """Get a string of the kernel, optionally stripping the preamble"""
        kern = self.kernel_map[test_name].kernels[kernel_number]
        basename = kern.filename
        fullname = os.path.join(self.directory, "cuda", basename)
        kern.code = ""
        with open(fullname, "r") as f:
            for i, line in enumerate(f.readlines()):
                if kern.index_type is None:
                    m = re.search(r"typedef\s+(\S*)\s+nvfuser_index_t;", line)
                    if m is not None:
                        kern.index_type = m.groups()[0]
                if not strip_preamble or i >= self.preamble_size_lines:
                    # replace kernel934 with kernel1 to facilitate diffing
                    # also match kernel_43 to handle new-style naming with static fusion count
                    kern.code += re.sub(r"\bnvfuser_\d+\b", "nvfuser_N", line)
        kern.code = kern.code.rstrip()
        if strip_preamble and kern.code[-1] == "}":
            # trailing curly brace is close of namespace. This will clean it up so that we have just the kernel
            kern.code = kern.code[:-1].rstrip()
        # find ptx file if it exists
        ptx_basename = os.path.splitext(basename)[0] + ".ptx"
        ptx_fullname = os.path.join(self.directory, "ptx", ptx_basename)
        try:
            kern.ptx = open(ptx_fullname, "r").read().rstrip()
        except FileNotFoundError:
            pass
        return kern

    def join(self, other: "TestRun"):
        """Concatenate other with self"""
        # Stuff that has to match
        assert self.git == other.git
        assert self.preamble_size_lines == other.preamble_size_lines
        assert self.preamble == other.preamble
        assert self.command_type == other.command_type

        # don't update name of this command

        # Append differing nvcc versions as new rows, otherwise keep equal
        if other.nvcc_version != self.nvcc_version:
            self.nvcc_version += other.nvcc_version

        # We expect the env to differ since we are probably joining after
        # running multiple shards on different nodes in parallel. So just
        # concatenate the envs, with a blank line between
        if other.env != self.env:
            self.env += f"\n{other.env}"

        # keep a list of all GPUs involved
        self.gpu_names += other.gpu_names

        # concatenate command as if we ran the commands in sequence
        self.command += f" && {other.command}"

        # if one of the inputs is an error (non-zero), preserve it
        self.exit_code += other.exit_code

        # now merge the kernel maps with one another
        for test_name, test_obj in other.kernel_map.items():
            assert test_obj.name == test_name
            if test_name == "Ungrouped Kernels":
                if test_name in self.kernel_map:
                    self.kernel_map[test_name].kernels += test_obj.kernels
                    if not test_obj.passed:
                        self.kernel_map[test_name].passed = False
                    # Don't merge benchmark results. We should not have
                    # ungrouped kernels with these fields anyway, since we
                    # expect all kernels to fall under some benchmark if the
                    # command_type is known to be a benchmark
                    assert self.kernel_map[test_name].benchmark_result is None
                    assert test_obj.benchmark_result is None
                    continue
            else:
                assert (
                    test_name not in self.kernel_map
                ), f"Cannot join test runs containing the same test {test_name}"
            self.kernel_map[test_name] = test_obj


@dataclass_json
@dataclass
class KernelDiff:
    testname: str
    kernel_num: int
    kernel1: CompiledKernel
    kernel2: CompiledKernel
    diff_lines: InitVar[list[str]] = []
    ptx_diff_lines: InitVar[list[str] | None] = []
    diff: str = field(init=False)
    new_lines: int = 0
    removed_lines: int = 0
    ptx_diff: str | None = None
    new_ptx_lines: int = 0
    removed_ptx_lines: int = 0

    def __post_init__(self, diff_lines: list[str], ptx_diff_lines: list[str] | None):
        self.diff = "\n".join(diff_lines)

        for line in diff_lines:
            if line[:2] == "+ ":
                self.new_lines += 1
            elif line[:2] == "- ":
                self.removed_lines += 1

        if ptx_diff_lines is not None:
            self.ptx_diff = "\n".join(ptx_diff_lines)

            for line in ptx_diff_lines:
                if line[:2] == "+ ":
                    self.new_ptx_lines += 1
                elif line[:2] == "- ":
                    self.removed_ptx_lines += 1


@dataclass_json
@dataclass
class TestDiff:
    testname: str
    test1: CompiledTest
    test2: CompiledTest
    kernel_diffs: list[KernelDiff] | None = None


def sanitize_ptx_lines(lines: list[str]) -> list[str]:
    """Remove comments and remove kernel id"""
    sanitary_lines = []
    for l in lines:
        # Replace mangled kernel names like
        #   _ZN76_GLOBAL__N__00000000_37___tmp_kernel_pointwise_f0_c1_r0_g0_cu_8995cef2_3255329nvfuser_pointwise_f0_c1_r0_g0ENS_6TensorIfLi2ELi2EEES1_S1_
        # or
        #   _ZN76_GLOBAL__N__00000000_37___tmp_kernel_4_cu_8995cef2_3255329nvfuser_4ENS_6TensorIfLi2ELi2EEES1_S1_
        # with
        #   _ZN11kernelscope6kernelENS_6TensorIfLi2ELi2EEES1_S1_

        # demangle first two parts after _ZN and replace with "kernelscope" and "kernel"
        m = re.match(r"^(?P<prefix>^.*\b_Z?ZN)(?P<scopenamelen>\d+)_", l)
        if m is not None:
            d = m.groupdict()
            scopenamelen = int(d["scopenamelen"])
            # demangle second part in remainder after scope name
            remainder = l[(len(d["prefix"]) + len(d["scopenamelen"]) + scopenamelen) :]
            mrem = re.match(r"^(?P<varnamelen>\d+)", remainder)
            if mrem is not None:
                drem = mrem.groupdict()
                varnamelen = int(drem["varnamelen"])
                remainder = (
                    "6kernel" + remainder[len(drem["varnamelen"]) + varnamelen :]
                )
            l = d["prefix"] + "11kernelscope" + remainder

        # Remove comments. This fixes mismatches in PTX "callseq" comments, which appear to be non-repeatable.
        l = re.sub(r"//.*$", "", l)
        sanitary_lines.append(l)
    return sanitary_lines


@dataclass_json
@dataclass
class TestDifferences:
    run1: TestRun
    run2: TestRun
    # either a list of diffs, or different numbers of kernels present
    test_diffs: list[TestDiff] = field(default_factory=list)
    new_tests: list[CompiledTest] = field(default_factory=list)
    removed_tests: list[CompiledTest] = field(default_factory=list)
    total_num_diffs: int = 0
    show_diffs: InitVar[bool] = False
    inclusion_criterion: InitVar[str] = "mismatched_cuda_or_ptx"
    preamble_diff: str = field(init=False)
    env_diff: str = field(init=False)

    def __post_init__(self, show_diffs: bool, kernel_inclusion_criterion: str):
        if self.run1.command != self.run2.command:
            print("WARNING: commands differ between runs", file=sys.stderr)
            print(f"  {self.run1.directory}: {self.run1.command}", file=sys.stderr)
            print(f"  {self.run2.directory}: {self.run2.command}", file=sys.stderr)

        if self.run1.exit_code != self.run1.exit_code:
            print(
                f"WARNING: Exit codes {self.run1.exit_code} and {self.run2.exit_code} do not match.",
                file=sys.stderr,
            )

        self.preamble_diff = "\n".join(
            difflib.unified_diff(
                self.run1.preamble.splitlines(),
                self.run2.preamble.splitlines(),
                fromfile=self.run1.name,
                tofile=self.run2.name,
                n=5,
            )
        )
        if len(self.preamble_diff) > 0:
            print("Preambles differ between runs indicating changes to runtime files")

        self.env_diff = "\n".join(
            difflib.unified_diff(
                self.run1.env.splitlines(),
                self.run2.env.splitlines(),
                fromfile=self.run1.name,
                tofile=self.run2.name,
                n=5,
            )
        )

        for testname, compiled_test1 in self.run1.kernel_map.items():
            if testname not in self.run2.kernel_map:
                compiled_test1.kernels = [
                    self.run1.get_kernel(testname, i)
                    for i in range(len(compiled_test1.kernels))
                ]
                self.removed_tests.append(compiled_test1)
                continue

            compiled_test2 = self.run2.kernel_map[testname]

            test1_kernel_count = len(compiled_test1.kernels)
            test2_kernel_count = len(compiled_test2.kernels)
            minimum_kernel_count = min(test1_kernel_count, test2_kernel_count)
            if test1_kernel_count != test2_kernel_count:
                print(
                    f"WARNING: Test {testname} has {test1_kernel_count} kernels "
                    f"in {self.run1.directory} and {test2_kernel_count} kernels in {self.run2.directory}. "
                    f"Only showing diffs for the first {minimum_kernel_count} kernels in this test.",
                    file=sys.stderr,
                )
                self.test_diffs.append(
                    TestDiff(
                        testname,
                        compiled_test1,
                        compiled_test2,
                        None,
                    )
                )

            kernel_diffs = []
            for kernel_num in range(minimum_kernel_count):
                kern1 = self.run1.get_kernel(testname, kernel_num, strip_preamble=True)
                kern2 = self.run2.get_kernel(testname, kernel_num, strip_preamble=True)
                assert kern1.code is not None
                assert kern2.code is not None

                ptx_diff_lines = None
                if kern1.ptx is not None and kern2.ptx is not None:
                    ptx_diff_lines = list(
                        difflib.unified_diff(
                            sanitize_ptx_lines(kern1.ptx.splitlines()),
                            sanitize_ptx_lines(kern2.ptx.splitlines()),
                            fromfile=self.run1.name,
                            tofile=self.run2.name,
                            n=5,
                        )
                    )

                diff_lines = list(
                    difflib.unified_diff(
                        kern1.code.splitlines(),
                        kern2.code.splitlines(),
                        fromfile=self.run1.name,
                        tofile=self.run2.name,
                        n=5,
                    )
                )
                if (
                    kernel_inclusion_criterion == "all"
                    or (
                        kernel_inclusion_criterion == "mismatched_cuda_or_ptx"
                        and diff_lines is not None
                        and len(diff_lines) > 0
                    )
                    or (
                        kernel_inclusion_criterion
                        in ["mismatched_cuda_or_ptx", "mismatched_ptx"]
                        and ptx_diff_lines is not None
                        and len(ptx_diff_lines) > 0
                    )
                ):
                    kd = KernelDiff(
                        testname,
                        kernel_num + 1,
                        kern1,
                        kern2,
                        diff_lines,
                        ptx_diff_lines=ptx_diff_lines,
                    )
                    if show_diffs:
                        print(testname, kernel_num, kd.diff)
                    self.total_num_diffs += 1
                    kernel_diffs.append(kd)

            if len(kernel_diffs) > 0:
                self.test_diffs.append(
                    TestDiff(
                        testname,
                        compiled_test1,
                        compiled_test2,
                        kernel_diffs,
                    )
                )

        for testname, compiled_test2 in self.run2.kernel_map.items():
            if testname not in self.run1.kernel_map:
                compiled_test2.kernels = [
                    self.run2.get_kernel(testname, i)
                    for i in range(len(compiled_test2.kernels))
                ]
                self.new_tests.append(compiled_test2)

    def hide_env(self):
        """Remove private information like env vars and lib versions"""
        self.run1.env = None
        self.run2.env = None
        self.run1.nvcc_version = None
        self.run2.nvcc_version = None

    def generate_html(self, omit_preamble: bool, max_diffs: bool) -> str:
        """Return a self-contained HTML string summarizing the codegen comparison"""
        import jinja2

        tools_dir = os.path.dirname(__file__)
        template_dir = os.path.join(tools_dir, "templates")
        env = jinja2.Environment(
            loader=jinja2.FileSystemLoader(searchpath=template_dir)
        )
        template = env.get_template("codediff.html")
        # dict_factory lets us provide custom serializations for classes like Enums
        # https://stackoverflow.com/questions/61338539/how-to-use-enum-value-in-asdict-function-from-dataclasses-module
        context = asdict(
            self,
            dict_factory=lambda data: {
                # Serialize CommandType as string so that jinja can recognize it
                field: value.name if isinstance(value, CommandType) else value
                for field, value in data
            },
        )
        context["omit_preamble"] = omit_preamble
        context["max_diffs"] = max_diffs
        head_hash = (
            subprocess.run(["git", "rev-parse", "HEAD"], capture_output=True)
            .stdout.strip()
            .decode("utf-8")
        )
        context["tool_git"] = GitRev(head_hash)
        context["explain_api_url"] = os.environ.get(
            "CODEDIFF_EXPLAIN_API_URL", "/api/explain-diff"
        )

        return template.render(context)


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    subparsers = parser.add_subparsers(title="verb")

    parse_parser = subparsers.add_parser(
        "parse", help="Parse an output directory of run_command.sh into a JSON file"
    )
    parse_parser.add_argument(
        "dir",
        help="Directory containing 'stdout' and 'cuda/' resulting from run_command.sh",
    )
    parse_parser.add_argument("output_json", help="Location to write JSON file")

    def parse_dir(args: dict):
        tr = TestRun(args.dir)

        # load the code for each kernel
        for test_name, compiled_kernel in tr.kernel_map.items():
            for kernel_number in range(len(compiled_kernel.kernels)):
                tr.get_kernel(test_name, kernel_number, strip_preamble=True)

        with open(args.output_json, "w") as f:
            f.write(tr.to_json())

    parse_parser.set_defaults(func=parse_dir)

    join_parser = subparsers.add_parser(
        "join",
        help='Concatenate multiple command JSONs as if they were from a single command. This is useful for "unsharding" jobs that are computed using run_command.sh on multiple shards',
    )
    join_parser.add_argument(
        "-o", "--output", help="Location to write concatenated JSON file"
    )
    join_parser.add_argument(
        "input_jsons", nargs="+", help="Location of incoming JSON files"
    )

    def join_jsons(args: dict):
        assert len(args.input_jsons) > 0

        with open(args.input_jsons[0], "r") as f:
            td = TestRun.from_json(f.read())

        for filename in args.input_jsons[1:]:
            with open(filename, "r") as f:
                td_other = TestRun.from_json(f.read())
            td.join(td_other)

        with open(args.output, "w") as f:
            f.write(td.to_json())

    join_parser.set_defaults(func=join_jsons)

    diff_parser = subparsers.add_parser(
        "diff", help="Compute the difference between two parsed command outputs"
    )
    diff_parser.add_argument(
        "--kernel-inclusion-criterion",
        "-i",
        choices=("mismatched_cuda_or_ptx", "mismatched_ptx", "all"),
        default="mismatched_cuda_or_ptx",
        help="Which kernels should we include?",
    )
    diff_parser.add_argument(
        "--hide-diffs",
        "--no-print-diff",
        action="store_true",
        help="Print diffs to STDOUT?",
    )
    diff_parser.add_argument("input_json1", help="Location of first JSON file")
    diff_parser.add_argument("input_json2", help="Location of second JSON file")
    diff_parser.add_argument("output_json", help="Location to write output JSON file")

    def diff_jsons(args: dict):
        with open(args.input_json1, "r") as f:
            tr1 = TestRun.from_json(f.read())
        with open(args.input_json2, "r") as f:
            tr2 = TestRun.from_json(f.read())
        td = TestDifferences(
            tr1,
            tr2,
            show_diffs=not args.hide_diffs,
            inclusion_criterion=args.kernel_inclusion_criterion,
        )

        if len(td.test_diffs) == 0:
            print("No differences found in overlapping tests!")
        else:
            print(
                td.total_num_diffs,
                "kernel differences from",
                len(td.test_diffs),
                "tests found",
            )
        if len(td.new_tests) > 0:
            print(len(td.new_tests), "new tests found")
        if len(td.removed_tests) > 0:
            print(len(td.removed_tests), "removed tests found")

        with open(args.output_json, "w") as f:
            f.write(td.to_json())

        # Return 1 if preamble or any kernels are changed, else 0
        exit(1 if len(td.test_diffs) > 0 or len(td.preamble_diff) > 0 else 0)

    diff_parser.set_defaults(func=diff_jsons)

    report_parser = subparsers.add_parser(
        "diff_report", help="Compute an HTML report from a diff JSON"
    )
    report_parser.add_argument(
        "--hide-env",
        action="store_true",
        help="Hide environment variables and nvcc versions in output?",
    )
    report_parser.add_argument(
        "--max-diffs",
        default=200,
        type=int,
        help="Limit number of included kernel diffs in HTML output to this many (does not affect exit code).",
    )
    report_parser.add_argument(
        "--omit-preamble",
        action="store_true",
        help="Omit the preamble in HTML output?",
    )
    report_parser.add_argument("input_json", help="Location of diff JSON file")
    report_parser.add_argument("output_html", help="Location to write output HTML file")

    def diff_report(args: dict):
        with open(args.input_json, "r") as f:
            td = TestDifferences.from_json(f.read())

        if args.hide_env:
            td.hide_env()

        with open(args.output_html, "w") as f:
            f.write(
                td.generate_html(
                    omit_preamble=args.omit_preamble, max_diffs=args.max_diffs
                )
            )

    report_parser.set_defaults(func=diff_report)

    args = parser.parse_args()

    args.func(args)
