This directory provides scripts and tests to check TVM with future versions of ROCm releases.

General testing procedure is to create or download a ROCm docker container with components
under test (see ../dockerfile directory) and then run scripts from this directory.

These tests include:
   run_bench.sh - run one benchmark to make sure we get results
   run_quickstart.sh - run the overall quickstart example to make sure we get results.
   run_rocm_unittest.sh - run unit tests that exercise ROCm