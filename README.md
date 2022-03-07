# spla <img align="right" width="20%" src="https://github.com/JetBrains-Research/spla/raw/main/docs/logos/spla-logo.png?raw=true&sanitize=true">

[![JB Research](https://jb.gg/badges/research-flat-square.svg)](https://research.jetbrains.org/)
[![Build](https://github.com/JetBrains-Research/spla/actions/workflows/build.yml/badge.svg?branch=main)](https://github.com/JetBrains-Research/spla/actions/workflows/build.yml)
[![Pages](https://github.com/JetBrains-Research/spla/actions/workflows/docs.yml/badge.svg?branch=main)](https://jetbrains-research.github.io/spla/)
[![Clang Format](https://github.com/JetBrains-Research/spla/actions/workflows/clang-format.yml/badge.svg?branch=main)](https://github.com/JetBrains-Research/spla/actions/workflows/clang-format.yml)
[![License](https://img.shields.io/badge/license-MIT-blue)](https://github.com/JetBrains-Research/spla/blob/master/LICENSE.md)

**spla** is a sparse linear algebra framework for multi-GPU computations based on `OpenCL` and `Cuda`. It provides 
linear algebra primitives, such as matrices, vectors and scalars, supports wide variety of operations, and gives an 
ability to customize underlying values types and parametrise operations using arbitrary user defined functions.

> Note: project under heavy development! Not ready for usage.

### Platforms

- Windows 10
- Linux (Ubuntu 20.04)
- MacOS (Catalina 10.15)

## Building from sources

### Prerequisites

- Common:
    - Git (to get source code)
    - CMake (the latest version)
    - Ninja (as build files generator)
    - OpenCL 1.2+ SDK
    - Cuda SDK
    - Python 3.7+
- Windows 10:
    - Microsoft Visual C++ Compiler (MSVC) with C++ 17 support
    - x64 Native Tools Command Prompt for VS
- Ubuntu 20.04:
    - GNU C++ Compiler with C++ 17 support
- MaсOS Catalina 10.15:
    - Clang Compiler with C++ 17 support

### Get source code

The following code snippet downloads project source code repository, enters project root folder and runs submodules init
in order to get dependencies source code initialized. Must be executed from the folder where you want to locate project.

```shell
$ git clone https://github.com/JetBrains-Research/spla.git
$ cd spla
$ git submodule update --init --recursive
```

### Configure and run build

The following code snippet runs cmake build configuration process with output into `build` directory, in `Release` mode
with tests `SPLA_BUILD_TESTS=ON` enabled. Then runs build process for `build` directory in verbose mode with `-j 4` four
system threads. Pass optionally `SPLA_BUILD_EXAMPLES=OFF` to disable build of library example applications. Must be
executed from project root folder.

```shell
$ cmake . -B build -G Ninja -DCMAKE_BUILD_TYPE=Release -DSPLA_BUILD_TESTS=ON
$ cmake --build build --target all --verbose -j 4
```

On macOS, you can optionally specify target binaries architecture to build. Pass option `-DCMAKE_OSX_ARCHITECTURES`
with `x86_64` or `arm64` respectively. By default, build falls back to `CMAKE_SYSTEM_PROCESSOR` specified architecture.
See example bellow, replace `<arch>` with desired architecture for your build.

```shell
$ cmake . -B build -G Ninja -DCMAKE_BUILD_TYPE=Release -DSPLA_BUILD_TESTS=ON -DCMAKE_OSX_ARCHITECTURES=<arch>
$ cmake --build build --target all --verbose -j 4
```

### Run unit-tests

The following code snippet executed python script, which allows to run all native C++ library unit-tests, located in
build directory, specified in `--build-dir=build`. Must be executed from project root folder.

```shell
python ./scripts/run_tests.py --build-dir=build
```

## Directory structure

```
spla
├── .github - GitHub Actions CI/CD setup 
├── docs - documents, text files and various helpful stuff
├── scripts - short utility programs 
├── spla - library source files (header-only)
│   ├── backend - library implementation for a specific platform
│   │   ├── reference - naive host backend implemenatation
│   │   ├── opencl - opencl based library backend implemenatation
│   │   └── cuda - cuda based library backend implemenatation
│   ├── algorithms - library based common liinear algebra algoruthms
│   ├── detail - internal classes
│   ├── expression - computational expressions implementation
│   ├── io - input/output operations for data manipulation
│   ├── storage - liberary storage classes
│   └── spla.hpp - all in one place library interface
├── examples - example applications with library usage
│   ├── bfs - bfs application
│   ├── sssp - sssp application
│   └── tc - tc application
├── tests - gtest-based unit-tests collection
├── package - python-package files
│   ├── pyspla - library python wrapper source code
│   └── tests - python package regression tests   
├── deps - project dependencies
│   ├── cxxopts - lightweight C++ command line option parser
│   ├── gtest - google test framework for unit testing
│   └── taskflow - taskflow library for parallel tasks execution
└── CMakeLists.txt - library cmake config, add this as sub-directory to your project
```

## Contributors

- Egor Orachyov (Github: [@EgorOrachyov](https://github.com/EgorOrachyov))
- Gleb Marin (Github: [@Glebanister](https://github.com/Glebanister))
- Semyon Grigorev (Github: [@gsvgit](https://github.com/gsvgit))

## Citation

```ignorelang
@online{spla,
  author = {Orachyov, Egor and Marin, Gleb and Grigorev, Semyon},
  title = {spla: Generalized sparse linear algebra framework for multi-GPU computations},
  year = 2021,
  url = {https://github.com/JetBrains-Research/spla},
  note = {Version 0.0.0}
}
```

## License

This project licensed under MIT License. License text can be found in the
[license file](https://github.com/JetBrains-Research/spla/blob/master/LICENSE.md).

## Acknowledgments <img align="right" width="15%" src="https://github.com/JetBrains-Research/spla/raw/main/docs/logos/jetbrains-logo.png?raw=true&sanitize=true">

This is a research project of the Programming Languages and Tools Laboratory at JetBrains-Research. Laboratory
website [link](https://research.jetbrains.org/groups/plt_lab/).
