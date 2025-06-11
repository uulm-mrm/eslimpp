# eSLIM++ Evaluation
The scripts available in `publication/eSLIM++` are reproducing the evaluation results of the accompanied publication or provide even more detailed information.

Running Python evaluations is similar to all python evaluation, see [publications/README.md](../README.md)

For installing and running the C++ based grid_assessment evaluation the library and evaluation must be build fist.\
This can be done by the following:

```bash
  # building and installing the c++ library
  cd [path_to_eSLIM++_root]/library
  mkdir build
  cd build
  cmake ..
  make install # header only should be done in no time
  
  # building and installing the evaluation executable
  cd [path_to_eSLIM++_root]/publications/eSLIM++/grid
  mkdir build
  cd build
  cmake ..
  make -j8
```

Now the evaluation executable is available within the later build directory and can be called with
```bash
    ./[path_to_eSLIM++_root]/publications/eSLIM++/grid/build/grid_assessment [NUMBER OF CELLS > 5000] [NUMBER OF RUNS]
```
## configuring the grid_assessment evaluation
The grid_assessment evaluation further allows some configuration.\
By default, the CUDA evaluation is build and added to the eval script if available.\
To deactivate the use of CUDA run cmake with
```bash
  cmake .. -DUSE_CUDA=OFF
```
Additionally, if the DST evaluation should be added to the script run
```bash
  cmake .. -DRUN_DST
```
It shall be mentioned that, at the of writing, the use of the dst library was not directly streight forward.
How certain (fusion)operators are meant to be applied was not fully apparent to the author.
Respectively, the DST evaluation uses one specific operator for fusion, and the categorization is deactivated, since the author was not aware of how to access the belief_mass of a certain set.
That being said, the DST Evaluation is available, but take a little longer; for more runs (>10) the eval execution takes some time.
