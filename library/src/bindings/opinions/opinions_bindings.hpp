#include <nanobind/nanobind.h>
#include <nanobind/operators.h>
#include <nanobind/stl/string.h>
#include <nanobind/stl/pair.h>
#include <nanobind/stl/tuple.h>
#include <nanobind/stl/array.h>
#include <nanobind/stl/vector.h>
#include <nanobind/stl/optional.h>

#include <string>

#include "../template_combination_helper.hpp"

void loadOpinionBindings(::nanobind::module_& bound_module);
void loadOpinionNoBaseBindings(::nanobind::module_& bound_module);
void loadTrustedOpinionBindings(::nanobind::module_& bound_module);
