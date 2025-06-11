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

void loadMultiSourceFusionOperatorBindings(::nanobind::module_& bound_module);
void loadMultiSourceConflictOperatorBindings(::nanobind::module_& bound_module);
void loadMultiSourceTrustRevisionOperatorBindings(::nanobind::module_& bound_module);
void loadMultiSourceTrustedFusionOperatorBindings(::nanobind::module_& bound_module);
