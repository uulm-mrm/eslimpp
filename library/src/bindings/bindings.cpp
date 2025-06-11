#include <nanobind/nanobind.h>

#include <string>

#include "bindings.hpp"

namespace nb = nanobind;

NB_MODULE(_subjective_logic_lib_python_api, m)
{
  loadCudaCompatibleArrayBindings(m);
  loadDirichletDistributionBindings(m);
  loadOpinionBindings(m);
  loadOpinionNoBaseBindings(m);
  loadTrustedOpinionBindings(m);
  loadMultiSourceFusionOperatorBindings(m);
  loadMultiSourceConflictOperatorBindings(m);
  loadMultiSourceTrustRevisionOperatorBindings(m);
  loadMultiSourceTrustedFusionOperatorBindings(m);
}