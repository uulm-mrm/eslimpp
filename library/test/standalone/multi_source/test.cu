
#include <iostream>

// Fusion
void check_run_prior_average();
void check_run_preprocess_uncertainties();
void check_run_fuse_opinions();

// Conflict
void check_run_uncert_differentials();
void check_run_conflict_shares();

// Revision factors
void check_run_revsion_factors();

// Trusted fusion
void check_run_trusted_fusion();

int main(int argc, char **argv) {

  // Fusion
  std::cout << "run fusion tests" << std::endl;
  check_run_prior_average();
  check_run_preprocess_uncertainties();
  check_run_fuse_opinions();

  std::cout << "\n\n";

  // Conflict
  std::cout << "run conflict tests" << std::endl;
  check_run_uncert_differentials();
  check_run_conflict_shares();

  std::cout << "\n\n";

  // Revision factors
  std::cout << "run revision factor tests" << std::endl;
  check_run_revsion_factors();

  std::cout << "\n\n";

  // Trusted fusion
  std::cout << "run trust revision tests" << std::endl;
  check_run_trusted_fusion();

  std::cout << "\n\n";

  return 0;
}
