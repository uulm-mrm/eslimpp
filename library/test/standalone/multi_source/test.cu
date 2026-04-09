


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
  check_run_prior_average();
  check_run_preprocess_uncertainties();
  check_run_fuse_opinions();

  // Conflict
  check_run_uncert_differentials();
  check_run_conflict_shares();

  // Revision factors
  check_run_revsion_factors();

  // Trusted fusion
  check_run_trusted_fusion();

  return 0;
}
