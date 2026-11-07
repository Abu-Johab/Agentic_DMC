import os
import json
import unittest
import numpy as np

# Adversarial Retest Suite: Ensures Repaired Gates REJECT Deliberately Tampered Evidence
WORKSPACE_ROOT = os.path.normpath(os.path.dirname(__file__))

def test_adversarial_tampering():
    print("=== STARTING ADVERSARIAL RETEST SUITE ===")
    results = []
    v2_dir = os.path.join(WORKSPACE_ROOT, "agentic_dmc", "outputs", "verified_final", "run_20261010_v2")

    # 1. Adversarial Attack 1: Corrupted Dataset CSV Path in Test 01
    print("\n--- Attack 1: Corrupted Dataset Link ---")
    orig_ds_path = os.path.join(WORKSPACE_ROOT, "SpectroFood_dataset.csv")
    backup_ds_path = os.path.join(WORKSPACE_ROOT, "SpectroFood_dataset.csv.bak")

    try:
        os.rename(orig_ds_path, backup_ds_path)
        suite = unittest.TestLoader().loadTestsFromName("agentic_dmc.tests.test_gate_verification.TestCompletionGateVerification.test_01_protection_manifest_hash_check")
        test_res = unittest.TextTestRunner(verbosity=0).run(suite)
        if not test_res.wasSuccessful():
            print("  [SUCCESS] Repaired Gate REJECTED missing dataset CSV file as expected!")
            results.append(("Attack 1: Missing Dataset CSV", "PASSED_REJECTION"))
        else:
            print("  [FAILURE] Repaired Gate accepted missing dataset CSV!")
            results.append(("Attack 1: Missing Dataset CSV", "FAILED_ACCEPTED"))
    finally:
        if os.path.exists(backup_ds_path):
            os.rename(backup_ds_path, orig_ds_path)

    # 2. Adversarial Attack 2: Corrupted Master Registry Metric (True Nested CV RMSE = 0.2785 g/g)
    print("\n--- Attack 2: Corrupted Nested CV Registry Metric ---")
    nested_path = os.path.join(v2_dir, "exp2_nested_cv_all_models.json")
    with open(nested_path, "r", encoding="utf-8") as f:
        orig_nested = json.load(f)

    tampered_nested = json.loads(json.dumps(orig_nested))
    tampered_nested["single_5x3_nested_cv_metrics"]["SVR"]["rmse"] = 0.9999 # Deliberately false RMSE

    try:
        with open(nested_path, "w", encoding="utf-8") as f:
            json.dump(tampered_nested, f, indent=2)
        suite = unittest.TestLoader().loadTestsFromName("agentic_dmc.tests.test_gate_verification.TestCompletionGateVerification.test_03_metric_disambiguation")
        test_res = unittest.TextTestRunner(verbosity=0).run(suite)
        if not test_res.wasSuccessful():
            print("  [SUCCESS] Repaired Gate REJECTED corrupted SVR nested CV metric as expected!")
            results.append(("Attack 2: Corrupted Nested Metric", "PASSED_REJECTION"))
        else:
            print("  [FAILURE] Repaired Gate accepted corrupted nested metric!")
            results.append(("Attack 2: Corrupted Nested Metric", "FAILED_ACCEPTED"))
    finally:
        with open(nested_path, "w", encoding="utf-8") as f:
            json.dump(orig_nested, f, indent=2)

    # 3. Adversarial Attack 3: Tampered Spatial Difference Records
    print("\n--- Attack 3: Tampered Spatial Difference Records ---")
    sp_path = os.path.join(v2_dir, "exp4_spatial_37_sample_audit.json")
    with open(sp_path, "r", encoding="utf-8") as f:
        orig_sp = json.load(f)

    tampered_sp = json.loads(json.dumps(orig_sp))
    tampered_sp["mean_diff_C_minus_B_aggregation"] = 0.0010 # Tampered mean aggregation shift (<0.05)

    try:
        with open(sp_path, "w", encoding="utf-8") as f:
            json.dump(tampered_sp, f, indent=2)
        suite = unittest.TestLoader().loadTestsFromName("agentic_dmc.tests.test_gate_verification.TestCompletionGateVerification.test_04_spatial_stage_consistency")
        test_res = unittest.TextTestRunner(verbosity=0).run(suite)
        if not test_res.wasSuccessful():
            print("  [SUCCESS] Repaired Gate REJECTED tampered spatial aggregation shift as expected!")
            results.append(("Attack 3: Tampered Spatial Shift", "PASSED_REJECTION"))
        else:
            print("  [FAILURE] Repaired Gate accepted tampered spatial shift!")
            results.append(("Attack 3: Tampered Spatial Shift", "FAILED_ACCEPTED"))
    finally:
        with open(sp_path, "w", encoding="utf-8") as f:
            json.dump(orig_sp, f, indent=2)

    # 4. Adversarial Attack 4: Tampered SHAP Rank Stability
    print("\n--- Attack 4: Tampered SHAP Rank Stability ---")
    shap_path = os.path.join(v2_dir, "shap_corrections_audit.json")
    with open(shap_path, "r", encoding="utf-8") as f:
        orig_shap = json.load(f)

    tampered_shap = json.loads(json.dumps(orig_shap))
    tampered_shap["five_seed_shap_spearman_rank_stability_mean"] = "INVALID_RHO" # Corrupted type

    try:
        with open(shap_path, "w", encoding="utf-8") as f:
            json.dump(tampered_shap, f, indent=2)
        suite = unittest.TestLoader().loadTestsFromName("agentic_dmc.tests.test_gate_verification.TestCompletionGateVerification.test_05_shap_residual_diagnostics")
        test_res = unittest.TextTestRunner(verbosity=0).run(suite)
        if not test_res.wasSuccessful():
            print("  [SUCCESS] Repaired Gate REJECTED corrupted SHAP rank stability value as expected!")
            results.append(("Attack 4: Corrupted SHAP Value", "PASSED_REJECTION"))
        else:
            print("  [FAILURE] Repaired Gate accepted corrupted SHAP value!")
            results.append(("Attack 4: Corrupted SHAP Value", "FAILED_ACCEPTED"))
    finally:
        with open(shap_path, "w", encoding="utf-8") as f:
            json.dump(orig_shap, f, indent=2)

    # Summary
    print("\n=== ADVERSARIAL RETEST SUMMARY ===")
    all_passed = True
    for name, status in results:
        print(f"  {name:40s}: {status}")
        if status != "PASSED_REJECTION":
            all_passed = False

    print(f"\nOverall Adversarial Retest Status: {'ALL_ATTACKS_REJECTED_SUCCESSFULLY' if all_passed else 'SOME_ATTACKS_ACCEPTED'}")
    return all_passed

if __name__ == "__main__":
    test_adversarial_tampering()
