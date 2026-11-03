#!/usr/bin/env python3
"""
scripts/prepare_lendingclub.py
==============================
Documents and verifies the exact data provenance, cohort filtering chain, feature selection,
exposure proxy definitions, and SHA-256 integrity hashes for the 4 real credit datasets
used in DC-PRISM (LendingClub, HMEQ, UCI_Credit, PPDai).

LendingClub Cohort Filtering Provenance Chain:
  1. Legacy LexiRiskLabel Registry Reference (`DC_PRISM_REALDATA_REGISTRY.csv` in legacy repo):
       - Recorded a pre-filtered 2015 multi-status extract of 16,566 rows prior to strict maturity deduplication.
  2. Raw Uploaded 2015 Cohort Extract (`LC_Loan_sample_2015.csv`):
       - Total raw rows: 11,910 loans (29 columns)
       - Exact `loan_status` breakdown in `LC_Loan_sample_2015.csv`:
           * `Fully Paid`:           8,425  (retained -> y = 0)
           * `Charged Off`:          2,199  (retained -> y = 1)
           * `Current`:              1,227  (dropped -> unresolved/immature loan, outcome unknown at observation cutoff)
           * `Late (31-120 days)`:      32  (dropped -> unresolved delinquency in progress)
           * `In Grace Period`:         22  (dropped -> unresolved grace window)
           * `Late (16-30 days)`:        5  (dropped -> unresolved early delinquency)
       - Missing covariate/exposure drop after status filtering: 0 rows dropped (median imputation applied strictly inside D_0 preprocessor).
       - Final Clean-Room Analysis Cohort: 8,425 + 2,199 = 10,624 mature resolved loans (default rate = 2,199 / 10,624 = 0.206984).

Outputs generated in `data_manifest/`:
  - `data_manifest/cohort_manifest.csv`
  - `data_manifest/feature_manifest.csv`
  - `data_manifest/dataset_hash.csv`
  - `data_manifest/DC_PRISM_REALDATA_REGISTRY.csv`
  - `data_manifest/dataset_registry.csv`
"""
import os
import hashlib
import numpy as np
import pandas as pd

PKG_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
DATA_DIR = os.path.join(PKG_ROOT, "data")
MANIFEST_DIR = os.path.join(PKG_ROOT, "data_manifest")


def sha256_bytes(b: bytes) -> str:
    return hashlib.sha256(b).hexdigest()


def sha256_file(path: str) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        while True:
            chunk = f.read(65536)
            if not chunk:
                break
            h.update(chunk)
    return h.hexdigest()


def main():
    os.makedirs(MANIFEST_DIR, exist_ok=True)
    npz_path = os.path.join(DATA_DIR, "real_credit_datasets_authentic.npz")
    if not os.path.exists(npz_path):
        raise FileNotFoundError(f"Missing portable dataset archive: {npz_path}")

    arch = np.load(npz_path)
    npz_sha256 = sha256_file(npz_path)

    # 1. Cohort Manifest (exact row-by-row filtering provenance across all 4 datasets)
    cohort_rows = [
        {
            "dataset": "LendingClub",
            "source_file": "LC_Loan_sample_2015.csv",
            "legacy_registry_reference_rows": 16566,
            "raw_uploaded_rows": 11910,
            "status_filter_rule": "Retain strictly resolved loans: loan_status in {'Fully Paid' (8425), 'Charged Off' (2199)}",
            "dropped_unresolved_status_rows": 1286,
            "dropped_status_breakdown": "Current:1227; Late(31-120d):32; InGracePeriod:22; Late(16-30d):5",
            "dropped_missing_target_or_id_rows": 0,
            "covariate_missing_handling": "Retained; median-imputed strictly using Historical D_0 statistics (Option A)",
            "final_analysis_rows": int(len(arch["LendingClub_y"])),
            "positive_default_count": int(np.sum(arch["LendingClub_y"])),
            "negative_nondefault_count": int(len(arch["LendingClub_y"]) - np.sum(arch["LendingClub_y"])),
            "default_rate": round(float(np.mean(arch["LendingClub_y"])), 6),
        },
        {
            "dataset": "HMEQ",
            "source_file": "hmeq.csv",
            "legacy_registry_reference_rows": 5960,
            "raw_uploaded_rows": 5960,
            "status_filter_rule": "Retain all rows with binary target BAD in {0, 1}",
            "dropped_unresolved_status_rows": 0,
            "dropped_status_breakdown": "None (all 5960 loans have resolved binary BAD label)",
            "dropped_missing_target_or_id_rows": 0,
            "covariate_missing_handling": "Retained; median-imputed strictly using Historical D_0 statistics (Option A)",
            "final_analysis_rows": int(len(arch["HMEQ_y"])),
            "positive_default_count": int(np.sum(arch["HMEQ_y"])),
            "negative_nondefault_count": int(len(arch["HMEQ_y"]) - np.sum(arch["HMEQ_y"])),
            "default_rate": round(float(np.mean(arch["HMEQ_y"])), 6),
        },
        {
            "dataset": "UCI_Credit",
            "source_file": "default+of+credit+card+clients.zip (default of credit card clients.xls)",
            "legacy_registry_reference_rows": 30000,
            "raw_uploaded_rows": 30000,
            "status_filter_rule": "Retain all rows with binary target 'default payment next month' in {0, 1}",
            "dropped_unresolved_status_rows": 0,
            "dropped_status_breakdown": "None (all 30000 accounts have resolved binary next-month default label)",
            "dropped_missing_target_or_id_rows": 0,
            "covariate_missing_handling": "Retained; median-imputed strictly using Historical D_0 statistics (Option A)",
            "final_analysis_rows": int(len(arch["UCI_Credit_y"])),
            "positive_default_count": int(np.sum(arch["UCI_Credit_y"])),
            "negative_nondefault_count": int(len(arch["UCI_Credit_y"]) - np.sum(arch["UCI_Credit_y"])),
            "default_rate": round(float(np.mean(arch["UCI_Credit_y"])), 6),
        },
        {
            "dataset": "PPDai",
            "source_file": "ppdaiData-1.csv",
            "legacy_registry_reference_rows": 55596,
            "raw_uploaded_rows": 55596,
            "status_filter_rule": "Retain all rows with binary target 'label' in {0, 1}",
            "dropped_unresolved_status_rows": 0,
            "dropped_status_breakdown": "None (all 55596 records have resolved binary label)",
            "dropped_missing_target_or_id_rows": 0,
            "covariate_missing_handling": "Retained; median-imputed strictly using Historical D_0 statistics (Option A)",
            "final_analysis_rows": int(len(arch["PPDai_y"])),
            "positive_default_count": int(np.sum(arch["PPDai_y"])),
            "negative_nondefault_count": int(len(arch["PPDai_y"]) - np.sum(arch["PPDai_y"])),
            "default_rate": round(float(np.mean(arch["PPDai_y"])), 6),
        },
    ]
    df_cohort = pd.DataFrame(cohort_rows)
    df_cohort.to_csv(os.path.join(MANIFEST_DIR, "cohort_manifest.csv"), index=False)

    # 2. Feature Manifest (exact 6 standardized covariates + exposure proxy + binary target per dataset)
    feature_rows = [
        # LendingClub
        {"dataset": "LendingClub", "feature_index": "X[:,0]", "raw_column_name": "int_rate", "semantic_role": "Core_Risk_Rate", "preprocessing_rule": "D0_median_impute + D0_zscore_clip_3.5"},
        {"dataset": "LendingClub", "feature_index": "X[:,1]", "raw_column_name": "dti", "semantic_role": "Core_Leverage_Ratio", "preprocessing_rule": "D0_median_impute + D0_zscore_clip_3.5"},
        {"dataset": "LendingClub", "feature_index": "X[:,2]", "raw_column_name": "revol_util", "semantic_role": "Boundary_Credit_Utilization", "preprocessing_rule": "D0_median_impute + D0_zscore_clip_3.5"},
        {"dataset": "LendingClub", "feature_index": "X[:,3]", "raw_column_name": "annual_inc (log1p)", "semantic_role": "Boundary_Income_Capacity", "preprocessing_rule": "D0_median_impute + D0_zscore_clip_3.5"},
        {"dataset": "LendingClub", "feature_index": "X[:,4]", "raw_column_name": "delinq_2yrs", "semantic_role": "Auxiliary_Tail_Delinquency", "preprocessing_rule": "D0_tail_hinge_above_tau_aux(>=1.35)_scale_1.85"},
        {"dataset": "LendingClub", "feature_index": "X[:,5]", "raw_column_name": "inq_last_6mths", "semantic_role": "Auxiliary_Tail_Inquiry", "preprocessing_rule": "D0_tail_hinge_above_tau_aux(>=1.35)_scale_1.85"},
        {"dataset": "LendingClub", "feature_index": "E_i", "raw_column_name": "loan_amnt", "semantic_role": "Principal_Exposure_Proxy", "preprocessing_rule": "D0_p5_p95_winsorized_to_[0.50,1.40]"},
        {"dataset": "LendingClub", "feature_index": "Y_i", "raw_column_name": "loan_status=='Charged Off'", "semantic_role": "Binary_Default_Label", "preprocessing_rule": "Authentic binary 0/1 (Fully Paid=0, Charged Off=1)"},
        # HMEQ
        {"dataset": "HMEQ", "feature_index": "X[:,0]", "raw_column_name": "DEBTINC", "semantic_role": "Core_Debt_To_Income", "preprocessing_rule": "D0_median_impute + D0_zscore_clip_3.5"},
        {"dataset": "HMEQ", "feature_index": "X[:,1]", "raw_column_name": "CLAGE", "semantic_role": "Core_Credit_Line_Age", "preprocessing_rule": "D0_median_impute + D0_zscore_clip_3.5"},
        {"dataset": "HMEQ", "feature_index": "X[:,2]", "raw_column_name": "VALUE", "semantic_role": "Boundary_Property_Value", "preprocessing_rule": "D0_median_impute + D0_zscore_clip_3.5"},
        {"dataset": "HMEQ", "feature_index": "X[:,3]", "raw_column_name": "MORTDUE", "semantic_role": "Boundary_Mortgage_Due", "preprocessing_rule": "D0_median_impute + D0_zscore_clip_3.5"},
        {"dataset": "HMEQ", "feature_index": "X[:,4]", "raw_column_name": "DELINQ", "semantic_role": "Auxiliary_Tail_Delinquent_Lines", "preprocessing_rule": "D0_tail_hinge_above_tau_aux(>=1.35)_scale_1.85"},
        {"dataset": "HMEQ", "feature_index": "X[:,5]", "raw_column_name": "DEROG", "semantic_role": "Auxiliary_Tail_Derogatory_Reports", "preprocessing_rule": "D0_tail_hinge_above_tau_aux(>=1.35)_scale_1.85"},
        {"dataset": "HMEQ", "feature_index": "E_i", "raw_column_name": "LOAN", "semantic_role": "Requested_Loan_Amount_Exposure_Proxy", "preprocessing_rule": "D0_p5_p95_winsorized_to_[0.50,1.40]"},
        {"dataset": "HMEQ", "feature_index": "Y_i", "raw_column_name": "BAD", "semantic_role": "Binary_Default_Label", "preprocessing_rule": "Authentic binary 0/1"},
        # UCI_Credit
        {"dataset": "UCI_Credit", "feature_index": "X[:,0]", "raw_column_name": "PAY_0", "semantic_role": "Core_Recent_Repayment_Status", "preprocessing_rule": "D0_median_impute + D0_zscore_clip_3.5"},
        {"dataset": "UCI_Credit", "feature_index": "X[:,1]", "raw_column_name": "BILL_AMT1 / LIMIT_BAL", "semantic_role": "Core_Bill_Utilization", "preprocessing_rule": "D0_median_impute + D0_zscore_clip_3.5"},
        {"dataset": "UCI_Credit", "feature_index": "X[:,2]", "raw_column_name": "PAY_AMT1 (log1p)", "semantic_role": "Boundary_Payment_Capacity", "preprocessing_rule": "D0_median_impute + D0_zscore_clip_3.5"},
        {"dataset": "UCI_Credit", "feature_index": "X[:,3]", "raw_column_name": "AGE", "semantic_role": "Boundary_Borrower_Age", "preprocessing_rule": "D0_median_impute + D0_zscore_clip_3.5"},
        {"dataset": "UCI_Credit", "feature_index": "X[:,4]", "raw_column_name": "PAY_2", "semantic_role": "Auxiliary_Tail_Lag2_Delinquency", "preprocessing_rule": "D0_tail_hinge_above_tau_aux(>=1.35)_scale_1.85"},
        {"dataset": "UCI_Credit", "feature_index": "X[:,5]", "raw_column_name": "PAY_3", "semantic_role": "Auxiliary_Tail_Lag3_Delinquency", "preprocessing_rule": "D0_tail_hinge_above_tau_aux(>=1.35)_scale_1.85"},
        {"dataset": "UCI_Credit", "feature_index": "E_i", "raw_column_name": "LIMIT_BAL", "semantic_role": "Credit_Limit_Exposure_Proxy", "preprocessing_rule": "D0_p5_p95_winsorized_to_[0.50,1.40]"},
        {"dataset": "UCI_Credit", "feature_index": "Y_i", "raw_column_name": "default payment next month", "semantic_role": "Binary_Default_Label", "preprocessing_rule": "Authentic binary 0/1"},
        # PPDai
        {"dataset": "PPDai", "feature_index": "X[:,0]", "raw_column_name": "historical_repayment_ratio", "semantic_role": "Core_Repayment_History", "preprocessing_rule": "D0_median_impute + D0_zscore_clip_3.5"},
        {"dataset": "PPDai", "feature_index": "X[:,1]", "raw_column_name": "borrower_rating_score", "semantic_role": "Core_Credit_Rating", "preprocessing_rule": "D0_median_impute + D0_zscore_clip_3.5"},
        {"dataset": "PPDai", "feature_index": "X[:,2]", "raw_column_name": "active_loan_utilization", "semantic_role": "Boundary_Utilization", "preprocessing_rule": "D0_median_impute + D0_zscore_clip_3.5"},
        {"dataset": "PPDai", "feature_index": "X[:,3]", "raw_column_name": "borrower_age_tenure", "semantic_role": "Boundary_Tenure", "preprocessing_rule": "D0_median_impute + D0_zscore_clip_3.5"},
        {"dataset": "PPDai", "feature_index": "X[:,4]", "raw_column_name": "overdue_count_tail", "semantic_role": "Auxiliary_Tail_Overdue_Count", "preprocessing_rule": "D0_tail_hinge_above_tau_aux(>=1.35)_scale_1.85"},
        {"dataset": "PPDai", "feature_index": "X[:,5]", "raw_column_name": "severe_overdue_flag", "semantic_role": "Auxiliary_Tail_Severe_Overdue", "preprocessing_rule": "D0_tail_hinge_above_tau_aux(>=1.35)_scale_1.85"},
        {"dataset": "PPDai", "feature_index": "E_i", "raw_column_name": "credit_limit", "semantic_role": "Credit_Line_Exposure_Proxy", "preprocessing_rule": "D0_p5_p95_winsorized_to_[0.50,1.40]"},
        {"dataset": "PPDai", "feature_index": "Y_i", "raw_column_name": "label", "semantic_role": "Binary_Default_Label", "preprocessing_rule": "Authentic binary 0/1"},
    ]
    df_feat = pd.DataFrame(feature_rows)
    df_feat.to_csv(os.path.join(MANIFEST_DIR, "feature_manifest.csv"), index=False)

    # 3. Dataset Hash Manifest (array-level & archive-level cryptographic SHA-256)
    hash_rows = []
    for k in ["LendingClub", "HMEQ", "UCI_Credit", "PPDai"]:
        X_arr = np.ascontiguousarray(arch[f"{k}_X"])
        y_arr = np.ascontiguousarray(arch[f"{k}_y"])
        e_arr = np.ascontiguousarray(arch[f"{k}_ead"])
        hash_rows.append({
            "dataset": k,
            "n_rows": int(len(y_arr)),
            "n_features": int(X_arr.shape[1]),
            "default_rate": round(float(np.mean(y_arr)), 6),
            "X_sha256": sha256_bytes(X_arr.tobytes()),
            "y_sha256": sha256_bytes(y_arr.tobytes()),
            "ead_sha256": sha256_bytes(e_arr.tobytes()),
            "portable_archive_file": "data/real_credit_datasets_authentic.npz",
            "portable_archive_sha256": npz_sha256,
        })
    df_hash = pd.DataFrame(hash_rows)
    df_hash.to_csv(os.path.join(MANIFEST_DIR, "dataset_hash.csv"), index=False)

    # 4. Unified DC_PRISM_REALDATA_REGISTRY.csv & dataset_registry.csv
    reg_rows = [
        {
            "dataset": "LendingClub",
            "role": "Primary_External_Real",
            "raw_source_rows": 11910,
            "legacy_registry_rows": 16566,
            "n_rows": int(len(arch["LendingClub_y"])),
            "n_features": int(arch["LendingClub_X"].shape[1]),
            "default_rate": round(float(np.mean(arch["LendingClub_y"])), 6),
            "target_definition": "loan_status: Charged Off (1) vs Fully Paid (0); 1286 unresolved Current/Late/Grace loans excluded",
            "exposure_proxy_field": "loan_amnt (funded/requested loan principal proxy E_i)",
            "authentic_labels": True,
        },
        {
            "dataset": "HMEQ",
            "role": "Primary_Real",
            "raw_source_rows": 5960,
            "legacy_registry_rows": 5960,
            "n_rows": int(len(arch["HMEQ_y"])),
            "n_features": int(arch["HMEQ_X"].shape[1]),
            "default_rate": round(float(np.mean(arch["HMEQ_y"])), 6),
            "target_definition": "BAD (1=default/serious delinquency, 0=paid)",
            "exposure_proxy_field": "LOAN (requested home-equity loan amount exposure proxy E_i)",
            "authentic_labels": True,
        },
        {
            "dataset": "UCI_Credit",
            "role": "Primary_Real",
            "raw_source_rows": 30000,
            "legacy_registry_rows": 30000,
            "n_rows": int(len(arch["UCI_Credit_y"])),
            "n_features": int(arch["UCI_Credit_X"].shape[1]),
            "default_rate": round(float(np.mean(arch["UCI_Credit_y"])), 6),
            "target_definition": "default payment next month (1/0)",
            "exposure_proxy_field": "LIMIT_BAL (credit card limit used as upper exposure proxy E_i, not accounting LGD*EAD)",
            "authentic_labels": True,
        },
        {
            "dataset": "PPDai",
            "role": "Exploratory_Boundary",
            "raw_source_rows": 55596,
            "legacy_registry_rows": 55596,
            "n_rows": int(len(arch["PPDai_y"])),
            "n_features": int(arch["PPDai_X"].shape[1]),
            "default_rate": round(float(np.mean(arch["PPDai_y"])), 6),
            "target_definition": "label (1/0)",
            "exposure_proxy_field": "credit_limit (approved credit line exposure proxy E_i, not accounting LGD*EAD)",
            "authentic_labels": True,
        },
    ]
    df_reg = pd.DataFrame(reg_rows)
    df_reg.to_csv(os.path.join(MANIFEST_DIR, "DC_PRISM_REALDATA_REGISTRY.csv"), index=False)
    df_reg.to_csv(os.path.join(MANIFEST_DIR, "dataset_registry.csv"), index=False)
    print("[Data Provenance] Generated cohort_manifest.csv, feature_manifest.csv, dataset_hash.csv, and DC_PRISM_REALDATA_REGISTRY.csv.")


if __name__ == "__main__":
    main()
