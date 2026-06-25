# Phase 3 FULL Sweep — Byte-Equality Report

Sweep root: results/swe_percase_threshold_full_20260624T161532Z
Cells (threshold × chunk): 24
Total (case × threshold) patches: 60
Unique cases: 10

## Check 1: Within-cell byte-equality (placeholder_knn_lossy vs in-cell lossy)
Equal: 54/60
Not equal: 6/60

## Check 2: Cross-threshold byte-equality (same case, all 6 thresholds)
Same SHA across thresholds: 10/10 cases
Multiple SHAs: 0/10 cases

## Check 3: Anchor pool populated in any cell? NO

## Per-case × per-threshold table

| case_id | t=0.85 | t=0.9 | t=0.95 | t=0.97 | t=0.99 | t=1.0 |
|---|---|---|---|---|---|---|
| astropy__astropy-12907 | 725b2ff2(0) | 725b2ff2(0) | 725b2ff2(0) | 725b2ff2(0) | 725b2ff2(0) | 725b2ff2(0) |
| django__django-10097 | c6479475(0) | c6479475(0) | c6479475(0) | c6479475(0) | c6479475(0) | c6479475(0) |
| matplotlib__matplotlib-13989 | da39a3ee(0) | da39a3ee(0) | da39a3ee(0) | da39a3ee(0) | da39a3ee(0) | da39a3ee(0) |
| mwaskom__seaborn-3069 | ff771601(0) | ff771601(0) | ff771601(0) | ff771601(0) | ff771601(0) | ff771601(0) |
| pallets__flask-5014 | 0cec508d(0) | 0cec508d(0) | 0cec508d(0) | 0cec508d(0) | 0cec508d(0) | 0cec508d(0) |
| psf__requests-1142 | a1fd8183(0) | a1fd8183(0) | a1fd8183(0) | a1fd8183(0) | a1fd8183(0) | a1fd8183(0) |
| pydata__xarray-2905 | a5e26b6f(0) | a5e26b6f(0) | a5e26b6f(0) | a5e26b6f(0) | a5e26b6f(0) | a5e26b6f(0) |
| pylint-dev__pylint-4551 | d11714c5(0) | d11714c5(0) | d11714c5(0) | d11714c5(0) | d11714c5(0) | d11714c5(0) |
| pytest-dev__pytest-10051 | 1fea8d49(0) | 1fea8d49(0) | 1fea8d49(0) | 1fea8d49(0) | 1fea8d49(0) | 1fea8d49(0) |
| scikit-learn__scikit-learn-10297 | b51e6d55(0) | b51e6d55(0) | b51e6d55(0) | b51e6d55(0) | b51e6d55(0) | b51e6d55(0) |

## Per-cell detailed byte-equality

### threshold=0.85 chunk=c1_10
| case | pk_bytes | pk_sha | lsy_bytes | lsy_sha | equal | pool_hit | copy_method |
| scikit-learn__scikit-learn-10297 | 3365 | b51e6d55 | 3365 | b51e6d55 | ✅ | 0 | none |

### threshold=0.85 chunk=c3_01
| case | pk_bytes | pk_sha | lsy_bytes | lsy_sha | equal | pool_hit | copy_method |
| astropy__astropy-12907 | 3868 | 725b2ff2 | 3868 | 725b2ff2 | ✅ | 0 | none |
| django__django-10097 | 2570 | c6479475 | 2570 | c6479475 | ✅ | 0 | none |
| matplotlib__matplotlib-13989 | 0 | da39a3ee | 3561 | 79a09734 | ❌ | 0 | — |

### threshold=0.85 chunk=c3_04
| case | pk_bytes | pk_sha | lsy_bytes | lsy_sha | equal | pool_hit | copy_method |
| mwaskom__seaborn-3069 | 1061 | ff771601 | 1061 | ff771601 | ✅ | 0 | none |
| pallets__flask-5014 | 904 | 0cec508d | 904 | 0cec508d | ✅ | 0 | none |
| psf__requests-1142 | 431 | a1fd8183 | 431 | a1fd8183 | ✅ | 0 | none |

### threshold=0.85 chunk=c3_07
| case | pk_bytes | pk_sha | lsy_bytes | lsy_sha | equal | pool_hit | copy_method |
| pydata__xarray-2905 | 697 | a5e26b6f | 697 | a5e26b6f | ✅ | 0 | none |
| pylint-dev__pylint-4551 | 4147 | d11714c5 | 4147 | d11714c5 | ✅ | 0 | none |
| pytest-dev__pytest-10051 | 694 | 1fea8d49 | 694 | 1fea8d49 | ✅ | 0 | none |

### threshold=0.9 chunk=c1_10
| case | pk_bytes | pk_sha | lsy_bytes | lsy_sha | equal | pool_hit | copy_method |
| scikit-learn__scikit-learn-10297 | 3365 | b51e6d55 | 3365 | b51e6d55 | ✅ | 0 | none |

### threshold=0.9 chunk=c3_01
| case | pk_bytes | pk_sha | lsy_bytes | lsy_sha | equal | pool_hit | copy_method |
| astropy__astropy-12907 | 3868 | 725b2ff2 | 3868 | 725b2ff2 | ✅ | 0 | none |
| django__django-10097 | 2570 | c6479475 | 2570 | c6479475 | ✅ | 0 | none |
| matplotlib__matplotlib-13989 | 0 | da39a3ee | 3561 | 79a09734 | ❌ | 0 | — |

### threshold=0.9 chunk=c3_04
| case | pk_bytes | pk_sha | lsy_bytes | lsy_sha | equal | pool_hit | copy_method |
| mwaskom__seaborn-3069 | 1061 | ff771601 | 1061 | ff771601 | ✅ | 0 | none |
| pallets__flask-5014 | 904 | 0cec508d | 904 | 0cec508d | ✅ | 0 | none |
| psf__requests-1142 | 431 | a1fd8183 | 431 | a1fd8183 | ✅ | 0 | none |

### threshold=0.9 chunk=c3_07
| case | pk_bytes | pk_sha | lsy_bytes | lsy_sha | equal | pool_hit | copy_method |
| pydata__xarray-2905 | 697 | a5e26b6f | 697 | a5e26b6f | ✅ | 0 | none |
| pylint-dev__pylint-4551 | 4147 | d11714c5 | 4147 | d11714c5 | ✅ | 0 | none |
| pytest-dev__pytest-10051 | 694 | 1fea8d49 | 694 | 1fea8d49 | ✅ | 0 | none |

### threshold=0.95 chunk=c1_10
| case | pk_bytes | pk_sha | lsy_bytes | lsy_sha | equal | pool_hit | copy_method |
| scikit-learn__scikit-learn-10297 | 3365 | b51e6d55 | 3365 | b51e6d55 | ✅ | 0 | none |

### threshold=0.95 chunk=c3_01
| case | pk_bytes | pk_sha | lsy_bytes | lsy_sha | equal | pool_hit | copy_method |
| astropy__astropy-12907 | 3868 | 725b2ff2 | 3868 | 725b2ff2 | ✅ | 0 | none |
| django__django-10097 | 2570 | c6479475 | 2570 | c6479475 | ✅ | 0 | none |
| matplotlib__matplotlib-13989 | 0 | da39a3ee | 3561 | 79a09734 | ❌ | 0 | — |

### threshold=0.95 chunk=c3_04
| case | pk_bytes | pk_sha | lsy_bytes | lsy_sha | equal | pool_hit | copy_method |
| mwaskom__seaborn-3069 | 1061 | ff771601 | 1061 | ff771601 | ✅ | 0 | none |
| pallets__flask-5014 | 904 | 0cec508d | 904 | 0cec508d | ✅ | 0 | none |
| psf__requests-1142 | 431 | a1fd8183 | 431 | a1fd8183 | ✅ | 0 | none |

### threshold=0.95 chunk=c3_07
| case | pk_bytes | pk_sha | lsy_bytes | lsy_sha | equal | pool_hit | copy_method |
| pydata__xarray-2905 | 697 | a5e26b6f | 697 | a5e26b6f | ✅ | 0 | none |
| pylint-dev__pylint-4551 | 4147 | d11714c5 | 4147 | d11714c5 | ✅ | 0 | none |
| pytest-dev__pytest-10051 | 694 | 1fea8d49 | 694 | 1fea8d49 | ✅ | 0 | none |

### threshold=0.97 chunk=c1_10
| case | pk_bytes | pk_sha | lsy_bytes | lsy_sha | equal | pool_hit | copy_method |
| scikit-learn__scikit-learn-10297 | 3365 | b51e6d55 | 3365 | b51e6d55 | ✅ | 0 | none |

### threshold=0.97 chunk=c3_01
| case | pk_bytes | pk_sha | lsy_bytes | lsy_sha | equal | pool_hit | copy_method |
| astropy__astropy-12907 | 3868 | 725b2ff2 | 3868 | 725b2ff2 | ✅ | 0 | none |
| django__django-10097 | 2570 | c6479475 | 2570 | c6479475 | ✅ | 0 | none |
| matplotlib__matplotlib-13989 | 0 | da39a3ee | 3561 | 79a09734 | ❌ | 0 | — |

### threshold=0.97 chunk=c3_04
| case | pk_bytes | pk_sha | lsy_bytes | lsy_sha | equal | pool_hit | copy_method |
| mwaskom__seaborn-3069 | 1061 | ff771601 | 1061 | ff771601 | ✅ | 0 | none |
| pallets__flask-5014 | 904 | 0cec508d | 904 | 0cec508d | ✅ | 0 | none |
| psf__requests-1142 | 431 | a1fd8183 | 431 | a1fd8183 | ✅ | 0 | none |

### threshold=0.97 chunk=c3_07
| case | pk_bytes | pk_sha | lsy_bytes | lsy_sha | equal | pool_hit | copy_method |
| pydata__xarray-2905 | 697 | a5e26b6f | 697 | a5e26b6f | ✅ | 0 | none |
| pylint-dev__pylint-4551 | 4147 | d11714c5 | 4147 | d11714c5 | ✅ | 0 | none |
| pytest-dev__pytest-10051 | 694 | 1fea8d49 | 694 | 1fea8d49 | ✅ | 0 | none |

### threshold=0.99 chunk=c1_10
| case | pk_bytes | pk_sha | lsy_bytes | lsy_sha | equal | pool_hit | copy_method |
| scikit-learn__scikit-learn-10297 | 3365 | b51e6d55 | 3365 | b51e6d55 | ✅ | 0 | none |

### threshold=0.99 chunk=c3_01
| case | pk_bytes | pk_sha | lsy_bytes | lsy_sha | equal | pool_hit | copy_method |
| astropy__astropy-12907 | 3868 | 725b2ff2 | 3868 | 725b2ff2 | ✅ | 0 | none |
| django__django-10097 | 2570 | c6479475 | 2570 | c6479475 | ✅ | 0 | none |
| matplotlib__matplotlib-13989 | 0 | da39a3ee | 3561 | 79a09734 | ❌ | 0 | — |

### threshold=0.99 chunk=c3_04
| case | pk_bytes | pk_sha | lsy_bytes | lsy_sha | equal | pool_hit | copy_method |
| mwaskom__seaborn-3069 | 1061 | ff771601 | 1061 | ff771601 | ✅ | 0 | none |
| pallets__flask-5014 | 904 | 0cec508d | 904 | 0cec508d | ✅ | 0 | none |
| psf__requests-1142 | 431 | a1fd8183 | 431 | a1fd8183 | ✅ | 0 | none |

### threshold=0.99 chunk=c3_07
| case | pk_bytes | pk_sha | lsy_bytes | lsy_sha | equal | pool_hit | copy_method |
| pydata__xarray-2905 | 697 | a5e26b6f | 697 | a5e26b6f | ✅ | 0 | none |
| pylint-dev__pylint-4551 | 4147 | d11714c5 | 4147 | d11714c5 | ✅ | 0 | none |
| pytest-dev__pytest-10051 | 694 | 1fea8d49 | 694 | 1fea8d49 | ✅ | 0 | none |

### threshold=1.0 chunk=c1_10
| case | pk_bytes | pk_sha | lsy_bytes | lsy_sha | equal | pool_hit | copy_method |
| scikit-learn__scikit-learn-10297 | 3365 | b51e6d55 | 3365 | b51e6d55 | ✅ | 0 | none |

### threshold=1.0 chunk=c3_01
| case | pk_bytes | pk_sha | lsy_bytes | lsy_sha | equal | pool_hit | copy_method |
| astropy__astropy-12907 | 3868 | 725b2ff2 | 3868 | 725b2ff2 | ✅ | 0 | none |
| django__django-10097 | 2570 | c6479475 | 2570 | c6479475 | ✅ | 0 | none |
| matplotlib__matplotlib-13989 | 0 | da39a3ee | 3561 | 79a09734 | ❌ | 0 | — |

### threshold=1.0 chunk=c3_04
| case | pk_bytes | pk_sha | lsy_bytes | lsy_sha | equal | pool_hit | copy_method |
| mwaskom__seaborn-3069 | 1061 | ff771601 | 1061 | ff771601 | ✅ | 0 | none |
| pallets__flask-5014 | 904 | 0cec508d | 904 | 0cec508d | ✅ | 0 | none |
| psf__requests-1142 | 431 | a1fd8183 | 431 | a1fd8183 | ✅ | 0 | none |

### threshold=1.0 chunk=c3_07
| case | pk_bytes | pk_sha | lsy_bytes | lsy_sha | equal | pool_hit | copy_method |
| pydata__xarray-2905 | 697 | a5e26b6f | 697 | a5e26b6f | ✅ | 0 | none |
| pylint-dev__pylint-4551 | 4147 | d11714c5 | 4147 | d11714c5 | ✅ | 0 | none |
| pytest-dev__pytest-10051 | 694 | 1fea8d49 | 694 | 1fea8d49 | ✅ | 0 | none |

## Cross-experiment byte-equality
| case_id | phase3 placeholder_knn_lossy (any t) | phase2 lossy | phase2 v44 | all equal? |
|---|---|---|---|---|
| astropy__astropy-12907 | 725b2ff2 | c4922bc2 | c4922bc2 | ❌ |
| django__django-10097 | c6479475 | eab6149a | eab6149a | ❌ |
| matplotlib__matplotlib-13989 | da39a3ee | c3b8e597 | c3b8e597 | ❌ |
| mwaskom__seaborn-3069 | ff771601 | e7ff2e24 | e7ff2e24 | ❌ |
| pallets__flask-5014 | 0cec508d | 0cec508d | 0cec508d | ✅ |
| psf__requests-1142 | a1fd8183 | 1023547e | 1023547e | ❌ |
| pydata__xarray-2905 | a5e26b6f | 78dd2983 | 78dd2983 | ❌ |
| pylint-dev__pylint-4551 | d11714c5 | e2c1e7fb | e2c1e7fb | ❌ |
| pytest-dev__pytest-10051 | 1fea8d49 | 1fea8d49 | 1fea8d49 | ✅ |
| scikit-learn__scikit-learn-10297 | b51e6d55 | 369d87fa | 369d87fa | ❌ |

## Final summary
❌ Regression detected: 6 within-cell diffs, 0 cross-threshold diffs
