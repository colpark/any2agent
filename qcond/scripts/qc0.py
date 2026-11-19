"""QC-0: confirm reused assets, draw Practice (seed 0, 300 of the 497 AH bank rows) -> qcond/practice.json."""
import json, os, sys
sys.path.insert(0, os.path.dirname(__file__))
import numpy as np
from qc_common import *
R = rows(); Fp, Ft = feats(); E = np.load(f"{VSS}/emb_test.npy")[R["test"]]
A = astra19("bank", R["practice"], SOFT); At = astra19("test200", R["t200_test"], DIRECT); Q = qwen19(R["test"], DIRECT)
chk = {"practice_n": len(R["practice"]), "test_n": len(R["test"]), "t200_n": len(R["t200"]), "F_practice": Fp.shape, "F_test": Ft.shape,
       "emb_test": E.shape, "emb_test_nan": int(np.isnan(E).sum()), "astra_practice_missing": int(np.isnan(A).sum()),
       "astra_t200_missing": int(np.isnan(At).sum()), "qwen_test_default_0.5": int((Q == .5).sum()),
       "desc_bank": np.load(f"{ah.ST}/desc_bank.npy").shape, "desc_test": np.load(f"{ah.ST}/desc_test.npy").shape}
json.dump({"practice_bank_pos": practice().tolist(), "practice_rlpool_rows": R["practice"].tolist()}, open(f"{QC}/practice.json", "w"))
print(json.dumps({k: list(v) if isinstance(v, tuple) else v for k, v in chk.items()}))
