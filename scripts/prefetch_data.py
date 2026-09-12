"""Prefetch candidate benchmark datasets into the HF cache."""
import traceback
import mteb

TASKS = [
    "BIOSSES",
    "NFCorpus", "SciFact", "TRECCOVID", "CUREv1",
    "BrightBiologyRetrieval",
    "R2MEDBiologyRetrieval", "R2MEDBioinformaticsRetrieval",
    "BiorxivClusteringP2P.v2", "BiorxivClusteringS2S.v2",
    "MedrxivClusteringP2P.v2", "MedrxivClusteringS2S.v2",
    "SciRepEvalMeSHDescriptorsClassification",
    "SciRepEvalDRSMClassification",
    "PubChemSynonymPC", "PubChemWikiParagraphsPC",
    "BIRCO-ClinicalTrial",
]
for name in TASKS:
    try:
        t = mteb.get_task(name)
        t.load_data()
        print(f"OK   {name}")
    except Exception as e:
        print(f"FAIL {name}: {e}")
        traceback.print_exc()
