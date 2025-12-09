from src.query_disambiguation import detect_medications_in_query, _norm
from src.vector_store import VectorStoreManager

q = "Hverjar eru algengar aukaverkanir af dicloxacilin hjá unglingum?"

print("q_norm:", _norm(q))

# 1) Hard-coded Íbúfen only
single_meds = ["Dicloxacillin_Bluefish_SmPC_SmPC"]
print("\n=== SINGLE MED TEST ===")
print("norm(single_meds[0]):", _norm(single_meds[0]))
print("detect(single):", detect_medications_in_query(q, single_meds))

# 2) Same Íbúfen pulled from the vector store list
v = VectorStoreManager()
meds = v.get_unique_medications()
dicloxacillin_meds = [m for m in meds if "Dicloxacillin" in m]

print("\n=== STORE MED TEST ===")
print("dicloxacillin_meds:", dicloxacillin_meds)
print("normed dicloxacillin_meds:", [_norm(m) for m in dicloxacillin_meds])
print("detect(store subset):", detect_medications_in_query(q, dicloxacillin_meds))