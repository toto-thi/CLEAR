from pathlib import Path
from typing import List, Dict, Optional, Any
import numpy as np, torch
from PIL import Image
from open_clip import create_model_from_pretrained
from qdrant_client import QdrantClient

MODEL_ID = "hf-hub:microsoft/BiomedCLIP-PubMedBERT_256-vit_base_patch16_224"

def _l2_normalize(v: np.ndarray) -> np.ndarray:
    n = np.linalg.norm(v)
    return v / (n + 1e-12)

def _cos_sim(a: np.ndarray, b: np.ndarray) -> float:
    return float(np.dot(a, b))

def _extract_vector_from_qdrant_vec(vec_field: Any) -> Optional[np.ndarray]:
    """
    Qdrant can return:
      - list[float]
      - dict[str, list[float]] for named vectors
    """
    if vec_field is None:
        return None
    if isinstance(vec_field, dict):
        for _, v in vec_field.items():
            return np.asarray(v, dtype=np.float32)
        return None
    return np.asarray(vec_field, dtype=np.float32)

class QdrantBiomedCLIPRetriever:
    def __init__(
        self,
        collection: str,
        qdrant_url: str = "http://127.0.0.1:6333",
        device: Optional[str] = None,
        model_id: str = MODEL_ID
    ):
        self.collection = collection
        self.client = QdrantClient(url=qdrant_url, timeout=120.0)
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.model, self.preprocess = create_model_from_pretrained(model_id)
        if self.device.startswith("cuda"):
            torch.cuda.set_device(0)
        self.model.to(self.device).eval()

    @torch.no_grad()
    def _embed_image(self, pil_img: Image.Image) -> np.ndarray:
        t = self.preprocess(pil_img.convert("RGB")).unsqueeze(0).to(self.device)
        z = self.model.encode_image(t)                 # (1, D)
        z = z / z.norm(dim=-1, keepdim=True)
        return z.squeeze(0).float().cpu().numpy()

    def _coarse(self, q_vec: np.ndarray, limit: int) -> List[Dict[str, Any]]:
        hits = self.client.search(
            collection_name=self.collection,
            query_vector=q_vec.tolist(),
            limit=limit,
            with_payload=True,
            with_vectors=True,   
        )
        out = []
        for h in hits:
            p = h.payload or {}
            img = p.get("image_path")
            if not img:
                continue
            cand_vec = _extract_vector_from_qdrant_vec(getattr(h, "vector", None))
            out.append({
                "image_path": img,
                "age": p.get("age"),                      
                "sex": p.get("sex"),                      
                "anatom_site": p.get("anatom_site") or p.get("anatom_site_general"),
                "diagnosis": p.get("diagnosis"),
                "qdrant_score": float(h.score) if h.score is not None else None,
                "vec": cand_vec,   # may be None -> we’ll re-embed on demand
            })
        return out

    def _ensure_vectors(self, items: List[Dict[str, Any]]):
        # Re-embed only missing vectors.
        need = [it for it in items if it.get("vec") is None]
        if not need:
            return
        for it in need:
            try:
                v = self._embed_image(Image.open(Path(it["image_path"]).resolve()))
                it["vec"] = v
            except Exception:
                it["vec"] = None  # drop later if still None

    def _mmr(self,
             items: List[Dict[str, Any]],
             q_vec: np.ndarray,
             k: int,
             lambda_: float = 0.65,
             dx_penalty: float = 0.10) -> List[Dict[str, Any]]:
        """
        MMR on vectors + light same-diagnosis penalty to encourage variety.
        """
        selected: List[Dict[str, Any]] = []
        pool = [it for it in items if it.get("vec") is not None]

        # Precompute relevance to query
        for it in pool:
            it["rel"] = _cos_sim(q_vec, _l2_normalize(it["vec"]))

        # Greedy selection
        while pool and len(selected) < k:
            best = None
            best_score = -1e9
            for cand in pool:
                # redundancy = max cosine sim to any already selected
                if selected:
                    red = max(_cos_sim(_l2_normalize(cand["vec"]),
                                       _l2_normalize(s["vec"])) for s in selected)
                else:
                    red = 0.0
                # same-diagnosis penalty if diagnosis already in selection
                same_dx = any(
                    (cand.get("diagnosis") is not None and cand.get("diagnosis") == s.get("diagnosis"))
                    for s in selected
                )
                penalty = dx_penalty if same_dx else 0.0
                mmr_score = lambda_ * cand["rel"] - (1 - lambda_) * red - penalty
                if mmr_score > best_score:
                    best_score, best = mmr_score, cand
            selected.append(best)
            pool.remove(best)
        return selected

    def search(
        self,
        image_path: str,
        k: int = 3,
        exclude_same: bool = True,
        k_coarse: int = 100,
        mmr_lambda: float = 0.65,
        dx_penalty: float = 0.10,
    ) -> List[Dict]:
        """
        Return exactly k exemplars re-ranked by MMR with vector redundancy + same-dx penalty.
        """
        src_abs = str(Path(image_path).resolve())
        q_vec = self._embed_image(Image.open(src_abs))

        # 1) Coarse shortlist
        coarse = self._coarse(q_vec, limit=max(k_coarse, k + 20))

        # 2) Exclude the exact same file if it appears in the index
        if exclude_same:
            coarse = [c for c in coarse if str(Path(c["image_path"]).resolve()) != src_abs]

        # 3) Ensure vectors for all candidates (either from Qdrant or re-embed)
        self._ensure_vectors(coarse)
        coarse = [c for c in coarse if c.get("vec") is not None]
        if not coarse:
            return []
        
        # 4) MMR re-ranking 
        final = self._mmr(coarse, q_vec, k=k, lambda_=mmr_lambda, dx_penalty=dx_penalty)

        # 5) Minimal payload back
        return [
            {
                "image_path": c["image_path"],
                "age": c.get("age"),
                "sex": c.get("sex"),
                "anatom_site": c.get("anatom_site"),
                "diagnosis": c.get("diagnosis"),
                "score": float(f"{c.get('rel', 0.0):.2f}"),
            }
            for c in final
        ]
