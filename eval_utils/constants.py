LABEL_MAPPING = {
    "nevus": "Nevus",
    "Atypical Nevus": "Nevus",
    "Melanocytic nevus": "Nevus",
    "Nevus (atypical/compound)": "Nevus",
    "Nevus (atypical/dysplastic nevus)": "Nevus",
    "melanoma": "Melanoma",
    "Melanocytic: Melanoma": "Melanoma",
    "pigment benign keratosis (pbk)": "Pigmented benign keratosis",
    "pbk": "Pigmented benign keratosis",
    "Pigmented Benign Keratosis": "Pigmented benign keratosis",
    "Pigmented benign keratosis (pigmented seborrheic keratosis)": "Pigmented benign keratosis",
    "Pigmented benign keratosis (irritated/keratotic seborrheic-type lesion)": "Pigmented benign keratosis",
    "actinic keratosis (ak)": "Actinic Keratosis",
    "Actinic Keratosis": "Actinic Keratosis",
    "Actinic keratosis": "Actinic Keratosis",
    "Actinic keratosis (keratinocytic)": "Actinic Keratosis",
    "Actinic keratosis (pigmented)": "Actinic Keratosis",
    "Actinic keratosis (pigmented, keratinocytic lesion)": "Actinic Keratosis",
    "Actinic keratosis (keratinocytic neoplasia spectrum)": "Actinic Keratosis",
    "Actinic keratosis (keratinocytic lesion)": "Actinic Keratosis",
    "Actinic keratosis (pigmented keratinocytic lesion)": "Actinic Keratosis",
    "ak": "Actinic Keratosis",
    "basal cell carcinoma": "Basal cell carcinoma",
    "Basal Cell Carcinoma": "Basal cell carcinoma",
    "Pigmented Basal cell carcinoma": "Basal cell carcinoma",
    "scc": "Squamous cell carcinoma",
    "Squamous Cell Carcinoma": "Squamous cell carcinoma",
    "Squamous cell carcinoma (keratinocytic)": "Squamous cell carcinoma",
    "Squamous cell carcinoma (pigmented/keratotic)": "Squamous cell carcinoma",
    "df": "Dermatofibroma",
    "dermatofibroma": "Dermatofibroma",
    "Dermatofibroma (fibrohistiocytic lesion)": "Dermatofibroma",
    "Dermatofibroma (Fibrohistiocytic)": "Dermatofibroma"
}

DEFAULT_LEXICON = {
    # Melanocytic terms
    "pigment network", "negative network", "streaks", "pseudopods", "blue-white veil",
    "regression", "globules", "dots", "streak", "pseudopod", "blue white veil",
    # Keratinocytic terms
    "arborizing vessels", "leaf-like areas", "spoke-wheel", "blue-gray ovoid nests",
    "shiny white", "ulcer", "rolled border", "milia-like cysts", "comedo-like openings",
    "fissures", "ridges", "cerebriform", "moth-eaten", "fingerprint-like lines",
    "strawberry", "targetoid follicles", "rosettes", "white circles", "hairpin vessels",
    "glomerular vessels", "keratin plug", "scale", "crust", "telangiectasias",
    # Fibrohistiocytic terms
    "scar-like white", "peripheral delicate network", "chrysalis", "crystalline",
    # Generic spatial/structure terms
    "central", "peripheral", "symmetry", "asymmetry", "homogeneous", "heterogeneous",
    "structureless", "reticular", "annular", "granular", "network", "vessels", "ulceration",
    "border", "well-defined", "ill-defined", "irregular"
}

FAMILY_FROM_DX = {
    "Nevus": "Melanocytic",
    "Melanoma": "Melanocytic",
    "Basal cell carcinoma": "Keratinocytic",
    "Squamous cell carcinoma": "Keratinocytic",
    "Pigmented benign keratosis": "Keratinocytic",
    "Actinic Keratosis": "Keratinocytic",
    "Dermatofibroma": "Fibrohistiocytic",
}
