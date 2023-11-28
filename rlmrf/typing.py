from typing import TypedDict, Dict, List, Any, Optional
from PIL.Image import Image

class Doc(TypedDict):
    doc_id: str
    doc_score: str
    title: Optional[str]
    text: Optional[str]

class Question(TypedDict):
    text: str
    image_file: str
    image: Optional[Image]
    qid: str

class Response(TypedDict):
    text: str
    rid: str
    reward: float

class DataQA(TypedDict):

    question: Question              # Question type (e.g., text, image_file) to question
    ref_cap: Dict[str, str]         # Caption type to caption
    ref_ans: List[str]              # List of answers, first being most relevant
    split: str                      # "train" / "dev" / "test"
    dataset: str                    # e.g, "a-okvqa"

    blip_cap: Optional[str]
    docs: Optional[List[Doc]]        
    responses: Optional[List[Response]]
    top_response: Optional[Response]

    misc: Optional[Dict]

    
class LLaVAFinetuneData(TypedDict):
    """
    e.g.,
    {
        "id": "997bb945-628d-4724-b370-b84de974a19f",
        "image": "part-000001/997bb945-628d-4724-b370-b84de974a19f.jpg",
        "conversations": [
        {
            "from": "human",
            "value": "<image>\nWrite a prompt for Stable Diffusion to generate this image."
        },
        {
            "from": "gpt",
            "value": "a beautiful painting of chernobyl by nekro, pascal blanche, john harris, greg rutkowski, sin jong hun, moebius, simon stalenhag. in style of cg art. ray tracing. cel shading. hyper detailed. realistic. ue 5. maya. octane render. "
        },
        ]
    }
    """
    id: str
    image: str
    conversations: List[Dict]