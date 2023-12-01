from typing import TypedDict, Dict, List, Any, Optional
from PIL.Image import Image

class Doc(TypedDict):
    doc_id: str
    doc_score: str
    title: Optional[str]
    text: Optional[str]

class Question(TypedDict):
    question_id: str
    text: str
    image_id: Optional[str]
    image_file: Optional[str]

class Response(TypedDict):
    text: str
    rid: str
    reward: Optional[float]

class DataQA(TypedDict):
    question: Question
    split: str                      # "train" / "dev" / "test"
    dataset: str                    # e.g, "aokvqa"
    ref_cap: Optional[Dict]         # reference caption
    ref_ans: Optional[Dict]         # reference answer
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