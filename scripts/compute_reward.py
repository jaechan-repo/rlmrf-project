from typing import List
from rlmrf.typing import Data
from rlmrf.perplexity import LLaVAPerplexity

def compute_llava_reward(dataset: List[Data], 
                   batch_size: int):
    LLaVAPerplexity()