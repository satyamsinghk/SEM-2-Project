"""
NLP Feature Extractor for LLVM IR.

This module treats the LLVM Intermediate Representation (IR) as natural language 
and uses a pre-trained Large Language Model (e.g., CodeBERT) to extract dense 
vector embeddings. This represents the state-of-the-art approach to feature 
extraction in compiler optimization (2024+).
"""
import os
import numpy as np
from typing import List

from src.feature_extraction.autophase_features import ProgramCharacteristics
from src.utils.logger import setup_logger

logger = setup_logger("NLPFeatureExtractor")


class NLPFeatureExtractor:
    """
    Extracts dense embeddings from LLVM IR using CodeBERT.
    Produces a 768-dimensional feature vector per program.
    """
    def __init__(self, model_name: str = "microsoft/codebert-base", device: str = None):
        """
        Initialize the NLP Feature Extractor.
        
        Args:
            model_name: HuggingFace model identifier
            device: Compute device ('cuda', 'mps', or 'cpu')
        """
        self.model_name = model_name
        self.num_features = 768  # CodeBERT hidden size
        
        
        import torch
        if device is None:
            if torch.cuda.is_available():
                self.device = "cuda"
            elif torch.backends.mps.is_available():
                self.device = "mps"
            else:
                self.device = "cpu"
        else:
            self.device = device
            
        logger.info(f"Initializing NLPFeatureExtractor with {self.model_name} on {self.device}")
        
        try:
            from transformers import AutoTokenizer, AutoModel
            self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
            self.model = AutoModel.from_pretrained(self.model_name).to(self.device)
            self.model.eval()  # Set to evaluation mode
        except ImportError:
            logger.error("transformers package not installed. 'pip install transformers'")
            raise
        except Exception as e:
            logger.error(f"Failed to load model {self.model_name}: {e}")
            raise

    def _get_ir_text(self, program: ProgramCharacteristics) -> str:
        """
        Read the LLVM IR (.ll) file content.
        """
        if not os.path.exists(program.ir_file_path):
            raise FileNotFoundError(f"IR file not found: {program.ir_file_path}")
            
        with open(program.ir_file_path, 'r', errors='ignore') as f:
            # Optionally strip comments, metadata, etc. to fit within max_length
            content = f.read()
        return content

    def extract(self, program: ProgramCharacteristics) -> np.ndarray:
        """
        Extract NLP embedding for a single program.
        """
        ir_text = self._get_ir_text(program)
        
        # Tokenize (truncate logic since IR can be very long)
        import torch
        inputs = self.tokenizer(ir_text, return_tensors="pt", truncation=True, 
                                max_length=512, padding="max_length")
        
        inputs = {k: v.to(self.device) for k, v in inputs.items()}
        
        with torch.no_grad():
            outputs = self.model(**inputs)
            # Use the [CLS] token representation or mean pooling
            embedding = outputs.last_hidden_state[:, 0, :].squeeze(0).cpu().numpy()
            
        return embedding

    def extract_batch(self, programs: List[ProgramCharacteristics]) -> np.ndarray:
        """
        Extract NLP embeddings for a batch of programs.
        """
        logger.info(f"Extracting NLP embeddings for {len(programs)} programs...")
        features = np.zeros((len(programs), self.num_features))
        
        # We can implement proper batching, but for simplicity we iterate.
        for i, prog in enumerate(programs):
            features[i] = self.extract(prog)
            if (i+1) % 10 == 0:
                logger.debug(f"Extracted NLP features for {i+1}/{len(programs)} programs")
                
        return features

    def get_feature_names(self) -> List[str]:
        """Get names for the embedding dimensions."""
        return [f"nlp_emb_{i}" for i in range(self.num_features)]
