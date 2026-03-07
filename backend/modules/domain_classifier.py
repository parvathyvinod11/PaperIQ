"""
Research Domain Classifier
Classifies papers into research domains using TF-IDF + cosine similarity
with domain seed vocabularies (fast, no GPU needed).
Falls back to zero-shot classification if transformers available.
"""

from typing import Dict, List, Tuple
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np

DOMAIN_SEEDS = {
    "Artificial Intelligence": [
        "artificial intelligence machine learning deep learning neural network reinforcement learning "
        "supervised unsupervised optimization loss function gradient backpropagation training inference"
    ],
    "Computer Vision": [
        "image recognition object detection segmentation convolutional neural network CNN "
        "visual feature extraction bounding box pixel classification GAN generative adversarial"
    ],
    "Natural Language Processing": [
        "natural language processing text classification sentiment analysis named entity recognition "
        "machine translation language model BERT GPT transformer tokenization embedding"
    ],
    "Cybersecurity": [
        "cybersecurity intrusion detection malware network security vulnerability encryption "
        "attack defense firewall anomaly detection threat intelligence authentication"
    ],
    "Bioinformatics": [
        "bioinformatics genomics proteomics DNA RNA gene sequence protein structure "
        "biological network phylogenetics molecular docking drug discovery"
    ],
    "Robotics": [
        "robotics autonomous robot motion planning control system sensor perception "
        "SLAM localization manipulation grasping kinematics actuator reinforcement"
    ],
    "Data Science": [
        "data science big data analytics data mining feature engineering visualization "
        "statistical analysis regression clustering pattern recognition database"
    ],
    "Computer Networks": [
        "network protocol routing wireless communication bandwidth latency throughput "
        "peer-to-peer distributed system cloud computing edge computing IoT"
    ],
    "Software Engineering": [
        "software engineering requirements analysis design testing debugging maintenance "
        "agile scrum DevOps microservices API code quality refactoring"
    ],
    "Human-Computer Interaction": [
        "human computer interaction user interface usability accessibility design "
        "gesture eye tracking virtual reality augmented reality wearable"
    ],
}


class DomainClassifier:
    def __init__(self):
        self.domains = list(DOMAIN_SEEDS.keys())
        self.seed_texts = [DOMAIN_SEEDS[d][0] for d in self.domains]
        self.vectorizer = TfidfVectorizer(ngram_range=(1, 2), stop_words="english")
        # Fit on domain seeds
        self.domain_vectors = self.vectorizer.fit_transform(self.seed_texts)

    def classify(self, text: str, top_n: int = 3) -> Dict:
        """Classify text into research domains."""
        try:
            text_vector = self.vectorizer.transform([text[:5000]])
            similarities = cosine_similarity(text_vector, self.domain_vectors)[0]
            ranked = sorted(
                zip(self.domains, similarities.tolist()),
                key=lambda x: x[1],
                reverse=True,
            )
            primary_domain = ranked[0][0]
            confidence = round(float(ranked[0][1]), 4)
            top_domains = [
                {"domain": d, "confidence": round(float(s), 4)}
                for d, s in ranked[:top_n]
            ]
            return {
                "primary_domain": primary_domain,
                "confidence": confidence,
                "top_domains": top_domains,
            }
        except Exception as e:
            return {
                "primary_domain": "Unknown",
                "confidence": 0.0,
                "top_domains": [],
                "error": str(e),
            }
