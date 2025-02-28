from transformers import (
    AutoTokenizer, AutoModelForSeq2SeqLM, pipeline,
    AutoModelForQuestionAnswering, T5ForConditionalGeneration
)
import spacy
import networkx as nx
import numpy as np
from typing import List, Dict, Any, Tuple, Optional
from pydantic import BaseModel, Field
from collections import defaultdict
import re
import json
from sklearn.feature_extraction.text import TfidfVectorizer
from gensim.models import Word2Vec
import torch
from scipy.spatial.distance import cosine
import nltk
from nltk.tokenize import sent_tokenize, word_tokenize
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from dataclasses import dataclass
import logging
from enum import Enum
import gtts
from collections import Counter

# Download required NLTK data
from nltk_setup import NLTKDownloader

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class EntityType(str, Enum):
    COMPONENT = "component"
    ACTOR = "actor"
    STATE = "state"
    ACTION = "action"
    PROTOCOL = "protocol"
    MESSAGE = "message"
    DATA = "data"

class RelationType(str, Enum):
    SENDS = "sends"
    RECEIVES = "receives"
    PROCESSES = "processes"
    TRANSITIONS = "transitions"
    CONTAINS = "contains"
    DEPENDS_ON = "depends_on"
    TRIGGERS = "triggers"

@dataclass
class AnimationRequirement:
    """Specifies requirements for animation generation"""
    shapes: List[Dict[str, Any]]
    connections: List[Dict[str, Any]]
    transitions: List[Dict[str, Any]]
    layout: Dict[str, Any]
    timing: Dict[str, Any]
    narration: List[str]

class ConceptElement(BaseModel):
    id: str
    name: str
    type: EntityType
    description: str
    properties: Dict[str, Any] = Field(default_factory=dict)
    related_terms: List[str] = Field(default_factory=list)
    importance_score: float = 0.0

class ConceptRelation(BaseModel):
    from_element: str
    to_element: str
    type: RelationType
    description: str
    properties: Dict[str, Any] = Field(default_factory=dict)
    sequence_number: int

class ConceptStep(BaseModel):
    order: int
    elements_involved: List[str]
    action: str
    description: str
    preconditions: List[str] = Field(default_factory=list)
    postconditions: List[str] = Field(default_factory=list)
    timing: Optional[str] = None
    importance_score: float = 0.0

class DetailedConceptAnalysis(BaseModel):
    title: str
    summary: str
    domain: str
    elements: List[ConceptElement]
    relations: List[ConceptRelation]
    steps: List[ConceptStep]
    keywords: List[str]
    animation_requirements: Dict[str, Any]
    metadata: Dict[str, Any] = Field(default_factory=dict)

class ConceptAnalyzer:
    def __init__(self):
        """Initialize the ConceptAnalyzer with required models and components"""
        self._initialize_nltk()
        self._initialize_nlp_components()
        self._load_domain_knowledge()
        self.graph = nx.DiGraph()
    
    def _initialize_nltk(self):
        """Initialize NLTK data"""
        NLTKDownloader.download_nltk_data()
        
    def _initialize_nlp_components(self):
        """Initialize all NLP models and components"""
        logger.info("Initializing NLP components...")
        
        # Core NLP models
        self.tokenizer = AutoTokenizer.from_pretrained("google/flan-t5-base")
        self.model = T5ForConditionalGeneration.from_pretrained("google/flan-t5-base")
        self.nlp = spacy.load("en_core_web_sm")
        
        # Specialized models
        self.qa_model = AutoModelForQuestionAnswering.from_pretrained(
            "deepset/roberta-base-squad2"
        )
        self.zero_shot = pipeline("zero-shot-classification")
        
        # Text processing tools
        self.lemmatizer = WordNetLemmatizer()
        self.stop_words = set(stopwords.words('english'))
        self.tfidf = TfidfVectorizer(
            stop_words='english',
            ngram_range=(1, 3),
            max_features=1000
        )
        
        logger.info("NLP components initialized successfully")

    def _load_domain_knowledge(self):
        """Load domain-specific knowledge and rules"""
        self.domain_patterns = {
            'networking': {
                'keywords': ['TCP', 'IP', 'HTTP', 'DNS', 'protocol', 'packet'],
                'entities': ['server', 'client', 'router', 'packet'],
                'relations': ['sends', 'receives', 'forwards']
            },
            'database': {
                'keywords': ['SQL', 'transaction', 'ACID', 'query', 'database'],
                'entities': ['table', 'record', 'index', 'transaction'],
                'relations': ['reads', 'writes', 'updates']
            }
            # Add more domains as needed
        }

    def analyze_concept(self, text: str) -> DetailedConceptAnalysis:
        """Main method to analyze a technical concept"""
        logger.info(f"Starting concept analysis for text: {text[:100]}...")
        
        # Initial text processing
        processed_text = self._preprocess_text(text)
        domain = self._identify_domain(processed_text)
        
        # Extract core components
        elements = self._extract_elements(processed_text, domain)
        relations = self._extract_relations(processed_text, elements)
        steps = self._extract_steps(processed_text, elements, relations)
        
        # Generate additional analysis
        keywords = self._extract_keywords(processed_text)
        summary = self._generate_summary(processed_text, elements, steps)
        animation_reqs = self._generate_animation_requirements(
            elements, relations, steps
        )
        
        # Create detailed analysis
        analysis = DetailedConceptAnalysis(
            title=self._generate_title(processed_text),
            summary=summary,
            domain=domain,
            elements=elements,
            relations=relations,
            steps=steps,
            keywords=keywords,
            animation_requirements=animation_reqs,
            metadata=self._generate_metadata(processed_text)
        )
        
        logger.info("Concept analysis completed successfully")
        return analysis

    def _preprocess_text(self, text: str) -> str:
        """Preprocess the input text"""
        # Basic cleaning
        text = text.lower().strip()
        text = re.sub(r'\s+', ' ', text)
        
        # Tokenization and lemmatization
        tokens = word_tokenize(text)
        lemmatized = [self.lemmatizer.lemmatize(token) for token in tokens]
        
        return ' '.join(lemmatized)

    def _identify_domain(self, text: str) -> str:
        """Identify the technical domain of the concept"""
        domain_scores = {}
        
        for domain, patterns in self.domain_patterns.items():
            score = sum(1 for keyword in patterns['keywords'] 
                       if keyword.lower() in text.lower())
            domain_scores[domain] = score
        
        return max(domain_scores.items(), key=lambda x: x[1])[0]

    def _extract_elements(self, text: str, domain: str) -> List[ConceptElement]:
        """Extract key elements from the text"""
        elements = []
        doc = self.nlp(text)
        
        # Extract named entities
        entities = [(ent.text, ent.label_) for ent in doc.ents]
        
        # Extract noun phrases
        noun_phrases = [chunk.text for chunk in doc.noun_chunks]
        
        # Combine and classify elements
        potential_elements = set(
            [e[0] for e in entities] + noun_phrases + 
            self.domain_patterns[domain]['entities']
        )
        
        for element in potential_elements:
            element_type = self._classify_element_type(element, domain)
            if element_type:
                element_obj = ConceptElement(
                    id=self._generate_id(element),
                    name=element,
                    type=element_type,
                    description=self._generate_element_description(element),
                    related_terms=self._find_related_terms(element, text),
                    importance_score=self._calculate_importance(element, text)
                )
                elements.append(element_obj)
        
        return elements

    def _extract_relations(
        self, text: str, elements: List[ConceptElement]
    ) -> List[ConceptRelation]:
        """Extract relationships between elements"""
        relations = []
        doc = self.nlp(text)
        
        # Build element lookup
        element_lookup = {elem.name: elem for elem in elements}
        
        # Extract relations from dependency parse
        for sent in doc.sents:
            for token in sent:
                if token.dep_ in ['ROOT', 'VERB']:
                    subject = [w for w in token.lefts if w.dep_ == 'nsubj']
                    object = [w for w in token.rights if w.dep_ == 'dobj']
                    
                    if subject and object:
                        subj_text = subject[0].text
                        obj_text = object[0].text
                        
                        if (subj_text in element_lookup and 
                            obj_text in element_lookup):
                            relation = ConceptRelation(
                                from_element=element_lookup[subj_text].id,
                                to_element=element_lookup[obj_text].id,
                                type=self._classify_relation_type(token.text),
                                description=self._generate_relation_description(
                                    subj_text, token.text, obj_text
                                ),
                                sequence_number=len(relations)
                            )
                            relations.append(relation)
        
        return relations

    def _extract_steps(
        self, text: str, 
        elements: List[ConceptElement],
        relations: List[ConceptRelation]
    ) -> List[ConceptStep]:
        """Extract sequential steps from the concept"""
        steps = []
        sentences = sent_tokenize(text)
        
        # Build knowledge graph
        self._build_knowledge_graph(elements, relations)
        
        current_step = 1
        for sent in sentences:
            # Identify elements involved in this sentence
            elements_involved = self._identify_elements_in_text(sent, elements)
            
            if elements_involved:
                step = ConceptStep(
                    order=current_step,
                    elements_involved=[e.id for e in elements_involved],
                    action=self._extract_main_action(sent),
                    description=sent,
                    preconditions=self._identify_preconditions(
                        elements_involved, current_step
                    ),
                    postconditions=self._identify_postconditions(
                        elements_involved, current_step
                    ),
                    timing=self._extract_timing_info(sent),
                    importance_score=self._calculate_step_importance(
                        sent, current_step
                    )
                )
                steps.append(step)
                current_step += 1
        
        return steps

    def _generate_animation_requirements(
        self,
        elements: List[ConceptElement],
        relations: List[ConceptRelation],
        steps: List[ConceptStep]
    ) -> Dict[str, Any]:
        """Generate detailed animation requirements"""
        # Shape requirements for each element
        shapes = []
        for elem in elements:
            shape = {
                'id': elem.id,
                'type': self._get_shape_for_entity_type(elem.type),
                'label': elem.name,
                'style': self._get_style_for_entity_type(elem.type)
            }
            shapes.append(shape)
        
        # Connection requirements
        connections = []
        for rel in relations:
            conn = {
                'from': rel.from_element,
                'to': rel.to_element,
                'type': self._get_connection_type(rel.type),
                'label': rel.description,
                'style': self._get_style_for_relation_type(rel.type)
            }
            connections.append(conn)
        
        # Animation transitions
        transitions = []
        for step in steps:
            trans = {
                'step_order': step.order,
                'elements': step.elements_involved,
                'action': step.action,
                'timing': step.timing or "2s",
                'animation_type': self._get_animation_type(step)
            }
            transitions.append(trans)
        
        # Generate narration script
        narration = self._generate_narration_script(steps)
        
        return {
            'shapes': shapes,
            'connections': connections,
            'transitions': transitions,
            'layout': self._generate_layout_requirements(elements, relations),
            'narration': narration
        }

    def _get_shape_for_entity_type(self, entity_type: EntityType) -> str:
        """Map entity types to shapes"""
        shape_mapping = {
            EntityType.COMPONENT: 'rectangle',
            EntityType.ACTOR: 'circle',
            EntityType.STATE: 'rounded_rectangle',
            EntityType.ACTION: 'diamond',
            EntityType.PROTOCOL: 'hexagon',
            EntityType.MESSAGE: 'parallelogram',
            EntityType.DATA: 'cylinder'
        }
        return shape_mapping.get(entity_type, 'rectangle')

    def _generate_narration_script(self, steps: List[ConceptStep]) -> List[str]:
        """Generate narration script for each step"""
        narration = []
        for step in steps:
            # Generate clear, concise description for this step
            description = f"Step {step.order}: {step.description}"
            
            # Add emphasis for important steps
            if step.importance_score > 0.7:
                description = f"Important: {description}"
            
            # Add timing cues
            if step.timing:
                description += f" This typically takes {step.timing}."
            
            narration.append(description)
        
        return narration

    def _generate_voice_over(self, narration: List[str], output_path: str):
        """Generate voice-over audio for the animation"""
        # Combine all narration into one text
        full_text = " ".join(narration)
        
        # Generate speech
        tts = gtts.gTTS(text=full_text, lang='en')
        tts.save(output_path)
        
        return output_path

    # Utility methods
    def _generate_id(self, text: str) -> str:
        """Generate unique ID for an element"""
        return re.sub(r'\W+', '_', text.lower())

    def _calculate_importance(self, element: str, text: str) -> float:
        """Calculate importance score for an element"""
        # Use TF-IDF score as base
        tfidf_matrix = self.tfidf.fit_transform([text])
        feature_names = self.tfidf.get_feature_names_out()
        
        try:
            idx = list(feature_names).index(element.lower())
            score = tfidf_matrix[0, idx]
        except ValueError:
            score = 0
        
        # Adjust score based on position and frequency
        def _calculate_importance(self, element: str, text: str) -> float:
                    """Calculate importance score for an element"""
        # Use TF-IDF score as base
        tfidf_matrix = self.tfidf.fit_transform([text])
        feature_names = self.tfidf.get_feature_names_out()
        
        try:
            idx = list(feature_names).index(element.lower())
            score = tfidf_matrix[0, idx]
        except ValueError:
            score = 0
        
        # Adjust score based on position and frequency
        mentions = text.lower().count(element.lower())
        first_occurrence = text.lower().find(element.lower())
        text_length = len(text)
        
        # Position weight: earlier mentions get higher weight
        position_score = 1 - (first_occurrence / text_length) if first_occurrence != -1 else 0
        
        # Frequency weight: more mentions indicate importance
        frequency_score = min(mentions / 10, 1)  # Cap at 1
        
        # Combine scores with weights
        final_score = (
            0.4 * float(score) +  # TF-IDF weight
            0.3 * position_score +  # Position weight
            0.3 * frequency_score   # Frequency weight
        )
        
        return min(final_score, 1.0)  # Normalize to [0,1]

    def _build_knowledge_graph(self, elements: List[ConceptElement], 
                             relations: List[ConceptRelation]):
        """Build a knowledge graph from elements and relations"""
        self.graph.clear()
        
        # Add nodes (elements)
        for element in elements:
            self.graph.add_node(
                element.id,
                type=element.type,
                name=element.name,
                properties=element.properties
            )
        
        # Add edges (relations)
        for relation in relations:
            self.graph.add_edge(
                relation.from_element,
                relation.to_element,
                type=relation.type,
                description=relation.description,
                sequence=relation.sequence_number
            )

    def _identify_preconditions(self, elements: List[ConceptElement], 
                              current_step: int) -> List[str]:
        """Identify preconditions for a step based on the knowledge graph"""
        preconditions = []
        
        for element in elements:
            # Get incoming edges
            predecessors = list(self.graph.predecessors(element.id))
            
            for pred in predecessors:
                edge_data = self.graph.get_edge_data(pred, element.id)
                
                # Check if this is a dependency
                if edge_data['type'] in [RelationType.DEPENDS_ON, 
                                       RelationType.TRIGGERS]:
                    pred_node = self.graph.nodes[pred]
                    preconditions.append(
                        f"{pred_node['name']} must be {edge_data['description']}"
                    )
        
        return preconditions

    def _identify_postconditions(self, elements: List[ConceptElement], 
                               current_step: int) -> List[str]:
        """Identify postconditions for a step based on the knowledge graph"""
        postconditions = []
        
        for element in elements:
            # Get outgoing edges
            successors = list(self.graph.successors(element.id))
            
            for succ in successors:
                edge_data = self.graph.get_edge_data(element.id, succ)
                
                # Check if this creates a new state or condition
                if edge_data['type'] in [RelationType.TRANSITIONS, 
                                       RelationType.PROCESSES]:
                    succ_node = self.graph.nodes[succ]
                    postconditions.append(
                        f"{succ_node['name']} becomes {edge_data['description']}"
                    )
        
        return postconditions

    def _extract_timing_info(self, text: str) -> Optional[str]:
        """Extract timing information from text"""
        # Common timing patterns
        timing_patterns = [
            r'(\d+)\s*(ms|milliseconds?)',
            r'(\d+)\s*(s|seconds?)',
            r'(\d+)\s*(min|minutes?)',
            r'typically\s+takes\s+(\d+\s*(?:ms|s|min))',
            r'within\s+(\d+\s*(?:ms|s|min))'
        ]
        
        for pattern in timing_patterns:
            match = re.search(pattern, text, re.IGNORECASE)
            if match:
                return match.group(0)
        
        return None

    def _calculate_step_importance(self, text: str, step_number: int) -> float:
        """Calculate importance score for a step"""
        # Base score from text analysis
        base_score = self._calculate_importance(text, text)
        
        # Adjust based on step position
        position_weight = 1.0
        if step_number == 1:
            position_weight = 1.2  # First step usually important
        elif step_number == 2:
            position_weight = 1.1  # Second step often important
        
        # Look for importance indicators
        importance_indicators = [
            'important', 'crucial', 'critical', 'key', 'essential',
            'must', 'required', 'necessary'
        ]
        
        indicator_score = sum(
            1 for indicator in importance_indicators 
            if indicator in text.lower()
        ) * 0.1
        
        final_score = min(
            (base_score * position_weight) + indicator_score, 
            1.0
        )
        return final_score

    def _generate_layout_requirements(
        self, 
        elements: List[ConceptElement],
        relations: List[ConceptRelation]
    ) -> Dict[str, Any]:
        """Generate layout requirements for animation"""
        # Calculate optimal positioning
        G = nx.DiGraph()
        
        # Add nodes and edges to temp graph
        for elem in elements:
            G.add_node(elem.id)
        
        for rel in relations:
            G.add_edge(rel.from_element, rel.to_element)
        
        # Use graph layout algorithm
        pos = nx.spring_layout(G)
        
        # Convert positions to animation coordinates
        layout = {
            'positions': {
                elem_id: {'x': float(coord[0]), 'y': float(coord[1])}
                for elem_id, coord in pos.items()
            },
            'dimensions': {
                'width': 1000,
                'height': 800,
                'padding': 50
            },
            'grouping': self._identify_element_groups(elements, relations)
        }
        
        return layout

    def _identify_element_groups(
        self,
        elements: List[ConceptElement],
        relations: List[ConceptRelation]
    ) -> List[Dict[str, Any]]:
        """Identify groups of related elements"""
        groups = []
        
        # Build graph for community detection
        G = nx.Graph()
        for elem in elements:
            G.add_node(elem.id)
        
        for rel in relations:
            G.add_edge(rel.from_element, rel.to_element)
        
        # Detect communities
        communities = nx.community.greedy_modularity_communities(G)
        
        # Create groups
        for i, community in enumerate(communities):
            groups.append({
                'id': f'group_{i}',
                'elements': list(community),
                'label': self._generate_group_label(community, elements)
            })
        
        return groups

    def _generate_group_label(
        self,
        community: set,
        elements: List[ConceptElement]
    ) -> str:
        """Generate a descriptive label for a group of elements"""
        # Get types of elements in the group
        element_types = [
            elem.type for elem in elements 
            if elem.id in community
        ]
        
        # Count type frequencies
        type_counts = Counter(element_types)
        most_common_type = type_counts.most_common(1)[0][0]
        
        return f"{most_common_type.value.title()} Group"

    def _get_style_for_entity_type(
        self,
        entity_type: EntityType
    ) -> Dict[str, Any]:
        """Get visual style properties for an entity type"""
        style_mapping = {
            EntityType.COMPONENT: {
                'fill_color': '#4A90E2',
                'stroke_color': '#2171C7',
                'stroke_width': 2,
                'opacity': 0.9
            },
            EntityType.ACTOR: {
                'fill_color': '#F5A623',
                'stroke_color': '#E49B1D',
                'stroke_width': 2,
                'opacity': 0.9
            },
            EntityType.STATE: {
                'fill_color': '#7ED321',
                'stroke_color': '#69B61B',
                'stroke_width': 2,
                'opacity': 0.9
            },
            # Add styles for other entity types...
        }
        
        return style_mapping.get(entity_type, {
            'fill_color': '#9B9B9B',
            'stroke_color': '#7B7B7B',
            'stroke_width': 1,
            'opacity': 0.8
        })

    def _get_style_for_relation_type(
        self,
        relation_type: RelationType
    ) -> Dict[str, Any]:
        """Get visual style properties for a relation type"""
        style_mapping = {
            RelationType.SENDS: {
                'line_style': 'solid',
                'arrow_style': 'normal',
                'color': '#4A90E2',
                'width': 2
            },
            RelationType.RECEIVES: {
                'line_style': 'dashed',
                'arrow_style': 'normal',
                'color': '#7ED321',
                'width': 2
            },
            # Add styles for other relation types...
        }
        
        return style_mapping.get(relation_type, {
            'line_style': 'solid',
            'arrow_style': 'normal',
            'color': '#9B9B9B',
            'width': 1
        })

    def _get_animation_type(self, step: ConceptStep) -> str:
        """Determine appropriate animation type for a step"""
        if step.importance_score > 0.8:
            return 'highlight'
        elif len(step.elements_involved) > 2:
            return 'sequence'
        else:
            return 'simple'

    def _generate_animation_sequence(
        self,
        steps: List[ConceptStep]
    ) -> List[Dict[str, Any]]:
        """Generate detailed animation sequence"""
        sequence = []
        
        for step in steps:
            animations = []
            
            # Add pre-step highlighting if important
            if step.importance_score > 0.7:
                animations.append({
                    'type': 'highlight',
                    'elements': step.elements_involved,
                    'duration': 0.5
                })
            
            # Add main step animation
            animations.append({
                'type': self._get_animation_type(step),
                'elements': step.elements_involved,
                'duration': 1.0,
                'timing': step.timing or '2s',
                'effects': self._get_step_effects(step)
            })
            
            sequence.append({
                'step_number': step.order,
                'animations': animations,
                'narration': step.description,
                'pause_after': step.importance_score > 0.7
            })
        
        return sequence

    def _get_step_effects(self, step: ConceptStep) -> List[Dict[str, Any]]:
        """Get special effects for a step"""
        effects = []
        
        if step.importance_score > 0.8:
            effects.append({
                'type': 'glow',
                'color': '#FFD700',
                'duration': 0.5
            })
        
        if len(step.preconditions) > 0:
            effects.append({
                'type': 'fade_in',
                'duration': 0.3
            })
        
        return effects