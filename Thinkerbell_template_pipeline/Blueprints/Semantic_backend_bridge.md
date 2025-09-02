# Semantic Backend Bridge for Thinkerbell Engine
# Implements the sophisticated sentence transformer approach from the design doc

import json
import yaml
import numpy as np
from typing import Dict, List, Any
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class SemanticBrain:
    """
    The core semantic classification engine using sentence transformers
    """
    
    def __init__(self, model_name: str = "all-MiniLM-L6-v2"):
        logger.info(f"Loading semantic model: {model_name}")
        self.model = SentenceTransformer(model_name)
        self.anchors = self._load_anchors()
        self.anchor_embeddings = self._precompute_anchor_embeddings()
        
    def _load_anchors(self):
        """Load semantic anchors for classification"""
        return {
            "Hunch": """
                We think the real reason people avoid flossing is emotional, not rational.
                I suspect customers buy organic food to feel virtuous, not because they taste better.
                My gut feeling is that people check social media when they're anxious about real life.
                What if the reason meetings run long isn't poor planning, but fear of making decisions?
                I believe most product returns happen because buyers had unrealistic expectations.
                There's probably a connection between clutter in homes and overwhelm at work.
            """,
            "Wisdom": """
                Research shows that 68% of customers abandon carts due to unexpected shipping costs.
                Data from 50,000 transactions reveals that personalized recommendations increase sales by 35%.
                Studies prove that customers who interact with chatbots first are 40% more likely to purchase.
                Analytics indicate that email open rates drop 25% when subject lines exceed 50 characters.
                Evidence from A/B testing demonstrates that red buttons outperform blue by 21%.
                Statistics confirm that 73% of millennials are willing to pay more for sustainable products.
            """,
            "Nudge": """
                We should start sending cart abandonment emails within 3 hours instead of 24.
                I recommend placing testimonials right above the checkout button to reduce hesitation.
                Let's try offering a small discount to first-time visitors who spend more than 2 minutes browsing.
                The next step is implementing social proof notifications showing recent purchases.
                We need to redesign the pricing page to highlight the most popular plan.
                Consider adding a progress bar to show how close customers are to free shipping.
            """,
            "Spell": """
                Imagine if your shopping cart could predict what you'll need before you run out.
                Picture a checkout experience so smooth it feels like magic - no forms, no friction.
                What if we created a loyalty program that rewards customers for bringing friends?
                Envision packaging that transforms into something useful after the product is gone.
                Picture this: a subscription service that learns your preferences and surprises you.
                Imagine if every customer interaction felt like talking to your most helpful friend.
            """
        }
    
    def _precompute_anchor_embeddings(self):
        """Pre-compute embeddings for all anchor descriptions"""
        logger.info("Computing anchor embeddings...")
        anchor_texts = list(self.anchors.values())
        embeddings = self.model.encode(anchor_texts)
        return {category: embeddings[i] for i, category in enumerate(self.anchors.keys())}
    
    def classify_sentence(self, sentence: str, threshold: float = 0.3) -> tuple:
        """
        Classify a single sentence using semantic similarity
        Returns: (category, confidence_score)
        """
        sentence_embedding = self.model.encode([sentence])
        
        similarities = {}
        for category, anchor_embedding in self.anchor_embeddings.items():
            similarity = cosine_similarity(sentence_embedding, [anchor_embedding])[0][0]
            similarities[category] = similarity
        
        best_category = max(similarities.keys(), key=lambda k: similarities[k])
        best_score = similarities[best_category]
        
        # If confidence is too low, default to "Hunch"
        if best_score < threshold:
            return "Hunch", best_score
        
        return best_category, best_score
    
    def route_content(self, content: str, confidence_threshold: float = 0.3) -> Dict[str, List[Dict]]:
        """
        Route content into semantic buckets with confidence scores
        """
        sentences = self._split_sentences(content)
        routed = {category: [] for category in self.anchors.keys()}
        
        for sentence in sentences:
            if len(sentence.strip()) < 10:  # Skip very short sentences
                continue
                
            category, confidence = self.classify_sentence(sentence, confidence_threshold)
            
            routed[category].append({
                "text": sentence.strip(),
                "confidence": float(confidence),
                "embedding": self.model.encode([sentence])[0].tolist()  # Store for future learning
            })
        
        # Sort by confidence within each category
        for category in routed:
            routed[category].sort(key=lambda x: x["confidence"], reverse=True)
        
        return routed
    
    def _split_sentences(self, text: str) -> List[str]:
        """Smart sentence splitting that preserves meaning"""
        import re
        # Split on sentence endings but preserve meaning
        sentences = re.split(r'[.!?]+', text)
        return [s.strip() for s in sentences if s.strip()]
    
    def explain_classification(self, sentence: str) -> Dict[str, Any]:
        """
        Provide detailed explanation of why a sentence was classified a certain way
        """
        category, confidence = self.classify_sentence(sentence)
        
        # Get similarities to all categories
        sentence_embedding = self.model.encode([sentence])
        all_similarities = {}
        
        for cat, anchor_embedding in self.anchor_embeddings.items():
            similarity = cosine_similarity(sentence_embedding, [anchor_embedding])[0][0]
            all_similarities[cat] = float(similarity)
        
        return {
            "sentence": sentence,
            "predicted_category": category,
            "confidence": float(confidence),
            "all_similarities": all_similarities,
            "reasoning": f"Classified as '{category}' with {confidence:.2%} confidence"
        }

class ThinkerbellTemplateEngine:
    """
    Template engine that works with the semantic brain
    """
    
    def __init__(self):
        self.templates = self._load_templates()
    
    def _load_templates(self) -> Dict[str, Dict]:
        """Load formatting templates"""
        return {
            "slide_deck": {
                "header": "# {title}\n\n*Intelligently crafted by Thinkerbell's semantic engine*\n\n---\n\n",
                "section": "## {icon} {category}\n\n{content}\n\n---\n\n",
                "footer": "*Powered by AI semantic analysis*\n",
                "icons": {"Hunch": "💡", "Wisdom": "📊", "Nudge": "👉", "Spell": "✨"}
            },
            "strategy_doc": {
                "header": "# Strategic Brief: {title}\n\n## Executive Summary\n\n",
                "section": "### {icon} {category}\n\n{content}\n\n",
                "footer": "---\n\n*Document generated with semantic intelligence | {date}*\n",
                "icons": {"Hunch": "🎯", "Wisdom": "🧠", "Nudge": "⚡", "Spell": "🚀"}
            },
            "measurement_framework": {
                "header": "# Measurement Framework: {title}\n\n",
                "section": "**{category}**\n\n{content}\n\n",
                "footer": "---\n\n*Framework built with AI-driven content analysis*\n",
                "icons": {"Hunch": "📈", "Wisdom": "📊", "Nudge": "🎯", "Spell": "⭐"}
            }
        }
    
    def render(self, routed_content: Dict, template_type: str = "slide_deck", 
               title: str = "Thinkerbell Brief") -> str:
        """Render routed content using specified template"""
        
        template = self.templates.get(template_type, self.templates["slide_deck"])
        output = template["header"].format(title=title)
        
        for category, items in routed_content.items():
            if items:  # Only render categories with content
                icon = template["icons"].get(category, "▶️")
                
                # Format content items
                content_lines = []
                for item in items:
                    confidence_indicator = "🔥" if item["confidence"] > 0.7 else "🔍" if item["confidence"] > 0.5 else "💭"
                    content_lines.append(f"- {confidence_indicator} {item['text']}")
                
                content = "\n".join(content_lines)
                
                output += template["section"].format(
                    icon=icon,
                    category=category,
                    content=content
                )
        
        output += template["footer"].format(date="2025")
        return output

class SemanticPipeline:
    """
    Main pipeline that coordinates semantic analysis and formatting
    """
    
    def __init__(self, model_name: str = "all-MiniLM-L6-v2"):
        self.brain = SemanticBrain(model_name)
        self.formatter = ThinkerbellTemplateEngine()
        self.learning_data = []  # Store for future fine-tuning
    
    def process(self, raw_content: str, template_type: str = "slide_deck", 
                title: str = "AI-Generated Brief", confidence_threshold: float = 0.3) -> Dict[str, Any]:
        """
        Full pipeline: analyze → route → format
        """
        logger.info(f"Processing content with semantic pipeline...")
        
        # Semantic analysis
        routed_content = self.brain.route_content(raw_content, confidence_threshold)
        
        # Format output
        formatted_output = self.formatter.render(routed_content, template_type, title)
        
        # Collect analytics
        analytics = self._generate_analytics(routed_content)
        
        result = {
            "routed_content": routed_content,
            "formatted_output": formatted_output,
            "analytics": analytics,
            "metadata": {
                "template_type": template_type,
                "title": title,
                "confidence_threshold": confidence_threshold,
                "total_sentences": sum(len(items) for items in routed_content.values())
            }
        }
        
        # Store for learning
        self.learning_data.append({
            "input": raw_content,
            "output": result,
            "timestamp": "2025-07-22"  # In real implementation, use datetime
        })
        
        return result
    
    def _generate_analytics(self, routed_content: Dict) -> Dict[str, Any]:
        """Generate insights about the classification"""
        total_items = sum(len(items) for items in routed_content.values())
        
        if total_items == 0:
            return {"error": "No content to analyze"}
        
        distribution = {}
        avg_confidence = {}
        
        for category, items in routed_content.items():
            distribution[category] = {
                "count": len(items),
                "percentage": len(items) / total_items * 100
            }
            
            if items:
                avg_confidence[category] = sum(item["confidence"] for item in items) / len(items)
            else:
                avg_confidence[category] = 0
        
        return {
            "distribution": distribution,
            "average_confidence": avg_confidence,
            "dominant_category": max(distribution.keys(), key=lambda k: distribution[k]["count"]),
            "total_sentences": total_items
        }
    
    def explain_sentence(self, sentence: str) -> Dict[str, Any]:
        """Explain why a sentence was classified a certain way"""
        return self.brain.explain_classification(sentence)
    
    def get_learning_data(self) -> List[Dict]:
        """Return stored data for fine-tuning"""
        return self.learning_data
    
    def save_session(self, filepath: str):
        """Save learning data for future training"""
        with open(filepath, 'w') as f:
            json.dump(self.learning_data, f, indent=2)
        logger.info(f"Session data saved to {filepath}")

# Example usage and testing
if __name__ == "__main__":
    # Initialize pipeline
    pipeline = SemanticPipeline()
    
    # Test content
    test_content = """
    Our new campaign strategy needs refinement. I suspect that Gen Z responds better to 
    authentic storytelling than polished marketing. Research shows that 73% of consumers 
    prefer brands that feel genuine. We should focus on user-generated content and real 
    customer stories. Imagine if we created a platform where customers become the 
    storytellers themselves.
    """
    
    print("🧠 Processing with Semantic Pipeline...")
    result = pipeline.process(
        raw_content=test_content,
        template_type="slide_deck",
        title="Gen Z Campaign Strategy"
    )
    
    print("\n📊 Analytics:")
    print(json.dumps(result["analytics"], indent=2))
    
    print("\n📝 Formatted Output:")
    print(result["formatted_output"])
    
    print("\n🔍 Explanation for sample sentence:")
    sample_sentence = "Research shows that 73% of consumers prefer brands that feel genuine."
    explanation = pipeline.explain_sentence(sample_sentence)
    print(json.dumps(explanation, indent=2))
    
    # Save learning data
    pipeline.save_session("thinkerbell_learning_session.json")

# API Integration
def create_api_endpoint():
    """
    Example FastAPI endpoint for integration
    """
    from fastapi import FastAPI, HTTPException
    from pydantic import BaseModel
    
    app = FastAPI(title="Thinkerbell Semantic Engine API")
    pipeline = SemanticPipeline()
    
    class ProcessRequest(BaseModel):
        content: str
        template_type: str = "slide_deck"
        title: str = "AI Brief"
        confidence_threshold: float = 0.3
    
    @app.post("/process")
    async def process_content(request: ProcessRequest):
        try:
            result = pipeline.process(
                raw_content=request.content,
                template_type=request.template_type,
                title=request.title,
                confidence_threshold=request.confidence_threshold
            )
            return result
        except Exception as e:
            raise HTTPException(status_code=500, detail=str(e))
    
    @app.post("/explain")
    async def explain_sentence(sentence: str):
        try:
            return pipeline.explain_sentence(sentence)
        except Exception as e:
            raise HTTPException(status_code=500, detail=str(e))
    
    return app

# For Cursor Integration
class CursorIntegration:
    """
    Integration layer for Cursor IDE
    """
    
    def __init__(self):
        self.pipeline = SemanticPipeline()
    
    def process_file(self, filepath: str, output_format: str = "markdown") -> str:
        """Process a file and return formatted output"""
        with open(filepath, 'r') as f:
            content = f.read()
        
        result = self.pipeline.process(content)
        return result["formatted_output"]
    
    def process_selection(self, selected_text: str, template_type: str = "slide_deck") -> str:
        """Process selected text in Cursor"""
        result = self.pipeline.process(selected_text, template_type)
        return result["formatted_output"]
    
    def get_suggestions(self, partial_content: str) -> Dict[str, List[str]]:
        """Get content suggestions based on partial input"""
        routed = self.pipeline.brain.route_content(partial_content)
        
        suggestions = {}
        for category, items in routed.items():
            if not items:
                # Suggest content for empty categories
                suggestions[category] = [
                    f"Consider adding a {category.lower()} about...",
                    f"What {category.lower()} supports your main point?",
                    f"Include a {category.lower()} that..."
                ]
        
        return suggestions