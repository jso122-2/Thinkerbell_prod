import React, { useState, useEffect, useRef } from 'react';
import { Brain, Wand2, Lightbulb, TrendingUp, ArrowRight, Sparkles, Upload, Download, RefreshCw } from 'lucide-react';

// Semantic Classification Engine
class SemanticEngine {
  constructor() {
    // Semantic anchors - the "brain" of classification
    this.anchors = {
      'Hunch': {
        keywords: ['suspect', 'intuition', 'feeling', 'guess', 'hypothesis', 'theory', 'wonder', 'might', 'could', 'probably'],
        patterns: ['what if', 'i think', 'my gut says', 'feeling like', 'suspicion'],
        description: 'A clever suspicion or idea. Often intuitive and playful, not yet proven.',
        icon: '💡',
        color: 'bg-yellow-100 border-yellow-300'
      },
      'Wisdom': {
        keywords: ['data', 'research', 'study', 'evidence', 'shows', 'proves', 'statistics', 'analysis', 'insight', 'learned'],
        patterns: ['% of', 'research shows', 'data indicates', 'studies prove', 'evidence suggests'],
        description: 'A strategic insight based on data or experience.',
        icon: '📊',
        color: 'bg-blue-100 border-blue-300'
      },
      'Nudge': {
        keywords: ['should', 'recommend', 'suggest', 'action', 'next', 'do', 'try', 'implement', 'consider', 'move'],
        patterns: ['we should', 'recommend', 'next step', 'action item', 'let\'s try'],
        description: 'A recommended action, often subtle or behaviorally informed.',
        icon: '👉',
        color: 'bg-green-100 border-green-300'
      },
      'Spell': {
        keywords: ['magical', 'surprising', 'unexpected', 'creative', 'innovative', 'unique', 'extraordinary', 'amazing'],
        patterns: ['imagine if', 'what if we', 'picture this', 'magical moment', 'surprise twist'],
        description: 'A surprising creative flourish or magical execution idea.',
        icon: '✨',
        color: 'bg-purple-100 border-purple-300'
      }
    };
  }

  // Semantic classification using keyword/pattern matching + scoring
  classifySentence(sentence) {
    const scores = {};
    const lowerSentence = sentence.toLowerCase();
    
    Object.entries(this.anchors).forEach(([category, anchor]) => {
      let score = 0;
      
      // Keyword matching
      anchor.keywords.forEach(keyword => {
        if (lowerSentence.includes(keyword)) {
          score += 1;
        }
      });
      
      // Pattern matching (higher weight)
      anchor.patterns.forEach(pattern => {
        if (lowerSentence.includes(pattern)) {
          score += 2;
        }
      });
      
      scores[category] = score;
    });
    
    // Return best match or 'Hunch' as default
    const bestMatch = Object.keys(scores).reduce((a, b) => scores[a] > scores[b] ? a : b);
    return scores[bestMatch] > 0 ? bestMatch : 'Hunch';
  }

  // Route multiple sentences into buckets
  routeContent(content) {
    const sentences = this.splitIntoSentences(content);
    const routed = {
      'Hunch': [],
      'Wisdom': [],
      'Nudge': [],
      'Spell': []
    };
    
    sentences.forEach(sentence => {
      if (sentence.trim().length > 10) { // Filter out very short sentences
        const category = this.classifySentence(sentence);
        routed[category].push(sentence.trim());
      }
    });
    
    return routed;
  }

  splitIntoSentences(text) {
    return text.split(/[.!?]+/).filter(s => s.trim().length > 0);
  }
}

// Template Renderer
class ThinkerbellRenderer {
  constructor() {
    this.templates = {
      slide_deck: {
        renderSection: (category, content, anchor) => `
## ${anchor.icon} ${category}

${content.map(item => `- ${item}`).join('\n')}

---
        `,
        renderHeader: (title) => `# ${title}\n\n*Crafted with intelligence by Thinkerbell*\n\n---\n\n`
      },
      strategy_doc: {
        renderSection: (category, content, anchor) => `
### ${anchor.icon} ${category}

${content.map(item => `**${item}**`).join('\n\n')}

`,
        renderHeader: (title) => `# ${title}\n\n## Strategic Overview\n\n`
      },
      creative_brief: {
        renderSection: (category, content, anchor) => `
**${category}**: ${content.join(' ')}

`,
        renderHeader: (title) => `# Creative Brief: ${title}\n\n`
      }
    };
  }

  render(routedContent, templateType = 'slide_deck', title = 'Thinkerbell Brief') {
    const template = this.templates[templateType];
    let output = template.renderHeader(title);
    
    Object.entries(routedContent).forEach(([category, content]) => {
      if (content.length > 0) {
        const anchor = new SemanticEngine().anchors[category];
        output += template.renderSection(category, content, anchor);
      }
    });
    
    return output;
  }
}

// Main Component
export default function SemanticThinkerbellEngine() {
  const [rawInput, setRawInput] = useState(`Our new campaign strategy needs refinement. I suspect that Gen Z responds better to authentic storytelling than polished marketing. Research shows that 73% of consumers prefer brands that feel genuine. We should focus on user-generated content and real customer stories. Imagine if we created a platform where customers become the storytellers themselves.`);
  
  const [routedContent, setRoutedContent] = useState({});
  const [selectedTemplate, setSelectedTemplate] = useState('slide_deck');
  const [outputContent, setOutputContent] = useState('');
  const [isProcessing, setIsProcessing] = useState(false);
  
  const semanticEngine = useRef(new SemanticEngine());
  const renderer = useRef(new ThinkerbellRenderer());

  // Real-time processing
  useEffect(() => {
    if (rawInput.trim()) {
      setIsProcessing(true);
      
      // Simulate processing delay for better UX
      const timer = setTimeout(() => {
        const routed = semanticEngine.current.routeContent(rawInput);
        setRoutedContent(routed);
        
        const rendered = renderer.current.render(routed, selectedTemplate, 'Smart Campaign Strategy');
        setOutputContent(rendered);
        
        setIsProcessing(false);
      }, 300);
      
      return () => clearTimeout(timer);
    }
  }, [rawInput, selectedTemplate]);

  const handleExport = () => {
    const blob = new Blob([outputContent], { type: 'text/markdown' });
    const url = URL.createObjectURL(blob);
    const a = document.createElement('a');
    a.href = url;
    a.download = `thinkerbell-brief-${Date.now()}.md`;
    a.click();
    URL.revokeObjectURL(url);
  };

  const exampleInputs = [
    {
      name: "Campaign Strategy",
      content: "Our new campaign strategy needs refinement. I suspect that Gen Z responds better to authentic storytelling than polished marketing. Research shows that 73% of consumers prefer brands that feel genuine. We should focus on user-generated content and real customer stories. Imagine if we created a platform where customers become the storytellers themselves."
    },
    {
      name: "Product Launch",
      content: "The product launch feels risky without more user research. Studies indicate that 60% of product failures happen due to poor market fit. We recommend conducting focus groups before the big reveal. Picture this: a secret beta community that becomes our launch evangelists."
    },
    {
      name: "Social Strategy", 
      content: "Social media engagement is declining across platforms. I think we're posting too much corporate content. Analytics show that behind-the-scenes content gets 3x more engagement. We should shift to employee takeovers and office culture posts. What if each employee became a mini-influencer for the brand?"
    }
  ];

  return (
    <div className="min-h-screen bg-gradient-to-br from-purple-50 to-pink-50 p-6">
      <div className="max-w-7xl mx-auto">
        {/* Header */}
        <div className="text-center mb-8">
          <div className="flex items-center justify-center gap-3 mb-4">
            <Brain className="w-8 h-8 text-purple-600" />
            <h1 className="text-3xl font-bold bg-gradient-to-r from-purple-600 to-pink-600 bg-clip-text text-transparent">
              Semantic Thinkerbell Engine
            </h1>
            <Sparkles className="w-8 h-8 text-pink-600" />
          </div>
          <p className="text-gray-600 max-w-2xl mx-auto">
            AI-powered content classification that automatically routes your ideas into Thinkerbell's brand framework: 
            Hunches, Wisdom, Nudges, and Spells.
          </p>
        </div>

        <div className="grid lg:grid-cols-2 gap-8">
          {/* Input Section */}
          <div className="space-y-6">
            <div>
              <label className="block text-sm font-medium text-gray-700 mb-2">
                Raw Content Input
              </label>
              <textarea
                value={rawInput}
                onChange={(e) => setRawInput(e.target.value)}
                placeholder="Paste your raw strategy notes, campaign ideas, or any unstructured content..."
                className="w-full h-64 p-4 border border-gray-300 rounded-lg focus:ring-2 focus:ring-purple-500 focus:border-transparent resize-none"
              />
            </div>

            {/* Example Inputs */}
            <div>
              <label className="block text-sm font-medium text-gray-700 mb-2">
                Try Examples
              </label>
              <div className="grid gap-2">
                {exampleInputs.map((example, idx) => (
                  <button
                    key={idx}
                    onClick={() => setRawInput(example.content)}
                    className="text-left p-3 bg-white border border-gray-200 rounded-lg hover:border-purple-300 hover:bg-purple-50 transition-colors"
                  >
                    <div className="font-medium text-gray-800">{example.name}</div>
                    <div className="text-sm text-gray-500 truncate">{example.content.substring(0, 60)}...</div>
                  </button>
                ))}
              </div>
            </div>

            {/* Template Selection */}
            <div>
              <label className="block text-sm font-medium text-gray-700 mb-2">
                Output Template
              </label>
              <select
                value={selectedTemplate}
                onChange={(e) => setSelectedTemplate(e.target.value)}
                className="w-full p-3 border border-gray-300 rounded-lg focus:ring-2 focus:ring-purple-500"
              >
                <option value="slide_deck">Slide Deck</option>
                <option value="strategy_doc">Strategy Document</option>
                <option value="creative_brief">Creative Brief</option>
              </select>
            </div>
          </div>

          {/* Output Section */}
          <div className="space-y-6">
            {/* Semantic Classification Preview */}
            <div>
              <label className="block text-sm font-medium text-gray-700 mb-2">
                Semantic Classification
                {isProcessing && <RefreshCw className="inline w-4 h-4 ml-2 animate-spin" />}
              </label>
              <div className="grid grid-cols-2 gap-3">
                {Object.entries(semanticEngine.current.anchors).map(([category, anchor]) => {
                  const content = routedContent[category] || [];
                  return (
                    <div key={category} className={`p-3 rounded-lg border-2 ${anchor.color}`}>
                      <div className="flex items-center gap-2 mb-2">
                        <span className="text-lg">{anchor.icon}</span>
                        <span className="font-medium text-sm">{category}</span>
                      </div>
                      <div className="text-xs text-gray-600 mb-2">{content.length} items</div>
                      <div className="text-xs text-gray-500 max-h-16 overflow-y-auto">
                        {content.slice(0, 2).map((item, idx) => (
                          <div key={idx} className="truncate">{item}</div>
                        ))}
                        {content.length > 2 && <div>+{content.length - 2} more...</div>}
                      </div>
                    </div>
                  );
                })}
              </div>
            </div>

            {/* Formatted Output */}
            <div>
              <div className="flex items-center justify-between mb-2">
                <label className="block text-sm font-medium text-gray-700">
                  Formatted Output
                </label>
                <button
                  onClick={handleExport}
                  disabled={!outputContent}
                  className="flex items-center gap-2 px-3 py-1 text-sm bg-purple-600 text-white rounded-lg hover:bg-purple-700 disabled:opacity-50 disabled:cursor-not-allowed"
                >
                  <Download className="w-4 h-4" />
                  Export
                </button>
              </div>
              <div className="bg-gray-900 text-green-400 p-4 rounded-lg font-mono text-sm h-96 overflow-y-auto">
                <pre className="whitespace-pre-wrap">{outputContent || 'Formatted content will appear here...'}</pre>
              </div>
            </div>
          </div>
        </div>

        {/* Processing Status */}
        {isProcessing && (
          <div className="fixed bottom-4 right-4 bg-purple-600 text-white px-4 py-2 rounded-lg shadow-lg">
            <div className="flex items-center gap-2">
              <RefreshCw className="w-4 h-4 animate-spin" />
              Processing with semantic intelligence...
            </div>
          </div>
        )}
      </div>
    </div>
  );
}