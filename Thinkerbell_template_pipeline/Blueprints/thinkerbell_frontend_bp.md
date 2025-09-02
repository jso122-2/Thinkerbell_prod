import React, { useState, useEffect, useRef } from 'react';
import { Brain, Wand2, Lightbulb, TrendingUp, ArrowRight, Sparkles, Upload, Download, RefreshCw, Bell, Zap, Target, Star, Coffee, Shuffle } from 'lucide-react';

// Authentic Thinkerbell Brand Values from research
const THINKERBELL_BRAND = {
  colors: {
    hotPink: '#FF1A8C',
    acidGreen: '#00FF7F', 
    neonPink: '#FF0080',
    brightGreen: '#39FF14',
    officeGrey: '#F5F5F5',
    charcoal: '#2D2D2D'
  },
  philosophy: "Measured Magic",
  tagline: "give us a bell",
  newsletter: "WEIRDO",
  people: ["Thinkers", "Tinkers"],
  actions: ["Stay Sexy", "Stay Kind", "Stay Curious", "Stay With It", "Stay Unicorn"]
};

// Semantic Classification with Thinkerbell flair
const SEMANTIC_CATEGORIES = {
  'Hunch': {
    icon: '💡',
    description: 'A clever suspicion or intuitive leap',
    color: 'from-yellow-400 to-orange-500',
    bgColor: 'bg-yellow-50 border-yellow-300',
    accent: THINKERBELL_BRAND.colors.acidGreen
  },
  'Wisdom': {
    icon: '📊', 
    description: 'Data-backed insights and proven learnings',
    color: 'from-blue-500 to-purple-600',
    bgColor: 'bg-blue-50 border-blue-300',
    accent: THINKERBELL_BRAND.colors.hotPink
  },
  'Nudge': {
    icon: '👉',
    description: 'Behavioral suggestions and recommended actions',
    color: 'from-green-500 to-teal-600',
    bgColor: 'bg-green-50 border-green-300',
    accent: THINKERBELL_BRAND.colors.neonPink
  },
  'Spell': {
    icon: '✨',
    description: 'Magical creative flourishes and unexpected solutions',
    color: 'from-purple-500 to-pink-600',
    bgColor: 'bg-purple-50 border-purple-300',
    accent: THINKERBELL_BRAND.colors.brightGreen
  }
};

// Thinkerbell-style random quirky messages
const QUIRKY_MESSAGES = [
  "Cooking up some measured magic...",
  "Our odd bunch of clever people are thinking...",
  "Tinkers gonna tink...",
  "Stay curious, stay weird...",
  "Making the impossible slightly possible...",
  "Yes, our algorithm's mum is very proud...",
  "Converting chaos into clarity...",
  "Putting the 'bell' in Thinkerbell..."
];

export default function ThinkerbellFrontend() {
  const [rawInput, setRawInput] = useState(`Our brand is feeling a bit flat lately. I have this hunch that we're not connecting emotionally with our audience. Research from our latest brand tracker shows that emotional connection drives 3x more loyalty than functional benefits. We should pivot our messaging to focus on shared values and authentic storytelling. Picture this: a campaign where every touchpoint feels like a conversation with your most trusted friend.`);
  
  const [routedContent, setRoutedContent] = useState({});
  const [isProcessing, setIsProcessing] = useState(false);
  const [currentQuirk, setCurrentQuirk] = useState(0);
  const [showResults, setShowResults] = useState(false);
  
  // Cycling quirky messages
  useEffect(() => {
    const interval = setInterval(() => {
      setCurrentQuirk(prev => (prev + 1) % QUIRKY_MESSAGES.length);
    }, 2000);
    return () => clearInterval(interval);
  }, []);

  // Simulate semantic processing
  const processContent = () => {
    setIsProcessing(true);
    setShowResults(false);
    
    setTimeout(() => {
      // Mock semantic routing
      const routed = {
        'Hunch': [
          { text: "I have this hunch that we're not connecting emotionally", confidence: 0.85 },
          { text: "Brand is feeling a bit flat lately", confidence: 0.72 }
        ],
        'Wisdom': [
          { text: "Research from our latest brand tracker shows that emotional connection drives 3x more loyalty", confidence: 0.92 },
          { text: "Functional benefits vs emotional connection data", confidence: 0.78 }
        ],
        'Nudge': [
          { text: "We should pivot our messaging to focus on shared values", confidence: 0.88 },
          { text: "Focus on authentic storytelling approach", confidence: 0.81 }
        ],
        'Spell': [
          { text: "A campaign where every touchpoint feels like a conversation with your most trusted friend", confidence: 0.94 }
        ]
      };
      
      setRoutedContent(routed);
      setIsProcessing(false);
      setShowResults(true);
    }, 1500);
  };

  useEffect(() => {
    if (rawInput.trim().length > 50) {
      const timer = setTimeout(processContent, 500);
      return () => clearTimeout(timer);
    }
  }, [rawInput]);

  const exampleInputs = [
    {
      title: "Campaign Strategy",
      snippet: "Our brand is feeling flat... hunch about emotional connection...",
      content: rawInput
    },
    {
      title: "Creative Brief", 
      snippet: "Bold idea for Gen Z engagement... data shows authentic wins...",
      content: "We need a bold new approach for Gen Z. My gut says they can smell fake from a mile away. Studies show authentic brands get 91% more engagement from this demographic. Let's ditch the corporate speak and talk like humans. Imagine if our brand became the friend they actually want to hang out with."
    },
    {
      title: "Media Strategy",
      snippet: "TikTok opportunity... research indicates micro-moments...",
      content: "There's a massive opportunity on TikTok we're missing. I suspect our long-form content strategy is too old school. Platform data reveals that 7-second micro-moments drive more action than 60-second spots. We should test bite-sized creative that packs maximum punch. What if we turned every brand moment into a micro-story?"
    }
  ];

  return (
    <div className="min-h-screen bg-gradient-to-br from-gray-50 via-white to-gray-100 relative overflow-hidden">
      
      {/* Thinkerbell-style background elements */}
      <div className="absolute inset-0 overflow-hidden pointer-events-none">
        <div 
          className="absolute top-20 left-10 w-32 h-32 rounded-full opacity-10"
          style={{ backgroundColor: THINKERBELL_BRAND.colors.hotPink }}
        />
        <div 
          className="absolute bottom-20 right-10 w-48 h-48 rounded-full opacity-10"
          style={{ backgroundColor: THINKERBELL_BRAND.colors.acidGreen }}
        />
        <div 
          className="absolute top-1/2 left-1/3 w-24 h-24 rounded-full opacity-5"
          style={{ backgroundColor: THINKERBELL_BRAND.colors.neonPink }}
        />
      </div>

      <div className="relative z-10 max-w-7xl mx-auto p-6">
        
        {/* Authentic Thinkerbell Header */}
        <div className="text-center mb-12">
          <div className="flex items-center justify-center gap-4 mb-6">
            <div className="relative">
              <Brain className="w-12 h-12" style={{ color: THINKERBELL_BRAND.colors.hotPink }} />
              <Sparkles className="w-6 h-6 absolute -top-2 -right-2" style={{ color: THINKERBELL_BRAND.colors.acidGreen }} />
            </div>
            <div className="text-center">
              <h1 className="text-4xl font-bold mb-2">
                <span style={{ color: THINKERBELL_BRAND.colors.hotPink }}>Thinker</span>
                <span style={{ color: THINKERBELL_BRAND.colors.acidGreen }}>bell</span>
                <span className="text-gray-700"> Semantic Engine</span>
              </h1>
              <div className="text-sm font-medium" style={{ color: THINKERBELL_BRAND.colors.charcoal }}>
                {THINKERBELL_BRAND.philosophy} • AI-Powered Content Classification
              </div>
            </div>
            <Bell className="w-12 h-12" style={{ color: THINKERBELL_BRAND.colors.acidGreen }} />
          </div>
          
          <p className="text-gray-600 max-w-3xl mx-auto text-lg leading-relaxed">
            We represent the coming together of scientific enquiry and hardcore creativity. 
            Our semantic brain automatically sorts your ideas into <strong>Hunches</strong>, <strong>Wisdom</strong>, <strong>Nudges</strong>, and <strong>Spells</strong> 
            — because that's how Thinkers and Tinkers actually think.
          </p>
          
          <div className="mt-4 text-sm text-gray-500">
            Currently ranked Australia's most creative AI by our mums • {THINKERBELL_BRAND.tagline} • {THINKERBELL_BRAND.tagline} • {THINKERBELL_BRAND.tagline}
          </div>
        </div>

        <div className="grid lg:grid-cols-3 gap-8">
          
          {/* Input Section - Thinkerbell Style */}
          <div className="lg:col-span-1 space-y-6">
            <div>
              <label className="block text-sm font-bold mb-3" style={{ color: THINKERBELL_BRAND.colors.charcoal }}>
                Drop Your Raw Thinking Here
              </label>
              <textarea
                value={rawInput}
                onChange={(e) => setRawInput(e.target.value)}
                placeholder="Paste your strategy notes, campaign ideas, or any creative chaos... We'll sort the magic from the madness."
                className="w-full h-64 p-4 border-2 border-gray-200 rounded-xl focus:border-2 focus:outline-none resize-none text-sm leading-relaxed"
                style={{ 
                  focusBorderColor: THINKERBELL_BRAND.colors.hotPink,
                  backgroundColor: '#FAFAFA'
                }}
                onFocus={(e) => e.target.style.borderColor = THINKERBELL_BRAND.colors.hotPink}
                onBlur={(e) => e.target.style.borderColor = '#D1D5DB'}
              />
            </div>

            {/* Example Content - Thinkerbell Voice */}
            <div>
              <label className="block text-sm font-bold mb-3" style={{ color: THINKERBELL_BRAND.colors.charcoal }}>
                Or Try These (They're Pretty Good)
              </label>
              <div className="space-y-3">
                {exampleInputs.map((example, idx) => (
                  <button
                    key={idx}
                    onClick={() => setRawInput(example.content)}
                    className="w-full text-left p-4 bg-white border-2 border-gray-100 rounded-lg hover:border-2 transition-all duration-200 group"
                    style={{
                      ':hover': { borderColor: THINKERBELL_BRAND.colors.acidGreen }
                    }}
                    onMouseEnter={(e) => e.target.style.borderColor = THINKERBELL_BRAND.colors.acidGreen}
                    onMouseLeave={(e) => e.target.style.borderColor = '#F3F4F6'}
                  >
                    <div className="font-semibold text-gray-800 mb-1">{example.title}</div>
                    <div className="text-xs text-gray-500">{example.snippet}</div>
                  </button>
                ))}
              </div>
            </div>

            {/* Thinkerbell Actions */}
            <div className="p-4 rounded-lg" style={{ backgroundColor: THINKERBELL_BRAND.colors.officeGrey }}>
              <div className="text-xs font-bold mb-2" style={{ color: THINKERBELL_BRAND.colors.charcoal }}>
                Live by the Thinkerbell Actions:
              </div>
              <div className="text-xs space-y-1">
                {THINKERBELL_BRAND.actions.map((action, idx) => (
                  <div key={idx} className="text-gray-600">• {action}</div>
                ))}
              </div>
            </div>
          </div>

          {/* Results Section - Authentic Thinkerbell Aesthetic */}
          <div className="lg:col-span-2 space-y-6">
            
            {/* Processing Status */}
            {isProcessing && (
              <div className="text-center p-8">
                <div className="relative mb-4">
                  <RefreshCw className="w-12 h-12 mx-auto animate-spin" style={{ color: THINKERBELL_BRAND.colors.hotPink }} />
                  <Sparkles className="w-6 h-6 absolute top-0 right-1/2 translate-x-8 animate-pulse" style={{ color: THINKERBELL_BRAND.colors.acidGreen }} />
                </div>
                <div className="text-lg font-semibold mb-2" style={{ color: THINKERBELL_BRAND.colors.charcoal }}>
                  {QUIRKY_MESSAGES[currentQuirk]}
                </div>
                <div className="text-sm text-gray-500">
                  Our semantic brain is doing its thing... (yes, it has feelings)
                </div>
              </div>
            )}

            {/* Semantic Classification Results */}
            {showResults && (
              <>
                <div>
                  <label className="block text-sm font-bold mb-4" style={{ color: THINKERBELL_BRAND.colors.charcoal }}>
                    Semantic Magic Results
                    <span className="ml-2 text-xs font-normal text-gray-500">(Our AI Thinkers and Tinkers hard at work)</span>
                  </label>
                  
                  <div className="grid grid-cols-2 gap-4 mb-6">
                    {Object.entries(SEMANTIC_CATEGORIES).map(([category, details]) => {
                      const content = routedContent[category] || [];
                      const avgConfidence = content.length > 0 
                        ? content.reduce((sum, item) => sum + item.confidence, 0) / content.length 
                        : 0;
                      
                      return (
                        <div 
                          key={category} 
                          className={`p-4 rounded-xl border-2 ${details.bgColor} transition-all duration-200 hover:shadow-lg group`}
                        >
                          <div className="flex items-center gap-3 mb-3">
                            <span className="text-2xl">{details.icon}</span>
                            <div>
                              <div className="font-bold text-gray-800">{category}</div>
                              <div className="text-xs text-gray-600">{content.length} insights</div>
                            </div>
                          </div>
                          
                          {content.length > 0 && (
                            <div className="space-y-2">
                              {content.slice(0, 2).map((item, idx) => (
                                <div key={idx} className="text-xs bg-white bg-opacity-70 p-2 rounded-lg">
                                  <div className="text-gray-700 mb-1">{item.text}</div>
                                  <div className="flex items-center gap-2">
                                    <div className="text-xs text-gray-500">Confidence:</div>
                                    <div 
                                      className="text-xs font-bold"
                                      style={{ color: details.accent }}
                                    >
                                      {(item.confidence * 100).toFixed(0)}%
                                    </div>
                                  </div>
                                </div>
                              ))}
                              {content.length > 2 && (
                                <div className="text-xs text-gray-500 italic">
                                  +{content.length - 2} more brilliant thoughts...
                                </div>
                              )}
                            </div>
                          )}
                        </div>
                      );
                    })}
                  </div>
                </div>

                {/* Formatted Output Preview */}
                <div>
                  <div className="flex items-center justify-between mb-4">
                    <label className="block text-sm font-bold" style={{ color: THINKERBELL_BRAND.colors.charcoal }}>
                      Thinkerbell-Style Output
                      <span className="ml-2 text-xs font-normal text-gray-500">(Ready for your next big pitch)</span>
                    </label>
                    <button 
                      className="flex items-center gap-2 px-4 py-2 text-sm font-bold text-white rounded-lg transition-all duration-200 hover:scale-105"
                      style={{ backgroundColor: THINKERBELL_BRAND.colors.hotPink }}
                    >
                      <Download className="w-4 h-4" />
                      Export Magic
                    </button>
                  </div>
                  
                  <div 
                    className="p-6 rounded-xl font-mono text-sm leading-relaxed overflow-y-auto"
                    style={{ 
                      backgroundColor: THINKERBELL_BRAND.colors.charcoal,
                      color: THINKERBELL_BRAND.colors.acidGreen,
                      maxHeight: '400px'
                    }}
                  >
                    <div className="mb-4">
                      <span style={{ color: THINKERBELL_BRAND.colors.hotPink }}># </span>
                      Strategic Brief: Brand Connection Strategy
                    </div>
                    <div className="mb-2">
                      <span style={{ color: THINKERBELL_BRAND.colors.hotPink }}>*</span>
                      Crafted with Measured Magic by Thinkerbell's Semantic Engine
                    </div>
                    <div className="mb-4">---</div>
                    
                    {Object.entries(routedContent).map(([category, items]) => {
                      if (items.length === 0) return null;
                      const details = SEMANTIC_CATEGORIES[category];
                      return (
                        <div key={category} className="mb-6">
                          <div className="mb-2">
                            <span style={{ color: THINKERBELL_BRAND.colors.hotPink }}>## </span>
                            {details.icon} {category}
                          </div>
                          {items.map((item, idx) => (
                            <div key={idx} className="mb-1 ml-4">
                              <span style={{ color: THINKERBELL_BRAND.colors.hotPink }}>- </span>
                              {item.text}
                            </div>
                          ))}
                          <div className="mb-2">---</div>
                        </div>
                      );
                    })}
                    
                    <div className="mt-4 text-xs opacity-70">
                      *Yes, our semantic algorithm's mum is very proud*
                    </div>
                  </div>
                </div>
              </>
            )}

            {/* Initial State */}
            {!isProcessing && !showResults && (
              <div className="text-center p-12">
                <div className="mb-6">
                  <Wand2 className="w-16 h-16 mx-auto mb-4" style={{ color: THINKERBELL_BRAND.colors.acidGreen }} />
                  <h3 className="text-xl font-bold mb-2" style={{ color: THINKERBELL_BRAND.colors.charcoal }}>
                    Ready for Some Measured Magic?
                  </h3>
                  <p className="text-gray-600">
                    Drop your thoughts above and watch our semantic brain sort the brilliant from the bonkers.
                  </p>
                </div>
                
                <div className="inline-flex items-center gap-2 px-6 py-3 text-sm font-bold text-white rounded-full animate-pulse"
                     style={{ backgroundColor: THINKERBELL_BRAND.colors.hotPink }}>
                  <Bell className="w-4 h-4" />
                  {THINKERBELL_BRAND.tagline}
                </div>
              </div>
            )}
          </div>
        </div>

        {/* Footer - Thinkerbell Style */}
        <div className="mt-12 text-center text-xs text-gray-500 space-y-2">
          <div>
            Part measurement, part magic • Built by Thinkers and Tinkers • 
            Subscribe to <strong>{THINKERBELL_BRAND.newsletter}</strong> for more weird brilliance
          </div>
          <div>
            Currently ranked Australia's #1 semantic engine by our algorithm's mums
          </div>
        </div>
      </div>
    </div>
  );
}