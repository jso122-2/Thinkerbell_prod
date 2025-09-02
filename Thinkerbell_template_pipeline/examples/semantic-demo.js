/**
 * Semantic Thinkerbell Pipeline Demo
 * Demonstrates advanced semantic classification and real-time processing
 */

const SemanticThinkerbellPipeline = require('../src/SemanticThinkerbellPipeline');

// Initialize semantic pipeline
const pipeline = new SemanticThinkerbellPipeline({
  enableSemanticRouting: true,
  enableConfidenceIndicators: true,
  enableContentSuggestions: true,
  enableValidation: true,
  enableRealTimePreview: true,
  confidenceThreshold: 0.3,
  // pythonBackendUrl: 'http://localhost:8000' // Uncomment if you have Python backend running
});

async function runSemanticDemo() {
  console.log('🧠 Semantic Thinkerbell Pipeline Demo\n');

  // Example 1: Semantic Classification and Routing
  console.log('=== Example 1: Semantic Intelligence ===');
  
  const complexContent = `
    Our brand strategy needs a major overhaul. I have a strong feeling that consumers 
    are becoming more conscious about sustainability. Recent studies indicate that 68% 
    of millennials are willing to pay more for sustainable products. We should pivot 
    our messaging to emphasize our eco-friendly practices. Picture this: an interactive 
    sustainability tracker that gamifies eco-conscious purchasing decisions.
  `;

  const result1 = await pipeline.processWithSemantics(complexContent, 'slide_deck', {
    includeAnalytics: true,
    includeDebugInfo: false
  });

  console.log('📊 Semantic Analysis:');
  console.log(JSON.stringify(result1.analytics, null, 2));
  
  console.log('\n📝 Enhanced Output:');
  console.log(result1.formattedOutput);

  // Example 2: Real-time Preview
  console.log('\n=== Example 2: Real-time Preview ===');
  
  const partialContent1 = "I think our social media strategy";
  const partialContent2 = "I think our social media strategy needs work. Data shows";
  const partialContent3 = "I think our social media strategy needs work. Data shows that engagement is down 30%. We should";

  for (const [index, content] of [partialContent1, partialContent2, partialContent3].entries()) {
    console.log(`\n📱 Preview ${index + 1}: "${content}"`);
    const preview = await pipeline.generateRealTimePreview(content, 'slide_deck', 0);
    
    console.log('Routing:', Object.fromEntries(
      Object.entries(preview.routing).map(([cat, items]) => [cat, items.length])
    ));
    
    if (preview.suggestions && Object.keys(preview.suggestions).length > 0) {
      console.log('Suggestions:', Object.keys(preview.suggestions));
    }
  }

  // Example 3: Classification Explanation
  console.log('\n=== Example 3: Classification Explanation ===');
  
  const testSentences = [
    "Research shows that 73% of consumers prefer authentic brands",
    "I suspect our audience responds better to stories than ads",
    "We should focus on user-generated content",
    "Imagine if we created magical shopping experiences"
  ];

  for (const sentence of testSentences) {
    const explanation = await pipeline.explainClassification(sentence);
    console.log(`\n"${sentence}"`);
    console.log(`→ ${explanation.category} (${(explanation.confidence * 100).toFixed(1)}% confidence)`);
    if (explanation.explanation) {
      console.log(`  ${explanation.explanation}`);
    }
  }

  // Example 4: Content Validation and Suggestions
  console.log('\n=== Example 4: Content Validation ===');
  
  const imbalancedContent = `
    Research shows that video content performs 300% better than static images.
    Studies indicate that 85% of consumers watch videos before making purchases.
    Data reveals that video ads have higher click-through rates.
    Analytics prove that video marketing delivers better ROI.
  `;

  const result4 = await pipeline.processWithSemantics(imbalancedContent, 'slide_deck', {
    enableValidation: true,
    enableContentSuggestions: true
  });

  console.log('⚠️ Validation Warnings:');
  if (result4.validation && result4.validation.warnings) {
    result4.validation.warnings.forEach(warning => console.log(`  - ${warning}`));
  }

  console.log('\n💡 Content Suggestions:');
  if (result4.suggestions) {
    Object.entries(result4.suggestions).forEach(([category, suggestions]) => {
      console.log(`  ${category}:`);
      suggestions.slice(0, 2).forEach(suggestion => console.log(`    - ${suggestion}`));
    });
  }

  // Example 5: Smart Recommendations
  console.log('\n=== Example 5: Smart Recommendations ===');
  
  const suggestions = await pipeline.getSmartSuggestions(imbalancedContent, 'strategy_doc');
  
  if (suggestions.recommendations && suggestions.recommendations.length > 0) {
    console.log('🎯 Smart Recommendations:');
    suggestions.recommendations.forEach(rec => {
      console.log(`  [${rec.priority.toUpperCase()}] ${rec.message}`);
    });
  }

  // Example 6: Batch Processing with Semantics
  console.log('\n=== Example 6: Batch Semantic Processing ===');
  
  const batchInputs = [
    {
      data: "I think mobile-first design is crucial. Studies show 60% of users browse on mobile.",
      templateName: 'slide_deck'
    },
    {
      data: "We should implement A/B testing. Analytics indicate conversion improvements.",
      templateName: 'strategy_doc'
    },
    {
      data: "Imagine personalized shopping experiences. AI could revolutionize retail.",
      templateName: 'creative_brief'
    }
  ];

  const batchResults = await pipeline.batchProcessWithSemantics(batchInputs);
  console.log(`📦 Processed ${batchResults.length} items`);
  batchResults.forEach((result, idx) => {
    if (result.success) {
      const analytics = result.result.analytics;
      if (analytics) {
        console.log(`  Item ${idx + 1}: ${analytics.dominant_category} dominant (${analytics.total_sentences} sentences)`);
      }
    }
  });

  // Example 7: User Learning Simulation
  console.log('\n=== Example 7: User Learning Simulation ===');
  
  const learningSentence = "We need to pivot our strategy based on market feedback";
  const classification = await pipeline.explainClassification(learningSentence);
  
  console.log(`Original classification: ${classification.category}`);
  
  // Simulate user correction
  const learning = pipeline.addUserCorrection(
    learningSentence, 
    'Nudge', 
    'This is clearly an action recommendation, not a hunch'
  );
  
  console.log(`Learning recorded: ${learning.message}`);

  // Example 8: Performance and Statistics
  console.log('\n=== Example 8: Performance Statistics ===');
  
  const stats = pipeline.getSemanticStats();
  console.log('📈 Pipeline Statistics:');
  console.log(`  - Anchors: ${stats.anchors?.length || 0}`);
  console.log(`  - Confidence threshold: ${stats.confidence_threshold}`);
  console.log(`  - Backend connected: ${stats.backend_connected}`);
  console.log(`  - Cache size: ${stats.cache_size}`);
  console.log(`  - Features enabled: ${Object.entries(stats.features_enabled).filter(([k,v]) => v).map(([k]) => k).join(', ')}`);

  // Example 9: Template Integration
  console.log('\n=== Example 9: Template Integration ===');
  
  const templateResult = await pipeline.processWithSemantics(complexContent, 'creative_brief', {
    useTemplateSystem: true
  });

  console.log('🎨 Creative Brief Format:');
  console.log(templateResult.templateFormatted || templateResult.formattedOutput);

  // Example 10: Advanced Configuration
  console.log('\n=== Example 10: Configuration Export ===');
  
  const config = pipeline.exportSemanticConfig();
  console.log('⚙️ Configuration Summary:');
  console.log(`  - Available templates: ${config.templates.length}`);
  console.log(`  - Semantic routing: ${config.config.enableSemanticRouting}`);
  console.log(`  - Confidence indicators: ${config.config.enableConfidenceIndicators}`);
  console.log(`  - Backend status: ${config.backend_status ? 'Connected' : 'Local only'}`);

  console.log('\n✨ Semantic Demo Complete!');
  
  // Clean up
  pipeline.clearPreviewCache();
  
  return {
    message: 'Semantic pipeline demo completed successfully',
    features_demonstrated: [
      'Semantic classification',
      'Real-time preview',
      'Classification explanation',
      'Content validation',
      'Smart recommendations',
      'Batch processing',
      'User learning',
      'Performance statistics',
      'Template integration',
      'Configuration management'
    ]
  };
}

// Utility function to test Python backend connection
async function testPythonBackend() {
  console.log('🐍 Testing Python Backend Connection...');
  
  const backendUrl = 'http://localhost:8000';
  const connected = await pipeline.connectToPythonBackend(backendUrl);
  
  if (connected) {
    console.log('✅ Python backend connected - using advanced semantic model');
    
    // Run a test with backend
    const testResult = await pipeline.runSemanticTest();
    console.log('🧪 Backend test result:', testResult.performance);
    
    return true;
  } else {
    console.log('ℹ️ Python backend not available - using local classification');
    return false;
  }
}

// Interactive classification demo
async function interactiveDemo() {
  console.log('\n🎮 Interactive Classification Demo');
  console.log('Enter sentences to see real-time semantic classification:');
  
  const readline = require('readline');
  const rl = readline.createInterface({
    input: process.stdin,
    output: process.stdout
  });

  const askQuestion = () => {
    rl.question('\n💬 Enter a sentence (or "quit" to exit): ', async (input) => {
      if (input.toLowerCase() === 'quit') {
        rl.close();
        return;
      }
      
      if (input.trim().length > 5) {
        const result = await pipeline.explainClassification(input);
        console.log(`🎯 ${result.category} (${(result.confidence * 100).toFixed(1)}%)`);
        if (result.explanation) {
          console.log(`💡 ${result.explanation}`);
        }
        
        // Show suggestions for improvement
        const preview = await pipeline.generateRealTimePreview(input);
        if (preview.validation && !preview.validation.valid) {
          console.log('💡 Tip: Try adding more specific content to strengthen classification');
        }
      } else {
        console.log('⚠️ Please enter a longer sentence for better classification');
      }
      
      askQuestion();
    });
  };

  askQuestion();
}

// Export functions for module usage
module.exports = {
  runSemanticDemo,
  testPythonBackend,
  interactiveDemo,
  pipeline
};

// Run demo if called directly
if (require.main === module) {
  runSemanticDemo()
    .then(() => {
      console.log('\n🎯 Want to try the interactive demo? Run: npm run semantic:interactive');
      console.log('🐍 Want to test Python backend? Run: npm run semantic:backend');
    })
    .catch(console.error);
} 