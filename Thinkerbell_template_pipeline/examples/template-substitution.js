/**
 * Template Substitution Examples
 * Shows how to replace mock templates with real Thinkerbell templates
 */

const ThinkerbellPipeline = require('../src/ThinkerbellPipeline');
const path = require('path');

async function demonstrateTemplateSubstitution() {
  console.log('🔄 Template Substitution Examples\n');

  const pipeline = new ThinkerbellPipeline();

  // Example 1: Single Template Substitution
  console.log('=== Example 1: Single Template Substitution ===');
  
  const realSlideTemplate = {
    header: "# {{title}} 🎯\n> *{{subtitle}}*\n\n**Thinkerbell Strategy Session**\n\n---\n\n",
    section: "## {{section_icon}} {{section_title}}\n\n{{section_content}}\n\n*Insight from the Thinkerbell team*\n\n---\n\n",
    footer: "\n*✨ Crafted with strategic magic by Thinkerbell*\n*{{date}}*"
  };

  // Substitute the slide deck template
  pipeline.substituteTemplates({
    slide_deck: realSlideTemplate
  });

  // Test with sample data
  const testData = {
    title: "Brand Transformation Strategy",
    subtitle: "Redefining market presence",
    sections: [
      { name: "Hunch", content: "Consumers crave authentic brand stories, not perfect facades" },
      { name: "Wisdom", content: "85% of purchasing decisions are driven by emotional connection" }
    ]
  };

  const output1 = await pipeline.processWithTemplate(testData, 'slide_deck');
  console.log(output1);

  // Example 2: Batch Template Substitution
  console.log('\n=== Example 2: Batch Template Substitution ===');
  
  const realTemplates = {
    strategy_doc: {
      header: "# {{title}}\n**Strategic Framework** | Thinkerbell\n\n> {{summary}}\n\n",
      section: "### 🎯 {{section_title}}\n\n{{section_content}}\n\n",
      callout: "> **💡 Thinkerbell Insight**: {{callout_content}}\n\n",
      footer: "---\n\n*Strategic framework developed by Thinkerbell*\n*Confidential | {{date}}*"
    },
    
    creative_brief: {
      header: "# Creative Brief: {{campaign_name}} ✨\n**Thinkerbell Creative Development**\n\n",
      section: "**{{section_title}}**\n\n{{section_content}}\n\n",
      magic_moment: "🌟 **The Thinkerbell Magic**\n\n{{magic_content}}\n\n*This is where strategy meets creativity*\n\n",
      approval_box: "---\n\n**Creative Approval Process**\n- [ ] Strategy Director\n- [ ] Creative Director  \n- [ ] Client Stakeholder\n- [ ] Thinkerbell Leadership\n\n*Approval Date: {{date}}*"
    }
  };

  pipeline.substituteTemplates(realTemplates);
  console.log('✅ Batch substitution complete');

  // Example 3: Loading Templates from Files (simulated)
  console.log('\n=== Example 3: File-based Template Loading ===');
  
  // This would load templates from actual files in a real implementation
  const templatePaths = {
    slide_deck: './templates/real_slide_deck.json',
    strategy_doc: './templates/real_strategy_doc.json',
    measurement_report: './templates/real_measurement.json'
  };

  try {
    // In a real scenario, this would load from actual files
    console.log('🔄 Would load templates from:', templatePaths);
    console.log('📁 Create these template files in your project for real usage');
  } catch (error) {
    console.log('ℹ️ Template files not found - using simulated loading');
  }

  // Example 4: Custom Template Creation
  console.log('\n=== Example 4: Custom Template Creation ===');
  
  const customTemplate = {
    header: "# {{title}} | Thinkerbell Innovation Lab\n\n**Innovation Brief**\n\n",
    section: "### {{section_icon}} {{section_title}}\n\n{{section_content}}\n\n*Innovation Score: ⭐⭐⭐⭐⭐*\n\n",
    innovation_box: "🔬 **Innovation Potential**\n\n{{innovation_content}}\n\n",
    footer: "\n---\n*Innovation brief by Thinkerbell Labs | {{date}}*"
  };

  const customMetadata = {
    description: "Innovation-focused template for breakthrough ideas",
    best_for: ["innovation briefs", "R&D projects", "breakthrough campaigns"],
    sections: ["opportunity", "innovation", "implementation", "impact"]
  };

  pipeline.addTemplate('innovation_brief', customTemplate, customMetadata);

  // Test custom template
  const innovationData = {
    title: "AI-Powered Personalization",
    sections: [
      { name: "Opportunity", content: "Hyper-personalization is the next frontier in customer experience" },
      { name: "Innovation", content: "Real-time AI that adapts messaging based on micro-expressions" }
    ],
    innovation_content: "Revolutionary empathy engine that reads customer emotion in real-time"
  };

  const innovationOutput = await pipeline.processWithTemplate(innovationData, 'innovation_brief');
  console.log(innovationOutput);

  // Example 5: Export and Import Configuration
  console.log('\n=== Example 5: Configuration Management ===');
  
  // Export current configuration
  const config = pipeline.exportConfig();
  console.log('📤 Exported configuration keys:', Object.keys(config));

  // Create new pipeline and import configuration
  const newPipeline = new ThinkerbellPipeline();
  newPipeline.importConfig(config);
  console.log('📥 Configuration imported to new pipeline instance');

  // Example 6: Template Validation
  console.log('\n=== Example 6: Template Validation ===');
  
  const invalidData = { sections: [] }; // Missing title
  
  try {
    await pipeline.processWithTemplate(invalidData, 'slide_deck');
  } catch (error) {
    console.log('⚠️ Validation caught the error:', error.message);
  }

  console.log('\n✅ Template substitution examples complete!');
  console.log('\n📋 Available templates after substitution:');
  pipeline.getAvailableTemplates().forEach(template => {
    console.log(`  - ${template.name}: ${template.info?.description || 'Custom template'}`);
  });
}

// Utility function to create example template files
async function createExampleTemplateFiles() {
  const fs = require('fs').promises;
  
  try {
    await fs.mkdir('./templates', { recursive: true });
    
    const exampleTemplate = {
      header: "# {{title}} - Real Template\n*Loaded from file system*\n\n",
      section: "## {{section_title}}\n{{section_content}}\n\n",
      footer: "*Template loaded from file | {{date}}*"
    };
    
    await fs.writeFile('./templates/example_template.json', JSON.stringify(exampleTemplate, null, 2));
    console.log('📁 Created example template file: ./templates/example_template.json');
  } catch (error) {
    console.log('⚠️ Could not create template files:', error.message);
  }
}

// Run examples
if (require.main === module) {
  demonstrateTemplateSubstitution()
    .then(() => createExampleTemplateFiles())
    .catch(console.error);
}

module.exports = { 
  demonstrateTemplateSubstitution, 
  createExampleTemplateFiles 
}; 