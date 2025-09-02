// Thinkerbell Content Formatting Pipeline
// Converts structured data into polished, witty outputs

class ThinkerbellFormatter {
  constructor() {
    // Mock templates - replace these with real Thinkerbell templates
    this.templates = {
      campaign_recap: {
        structure: ['hunch', 'wisdom', 'nudge', 'spell'],
        format: 'slides'
      },
      strategy_brief: {
        structure: ['challenge', 'insight', 'strategy', 'tactics'],
        format: 'document'
      },
      measurement_framework: {
        structure: ['hypothesis', 'metrics', 'targets', 'timeline'],
        format: 'table'
      },
      creative_brief: {
        structure: ['problem', 'audience', 'message', 'magic'],
        format: 'bullets'
      }
    };

    // Thinkerbell brand voice patterns
    this.voicePatterns = {
      playful: ['delightfully', 'surprisingly', 'charmingly', 'wickedly'],
      sharp: ['precisely', 'exactly', 'directly', 'clearly'],
      unexpected: ['plot twist:', 'here\'s the thing:', 'surprise:', 'wait for it:']
    };
  }

  // Main processing pipeline
  process(inputData, templateType, outputFormat = null) {
    // 1. Parse input data
    const parsedData = this.parseInput(inputData);
    
    // 2. Get template
    const template = this.getTemplate(templateType);
    
    // 3. Apply Thinkerbell voice
    const enhancedData = this.applyBrandVoice(parsedData);
    
    // 4. Format output
    const outputFormat_final = outputFormat || template.format;
    const formattedOutput = this.formatOutput(enhancedData, template, outputFormat_final);
    
    return formattedOutput;
  }

  parseInput(input) {
    // Handle different input formats
    if (typeof input === 'string') {
      try {
        // Try JSON first
        return JSON.parse(input);
      } catch {
        try {
          // Try YAML parsing (simplified)
          return this.parseYAML(input);
        } catch {
          // Treat as markdown
          return this.parseMarkdown(input);
        }
      }
    }
    return input;
  }

  parseYAML(yamlString) {
    // Simplified YAML parser for demo
    const lines = yamlString.split('\n');
    const result = {};
    let currentKey = null;
    
    for (const line of lines) {
      const trimmed = line.trim();
      if (trimmed.includes(':') && !trimmed.startsWith('-')) {
        const [key, value] = trimmed.split(':');
        currentKey = key.trim();
        result[currentKey] = value ? value.trim() : {};
      } else if (trimmed.startsWith('-') && currentKey) {
        if (!Array.isArray(result[currentKey])) {
          result[currentKey] = [];
        }
        result[currentKey].push(trimmed.substring(1).trim());
      }
    }
    return result;
  }

  parseMarkdown(mdString) {
    // Extract sections from markdown
    const sections = mdString.split('\n## ').map(section => {
      const lines = section.split('\n');
      const title = lines[0].replace('#', '').trim();
      const content = lines.slice(1).join('\n').trim();
      return { title, content };
    });
    return { sections };
  }

  getTemplate(templateType) {
    return this.templates[templateType] || this.templates.campaign_recap;
  }

  applyBrandVoice(data) {
    // Apply Thinkerbell's brand voice patterns
    const enhanced = JSON.parse(JSON.stringify(data)); // Deep clone
    
    // Enhance content with brand voice
    if (enhanced.sections) {
      enhanced.sections = enhanced.sections.map(section => ({
        ...section,
        content: this.enhanceText(section.content, section.name || section.title)
      }));
    }
    
    return enhanced;
  }

  enhanceText(text, sectionType) {
    // Apply different voice patterns based on section type
    if (!text) return text;
    
    const lowerType = (sectionType || '').toLowerCase();
    
    // Add some Thinkerbell flair based on section type
    if (lowerType.includes('hunch')) {
      return `💡 ${text}`;
    } else if (lowerType.includes('wisdom')) {
      return `📊 ${text}`;
    } else if (lowerType.includes('nudge')) {
      return `👉 ${text}`;
    } else if (lowerType.includes('spell')) {
      return `✨ ${text}`;
    }
    
    return text;
  }

  formatOutput(data, template, format) {
    switch (format) {
      case 'slides':
        return this.formatAsSlides(data, template);
      case 'document':
        return this.formatAsDocument(data, template);
      case 'table':
        return this.formatAsTable(data, template);
      case 'bullets':
        return this.formatAsBullets(data, template);
      default:
        return this.formatAsSlides(data, template);
    }
  }

  formatAsSlides(data, template) {
    let output = '';
    
    if (data.title) {
      output += `# ${data.title}\n\n`;
    }
    
    if (data.sections) {
      data.sections.forEach(section => {
        output += `## ${section.name || section.title}\n`;
        output += `${section.content}\n\n`;
      });
    }
    
    return output;
  }

  formatAsDocument(data, template) {
    let output = '';
    
    if (data.title) {
      output += `# ${data.title}\n\n`;
    }
    
    if (data.sections) {
      data.sections.forEach((section, index) => {
        output += `**${section.name || section.title}**\n\n`;
        output += `${section.content}\n\n`;
        if (index < data.sections.length - 1) {
          output += '---\n\n';
        }
      });
    }
    
    return output;
  }

  formatAsTable(data, template) {
    if (!data.sections) return 'No data to format as table';
    
    let output = '| Section | Content |\n';
    output += '|---------|----------|\n';
    
    data.sections.forEach(section => {
      const cleanContent = (section.content || '').replace(/\n/g, ' ').replace(/\|/g, '\\|');
      output += `| ${section.name || section.title} | ${cleanContent} |\n`;
    });
    
    return output;
  }

  formatAsBullets(data, template) {
    let output = '';
    
    if (data.title) {
      output += `# ${data.title}\n\n`;
    }
    
    if (data.sections) {
      data.sections.forEach(section => {
        output += `- **${section.name || section.title}**: ${section.content}\n`;
      });
    }
    
    return output;
  }
}

// Example usage and testing
const formatter = new ThinkerbellFormatter();

// Mock input data examples
const mockCampaignData = {
  title: "Toothbrush Comedy Campaign",
  sections: [
    { name: "Hunch", content: "People are more likely to brush if they laugh first." },
    { name: "Wisdom", content: "68% of Gen Z say brushing is boring." },
    { name: "Nudge", content: "Put toothbrushes in comedy club bathrooms." },
    { name: "Spell", content: "A toothpaste that fizzes and glows on cue with punchlines." }
  ]
};

const mockYAMLInput = `title: Social Media Strategy Brief
sections:
  - name: Challenge
    content: Brand awareness is flatlining among 18-24s
  - name: Insight  
    content: This demographic consumes content in 7-second chunks
  - name: Strategy
    content: Micro-moments of brand magic, not long-form storytelling
  - name: Tactics
    content: TikTok duets with unexpected brand appearances`;

// Test the pipeline
console.log("=== CAMPAIGN RECAP (SLIDES) ===");
console.log(formatter.process(mockCampaignData, 'campaign_recap', 'slides'));

console.log("\n=== STRATEGY BRIEF (DOCUMENT) ===");
console.log(formatter.process(mockYAMLInput, 'strategy_brief', 'document'));

console.log("\n=== MEASUREMENT TABLE ===");
console.log(formatter.process(mockCampaignData, 'measurement_framework', 'table'));

// Export for module usage
if (typeof module !== 'undefined' && module.exports) {
  module.exports = ThinkerbellFormatter;
}