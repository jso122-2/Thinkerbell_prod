import React from 'react';

// Template page with model functionality
export default function TemplatePage() {
  console.log('📄 TemplatePage rendering...');
  
  // Form state for the template
  const [formData, setFormData] = React.useState({
    // Classification
    document_type: 'INFLUENCER_AGREEMENT',
    complexity: 'simple',
    industry: 'tech',
    
    // Basic Info
    client: '',
    brand: '',
    campaign: '',
    influencer_name: '',
    agency: '',
    
    // Commercial Terms
    fee: '',
    fee_numeric: 0,
    payment_terms: 'Net 30 days from invoice date',
    
    // Deliverables
    deliverables: [''],
    
    // Terms
    engagement_term: '',
    exclusivity_period: '',
    exclusivity_scope: ['competitors'],
    usage_term: '',
    territory: 'Australia',
    
    // Raw text input for model parsing
    raw_text: ''
  });
  
  const [isProcessing, setIsProcessing] = React.useState(false);
  const [generatedAgreement, setGeneratedAgreement] = React.useState('');
  const [modelResults, setModelResults] = React.useState<Record<string, string> | null>(null);
  const [activeTab, setActiveTab] = React.useState('single'); // 'single' or 'bulk'
  const [bulkText, setBulkText] = React.useState('');
  const [bulkResults, setBulkResults] = React.useState<Record<string, unknown>[]>([]);
  const [isBulkProcessing, setIsBulkProcessing] = React.useState(false);

  const updateFormField = (field: string, value: string | number | string[]) => {
    setFormData(prev => ({ ...prev, [field]: value }));
  };

  const addDeliverable = () => {
    setFormData(prev => ({
      ...prev,
      deliverables: [...prev.deliverables, '']
    }));
  };

  const updateDeliverable = (index: number, value: string) => {
    setFormData(prev => ({
      ...prev,
      deliverables: prev.deliverables.map((item, i) => i === index ? value : item)
    }));
  };

  const removeDeliverable = (index: number) => {
    setFormData(prev => ({
      ...prev,
      deliverables: prev.deliverables.filter((_, i) => i !== index)
    }));
  };

  // Model functionality - parse raw text into template
  const parseWithModel = async () => {
    if (!formData.raw_text.trim()) {
      alert('Please enter some text to parse');
      return;
    }

    setIsProcessing(true);
    try {
      // Simulate calling the sentence encoder model to extract fields
      console.log('🧠 Calling sentence encoder model...');
      
      // This would connect to MinIO bucket and use the model
      // For now, simulate the extraction based on the training data patterns
      await new Promise(resolve => setTimeout(resolve, 2000)); // Simulate processing time
      
      // Mock extraction results (this would come from the actual model)
      const mockExtraction = {
        client: extractClientFromText(formData.raw_text),
        brand: extractBrandFromText(formData.raw_text),
        campaign: extractCampaignFromText(formData.raw_text),
        fee: extractFeeFromText(formData.raw_text),
        deliverables: extractDeliverablesFromText(formData.raw_text),
        engagement_term: extractEngagementFromText(formData.raw_text),
        usage_term: extractUsageFromText(formData.raw_text),
        industry: detectIndustry(formData.raw_text)
      };

      // Update form with extracted data
      setFormData(prev => ({
        ...prev,
        ...mockExtraction,
        fee_numeric: parseFloat(mockExtraction.fee.replace(/[^0-9.]/g, '')) || 0
      }));

      setModelResults(mockExtraction);
      console.log('✅ Model extraction complete:', mockExtraction);
      
    } catch (error) {
      console.error('❌ Model processing failed:', error);
      alert('Model processing failed. Please try again.');
    } finally {
      setIsProcessing(false);
    }
  };

  // Simple extraction functions (these would be replaced by the actual model)
  const extractClientFromText = (text: string) => {
    const clientMatch = text.match(/agreement is between ([^and]+) and/i) || 
                       text.match(/on behalf of ([^.]+)/i);
    return clientMatch ? clientMatch[1].trim() : '';
  };

  const extractBrandFromText = (text: string) => {
    const brandMatch = text.match(/behalf of ([^.]+)/i);
    return brandMatch ? brandMatch[1].trim() : '';
  };

  const extractCampaignFromText = (text: string) => {
    const campaignMatch = text.match(/for the ([^.]+(?:Campaign|campaign)[^.]*)/i);
    return campaignMatch ? campaignMatch[1].trim() : '';
  };

  const extractFeeFromText = (text: string) => {
    const feeMatch = text.match(/Total Fee: \$?([\d,]+)/i) ||
                    text.match(/Fee: \$?([\d,]+)/i) ||
                    text.match(/\$?([\d,]+) AUD/i);
    return feeMatch ? `$${feeMatch[1]}` : '';
  };

  const extractDeliverablesFromText = (text: string) => {
    const deliverablesSection = text.match(/DELIVERABLES:[\s\S]*?(?=\n\n|\nTERMS|$)/i);
    if (!deliverablesSection) return [''];
    
    const deliverables = deliverablesSection[0]
      .split('\n')
      .filter(line => line.trim().startsWith('-'))
      .map(line => line.replace(/^-\s*/, '').trim());
    
    return deliverables.length > 0 ? deliverables : [''];
  };

  const extractEngagementFromText = (text: string) => {
    const engagementMatch = text.match(/Engagement Period: ([^\\n]+)/i);
    return engagementMatch ? engagementMatch[1].trim() : '';
  };

  const extractUsageFromText = (text: string) => {
    const usageMatch = text.match(/Usage Rights: ([^\\n]+)/i);
    return usageMatch ? usageMatch[1].trim() : '';
  };

  const detectIndustry = (text: string) => {
    const techKeywords = ['tech', 'technology', 'innovation', 'digital'];
    const fashionKeywords = ['fashion', 'beauty', 'style', 'wellness'];
    const lowerText = text.toLowerCase();
    
    if (techKeywords.some(keyword => lowerText.includes(keyword))) return 'tech';
    if (fashionKeywords.some(keyword => lowerText.includes(keyword))) return 'fashion';
    return 'general';
  };

  // Bulk text processing functionality
  const processBulkText = async () => {
    if (!bulkText.trim()) {
      alert('Please enter bulk text to process');
      return;
    }

    setIsBulkProcessing(true);
    setBulkResults([]);
    
    try {
      console.log('🧠 Processing bulk text with AI model...');
      
      // Split bulk text into chunks (500-600 words each)
      const chunks = splitIntoChunks(bulkText, 500, 600);
      console.log(`📄 Split into ${chunks.length} chunks`);
      
      const results = [];
      
      for (let i = 0; i < chunks.length; i++) {
        console.log(`🔄 Processing chunk ${i + 1}/${chunks.length}...`);
        
        // Simulate processing time for each chunk
        await new Promise(resolve => setTimeout(resolve, 1500));
        
        // Extract fields from this chunk
        const chunkExtraction = {
          chunk_id: i + 1,
          word_count: chunks[i].split(' ').length,
          text: chunks[i],
          extracted_fields: {
            client: extractClientFromText(chunks[i]),
            brand: extractBrandFromText(chunks[i]),
            campaign: extractCampaignFromText(chunks[i]),
            fee: extractFeeFromText(chunks[i]),
            deliverables: extractDeliverablesFromText(chunks[i]),
            engagement_term: extractEngagementFromText(chunks[i]),
            usage_term: extractUsageFromText(chunks[i]),
            industry: detectIndustry(chunks[i])
          },
          confidence_score: Math.random() * 0.3 + 0.7, // 0.7-1.0
          template_match: getRandomTemplateMatch()
        };
        
        results.push(chunkExtraction);
        setBulkResults([...results]); // Update UI progressively
      }
      
      console.log('✅ Bulk processing complete:', results);
      
    } catch (error) {
      console.error('❌ Bulk processing failed:', error);
      alert('Bulk processing failed. Please try again.');
    } finally {
      setIsBulkProcessing(false);
    }
  };

  const splitIntoChunks = (text: string, minWords: number, maxWords: number) => {
    const words = text.split(/\s+/);
    const chunks = [];
    let currentChunk = [];
    
    for (let i = 0; i < words.length; i++) {
      currentChunk.push(words[i]);
      
      // If we've reached max words or we're at the end and above min words
      if (currentChunk.length >= maxWords || 
          (i === words.length - 1 && currentChunk.length >= minWords)) {
        chunks.push(currentChunk.join(' '));
        currentChunk = [];
      }
    }
    
    // Add remaining words if any
    if (currentChunk.length > 0) {
      if (chunks.length > 0) {
        // Append to last chunk if it exists
        chunks[chunks.length - 1] += ' ' + currentChunk.join(' ');
      } else {
        // Create new chunk if this is the only content
        chunks.push(currentChunk.join(' '));
      }
    }
    
    return chunks;
  };

  const getRandomTemplateMatch = () => {
    const templates = [
      'profile_Mattel Brickshop x Craig Contract FINAL.docx',
      'profile_AMAZON _ PBDD Talent Agreement - Jackie Gillies DRAFT',
      'profile_TKB Influencer Agreement - Dr Claire x Australia Post',
      'profile_Rexona _ WBD Foot Model Talent Agreement',
      'profile_Dove Advanced Care Body Wash Launch_Jhyll Teplin'
    ];
    return templates[Math.floor(Math.random() * templates.length)];
  };

  const generateBulkAgreements = () => {
    const agreements = bulkResults.map(result => {
      const fields = result.extracted_fields;
      return `
=== AGREEMENT ${result.chunk_id} ===
Word Count: ${result.word_count} | Confidence: ${(result.confidence_score * 100).toFixed(1)}%

INFLUENCER AGREEMENT

This agreement is between ${fields.brand || '[BRAND]'} and the Influencer for ${fields.campaign || '[CAMPAIGN]'}.

COMMERCIAL TERMS:
- Total Fee: ${fields.fee || '[FEE]'} AUD (inclusive of GST)
- Payment Terms: Net 30 days from invoice date
- Invoice Requirements: Tax invoice required upon completion

DELIVERABLES:
${fields.deliverables.filter((d: string) => d.trim()).map((d: string) => `- ${d}`).join('\n') || '- [DELIVERABLES]'}

TERMS AND CONDITIONS:
- Engagement Period: ${fields.engagement_term || '[ENGAGEMENT_TERM]'}
- Territory: Australia
- Usage Rights: ${fields.usage_term || '[USAGE_TERM]'}
- Content Rights: ${fields.brand || '[BRAND]'} retains full commercial rights

Template Match: ${result.template_match}
Industry: ${fields.industry}

==============================
      `.trim();
    }).join('\n\n');
    
    return agreements;
  };

  // Generate the agreement from the form data
  const generateAgreement = () => {
    const agreement = `
INFLUENCER AGREEMENT

This agreement is between ${formData.brand || '[BRAND]'} and the Influencer for ${formData.campaign || '[CAMPAIGN]'}.

COMMERCIAL TERMS:
- Total Fee: ${formData.fee || '[FEE]'} AUD (inclusive of GST)
- Payment Terms: ${formData.payment_terms}
- Invoice Requirements: Tax invoice required upon completion

DELIVERABLES:
${formData.deliverables.filter(d => d.trim()).map(d => `- ${d}`).join('\n') || '- [DELIVERABLES]'}

TERMS AND CONDITIONS:
- Engagement Period: ${formData.engagement_term || '[ENGAGEMENT_TERM]'}
- Exclusivity Period: ${formData.exclusivity_period || '[EXCLUSIVITY_PERIOD]'}
- Territory: ${formData.territory}
- Usage Rights: ${formData.usage_term || '[USAGE_TERM]'}
- Content Rights: ${formData.brand || '[BRAND]'} retains full commercial rights

COMPLIANCE:
- All content must comply with ACMA guidelines and ASA codes
- Content is subject to client approval before publication
- Disclosure requirements must be met per ACCC guidelines

LEGAL:
This agreement is governed by Australian law and subject to the jurisdiction of Australian courts.

Agreed terms are subject to final contract execution.
    `.trim();

    setGeneratedAgreement(agreement);
  };

  return (
    <div className="space-y-8 p-8">
      <div className="relative">
        <h1 className="text-5xl font-black text-black mb-4">
          Agreement <span className="tb-accent-pink">Template</span>
        </h1>
        <p className="text-xl text-gray-600 mb-6 font-bold">
          AI-powered influencer agreement generation
        </p>
        <div className="tb-zigzag mb-6"></div>
        
        {/* Tab Navigation */}
        <div className="flex space-x-4 mb-6">
          <button
            onClick={() => setActiveTab('single')}
            className={`px-6 py-3 font-bold rounded-lg transition-all duration-200 ${
              activeTab === 'single'
                ? 'bg-tb-magenta text-white shadow-lg'
                : 'bg-gray-200 text-gray-700 hover:bg-gray-300'
            }`}
          >
            📄 Single Document
          </button>
          <button
            onClick={() => setActiveTab('bulk')}
            className={`px-6 py-3 font-bold rounded-lg transition-all duration-200 ${
              activeTab === 'bulk'
                ? 'bg-tb-magenta text-white shadow-lg'
                : 'bg-gray-200 text-gray-700 hover:bg-gray-300'
            }`}
          >
            📦 Bulk Processing
          </button>
        </div>
      </div>

      {/* Single Document Processing */}
      {activeTab === 'single' && (
        <div className="grid grid-cols-1 xl:grid-cols-3 gap-8">
          {/* Input Panel - Raw Text */}
          <div className="tb-card">
            <h3 className="text-2xl font-black text-black mb-6">🧠 Text Input & AI Parsing</h3>
            
            <div className="space-y-4">
              <div>
                <label className="block text-sm font-bold text-black mb-2">
                  Raw Agreement Text
                </label>
                <textarea
                  value={formData.raw_text}
                  onChange={(e) => updateFormField('raw_text', e.target.value)}
                  className="w-full h-48 p-3 border-2 border-gray-200 rounded-lg font-medium focus:border-tb-magenta focus:outline-none resize-none"
                  placeholder="Paste or type the raw agreement text here. The AI model will extract structured data..."
                />
              </div>
              
              <button
                onClick={parseWithModel}
                disabled={isProcessing}
                className="w-full bg-tb-magenta text-white font-black py-4 px-6 rounded-lg hover:bg-pink-700 transition-all duration-200 transform hover:scale-105 disabled:opacity-50 disabled:transform-none"
              >
                {isProcessing ? (
                  <>
                    <div className="inline-block animate-spin rounded-full h-4 w-4 border-b-2 border-white mr-2"></div>
                    🧠 AI PROCESSING...
                  </>
                ) : (
                  '🧠 PARSE WITH AI MODEL'
                )}
              </button>

              {modelResults && (
                <div className="mt-4 p-4 bg-green-50 rounded-lg">
                  <h4 className="font-bold text-green-800 mb-2">✅ AI Extraction Results</h4>
                  <div className="text-sm space-y-1">
                    <p><strong>Client:</strong> {modelResults.client || 'Not found'}</p>
                    <p><strong>Campaign:</strong> {modelResults.campaign || 'Not found'}</p>
                    <p><strong>Fee:</strong> {modelResults.fee || 'Not found'}</p>
                    <p><strong>Industry:</strong> {modelResults.industry || 'Not detected'}</p>
                  </div>
                </div>
              )}
            </div>
          </div>

          {/* Template Form */}
          <div className="tb-card">
            <h3 className="text-2xl font-black text-black mb-6">📋 Agreement Template</h3>
            
            <div className="space-y-4">
              {/* Basic Information */}
              <div className="bg-gray-50 p-4 rounded-lg">
                <h4 className="font-bold text-black mb-3">Basic Information</h4>
                <div className="grid grid-cols-1 gap-3">
                  <input
                    type="text"
                    value={formData.client}
                    onChange={(e) => updateFormField('client', e.target.value)}
                    placeholder="Client Name"
                    className="w-full p-3 border-2 border-gray-200 rounded-lg font-medium focus:border-tb-magenta focus:outline-none"
                  />
                  <input
                    type="text"
                    value={formData.brand}
                    onChange={(e) => updateFormField('brand', e.target.value)}
                    placeholder="Brand Name"
                    className="w-full p-3 border-2 border-gray-200 rounded-lg font-medium focus:border-tb-magenta focus:outline-none"
                  />
                  <input
                    type="text"
                    value={formData.campaign}
                    onChange={(e) => updateFormField('campaign', e.target.value)}
                    placeholder="Campaign Name"
                    className="w-full p-3 border-2 border-gray-200 rounded-lg font-medium focus:border-tb-magenta focus:outline-none"
                  />
                  <select
                    value={formData.industry}
                    onChange={(e) => updateFormField('industry', e.target.value)}
                    className="w-full p-3 border-2 border-gray-200 rounded-lg font-medium focus:border-tb-magenta focus:outline-none"
                  >
                    <option value="tech">Technology</option>
                    <option value="fashion">Fashion & Beauty</option>
                    <option value="fitness">Health & Fitness</option>
                    <option value="food">Food & Beverage</option>
                    <option value="automotive">Automotive</option>
                    <option value="general">General</option>
                  </select>
                </div>
              </div>

              {/* Commercial Terms */}
              <div className="bg-blue-50 p-4 rounded-lg">
                <h4 className="font-bold text-black mb-3">Commercial Terms</h4>
                <div className="grid grid-cols-2 gap-3">
                  <input
                    type="text"
                    value={formData.fee}
                    onChange={(e) => updateFormField('fee', e.target.value)}
                    placeholder="Total Fee (e.g., $12,883)"
                    className="w-full p-3 border-2 border-gray-200 rounded-lg font-medium focus:border-tb-magenta focus:outline-none"
                  />
                  <select
                    value={formData.territory}
                    onChange={(e) => updateFormField('territory', e.target.value)}
                    className="w-full p-3 border-2 border-gray-200 rounded-lg font-medium focus:border-tb-magenta focus:outline-none"
                  >
                    <option value="Australia">Australia</option>
                    <option value="Global">Global</option>
                    <option value="APAC">APAC</option>
                    <option value="US">United States</option>
                  </select>
                </div>
              </div>

              {/* Deliverables */}
              <div className="bg-yellow-50 p-4 rounded-lg">
                <h4 className="font-bold text-black mb-3">Deliverables</h4>
                {formData.deliverables.map((deliverable, index) => (
                  <div key={index} className="flex gap-2 mb-2">
                    <input
                      type="text"
                      value={deliverable}
                      onChange={(e) => updateDeliverable(index, e.target.value)}
                      placeholder="e.g., 2 x Instagram posts"
                      className="flex-1 p-3 border-2 border-gray-200 rounded-lg font-medium focus:border-tb-magenta focus:outline-none"
                    />
                    {formData.deliverables.length > 1 && (
                      <button
                        onClick={() => removeDeliverable(index)}
                        className="px-3 py-2 bg-red-500 text-white rounded-lg hover:bg-red-600"
                      >
                        ✕
                      </button>
                    )}
                  </div>
                ))}
                <button
                  onClick={addDeliverable}
                  className="mt-2 px-4 py-2 bg-tb-magenta text-white rounded-lg hover:bg-pink-700 font-bold"
                >
                  + Add Deliverable
                </button>
              </div>

              {/* Terms */}
              <div className="bg-green-50 p-4 rounded-lg">
                <h4 className="font-bold text-black mb-3">Terms & Conditions</h4>
                <div className="grid grid-cols-1 gap-3">
                  <input
                    type="text"
                    value={formData.engagement_term}
                    onChange={(e) => updateFormField('engagement_term', e.target.value)}
                    placeholder="Engagement Period (e.g., 2 months)"
                    className="w-full p-3 border-2 border-gray-200 rounded-lg font-medium focus:border-tb-magenta focus:outline-none"
                  />
                  <input
                    type="text"
                    value={formData.exclusivity_period}
                    onChange={(e) => updateFormField('exclusivity_period', e.target.value)}
                    placeholder="Exclusivity Period (e.g., 6 weeks)"
                    className="w-full p-3 border-2 border-gray-200 rounded-lg font-medium focus:border-tb-magenta focus:outline-none"
                  />
                  <input
                    type="text"
                    value={formData.usage_term}
                    onChange={(e) => updateFormField('usage_term', e.target.value)}
                    placeholder="Usage Rights (e.g., 18 months from campaign launch)"
                    className="w-full p-3 border-2 border-gray-200 rounded-lg font-medium focus:border-tb-magenta focus:outline-none"
                  />
                </div>
              </div>

              <button
                onClick={generateAgreement}
                className="w-full bg-tb-green-500 text-white font-black py-4 px-6 rounded-lg hover:bg-green-600 transition-all duration-200 transform hover:scale-105"
              >
                📄 GENERATE AGREEMENT
              </button>
            </div>
          </div>

          {/* Generated Output */}
          <div className="tb-card">
            <h3 className="text-2xl font-black text-black mb-6">📄 Generated Agreement</h3>
            
            {generatedAgreement ? (
              <div className="space-y-4">
                <textarea
                  value={generatedAgreement}
                  readOnly
                  className="w-full h-96 p-4 border-2 border-gray-200 rounded-lg font-mono text-sm bg-gray-50 resize-none"
                />
                <div className="flex gap-2">
                  <button
                    onClick={() => navigator.clipboard.writeText(generatedAgreement)}
                    className="flex-1 bg-blue-500 text-white font-bold py-3 px-4 rounded-lg hover:bg-blue-600"
                  >
                    📋 Copy
                  </button>
                  <button
                    onClick={() => {
                      const blob = new Blob([generatedAgreement], { type: 'text/plain' });
                      const url = URL.createObjectURL(blob);
                      const a = document.createElement('a');
                      a.href = url;
                      a.download = `agreement_${formData.client || 'template'}.txt`;
                      a.click();
                    }}
                    className="flex-1 bg-tb-magenta text-white font-bold py-3 px-4 rounded-lg hover:bg-pink-700"
                  >
                    💾 Download
                  </button>
                </div>
              </div>
            ) : (
              <div className="h-96 flex items-center justify-center text-gray-500 border-2 border-dashed border-gray-300 rounded-lg">
                <div className="text-center">
                  <div className="text-4xl mb-4">📄</div>
                  <p className="font-medium">Generated agreement will appear here</p>
                  <p className="text-sm">Fill out the template and click "Generate Agreement"</p>
                </div>
              </div>
            )}
          </div>
        </div>
      )}

      {/* Bulk Text Processing */}
      {activeTab === 'bulk' && (
        <div className="grid grid-cols-1 lg:grid-cols-2 gap-8">
          {/* Bulk Input Panel */}
          <div className="tb-card">
            <h3 className="text-2xl font-black text-black mb-6">📦 Bulk Text Processing</h3>
            
            <div className="space-y-4">
              <div>
                <label className="block text-sm font-bold text-black mb-2">
                  Large Text Document (500-600 words per chunk)
                </label>
                <textarea
                  value={bulkText}
                  onChange={(e) => setBulkText(e.target.value)}
                  className="w-full h-80 p-4 border-2 border-gray-200 rounded-lg font-medium focus:border-tb-magenta focus:outline-none resize-none"
                  placeholder="Paste large document text here (multiple agreements, contracts, etc.). The AI will automatically split this into 500-600 word chunks and process each one individually..."
                />
                <div className="mt-2 text-sm text-gray-600">
                  <span className="font-medium">Word count:</span> {bulkText.split(/\s+/).filter(w => w.length > 0).length} words
                  {bulkText && (
                    <span className="ml-4">
                      <span className="font-medium">Estimated chunks:</span> {Math.ceil(bulkText.split(/\s+/).filter(w => w.length > 0).length / 550)}
                    </span>
                  )}
                </div>
              </div>
              
              <div className="bg-blue-50 p-4 rounded-lg">
                <h4 className="font-bold text-black mb-2">📋 Bulk Processing Features</h4>
                <ul className="text-sm space-y-1">
                  <li>• Automatic text chunking (500-600 words)</li>
                  <li>• Individual AI extraction per chunk</li>
                  <li>• Progressive processing with live updates</li>
                  <li>• Template matching and confidence scoring</li>
                  <li>• Bulk agreement generation</li>
                </ul>
              </div>
              
              <button
                onClick={processBulkText}
                disabled={isBulkProcessing || !bulkText.trim()}
                className="w-full bg-tb-magenta text-white font-black py-4 px-6 rounded-lg hover:bg-pink-700 transition-all duration-200 transform hover:scale-105 disabled:opacity-50 disabled:transform-none"
              >
                {isBulkProcessing ? (
                  <>
                    <div className="inline-block animate-spin rounded-full h-4 w-4 border-b-2 border-white mr-2"></div>
                    🧠 PROCESSING BULK TEXT... ({bulkResults.length} chunks completed)
                  </>
                ) : (
                  '🧠 PROCESS BULK TEXT WITH AI'
                )}
              </button>

              {bulkResults.length > 0 && (
                <div className="mt-6">
                  <button
                    onClick={() => {
                      const bulkAgreements = generateBulkAgreements();
                      const blob = new Blob([bulkAgreements], { type: 'text/plain' });
                      const url = URL.createObjectURL(blob);
                      const a = document.createElement('a');
                      a.href = url;
                      a.download = `bulk_agreements_${new Date().toISOString().split('T')[0]}.txt`;
                      a.click();
                    }}
                    className="w-full bg-tb-green-500 text-white font-black py-3 px-6 rounded-lg hover:bg-green-600 transition-all duration-200"
                  >
                    💾 DOWNLOAD ALL AGREEMENTS ({bulkResults.length} documents)
                  </button>
                </div>
              )}
            </div>
          </div>

          {/* Bulk Results Panel */}
          <div className="tb-card">
            <h3 className="text-2xl font-black text-black mb-6">📊 Processing Results</h3>
            
            {bulkResults.length === 0 && !isBulkProcessing ? (
              <div className="h-80 flex items-center justify-center text-gray-500 border-2 border-dashed border-gray-300 rounded-lg">
                <div className="text-center">
                  <div className="text-4xl mb-4">📦</div>
                  <p className="font-medium">Bulk processing results will appear here</p>
                  <p className="text-sm">Enter text and click "Process Bulk Text"</p>
                </div>
              </div>
            ) : (
              <div className="space-y-4 max-h-80 overflow-y-auto">
                {bulkResults.map((result, index) => (
                  <div key={index} className="border rounded-lg p-4 bg-gray-50">
                    <div className="flex justify-between items-start mb-2">
                      <h4 className="font-bold text-black">Chunk {result.chunk_id}</h4>
                      <div className="text-right">
                        <div className="text-sm font-medium text-gray-600">{result.word_count} words</div>
                        <div className={`text-xs font-bold ${
                          result.confidence_score > 0.8 ? 'text-green-600' : 
                          result.confidence_score > 0.6 ? 'text-yellow-600' : 'text-red-600'
                        }`}>
                          {(result.confidence_score * 100).toFixed(1)}% confidence
                        </div>
                      </div>
                    </div>
                    
                    <div className="grid grid-cols-2 gap-2 text-xs">
                      <div>
                        <span className="font-medium">Client:</span> {result.extracted_fields.client || 'N/A'}
                      </div>
                      <div>
                        <span className="font-medium">Fee:</span> {result.extracted_fields.fee || 'N/A'}
                      </div>
                      <div>
                        <span className="font-medium">Campaign:</span> {result.extracted_fields.campaign || 'N/A'}
                      </div>
                      <div>
                        <span className="font-medium">Industry:</span> {result.extracted_fields.industry || 'N/A'}
                      </div>
                    </div>
                    
                    <div className="mt-2 text-xs text-gray-500">
                      <span className="font-medium">Template:</span> {result.template_match}
                    </div>
                    
                    <details className="mt-2">
                      <summary className="cursor-pointer text-xs font-medium text-tb-magenta hover:text-pink-700">
                        View chunk text ▼
                      </summary>
                      <div className="mt-2 p-2 bg-white rounded text-xs font-mono">
                        {result.text.substring(0, 200)}...
                      </div>
                    </details>
                  </div>
                ))}
                
                {isBulkProcessing && (
                  <div className="border rounded-lg p-4 bg-blue-50 animate-pulse">
                    <div className="flex items-center space-x-2">
                      <div className="animate-spin rounded-full h-4 w-4 border-b-2 border-tb-magenta"></div>
                      <span className="font-medium text-black">Processing next chunk...</span>
                    </div>
                  </div>
                )}
              </div>
            )}
          </div>
        </div>
      )}
    </div>
  );
}
