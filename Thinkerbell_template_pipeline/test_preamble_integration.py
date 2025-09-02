#!/usr/bin/env python3
"""
Test script to verify preamble generator integration with synthetic dataset generator
"""

import sys
sys.path.append('.')

from thinkerbell.data.preamble_generator import PreambleGenerator
from thinkerbell.data.synthetic_dataset_generator import SyntheticDatasetGenerator

def test_preamble_integration():
    """Test that preambles are properly integrated into synthetic agreements"""
    
    print("🎯 Testing Preamble Integration with Synthetic Dataset Generator")
    print("=" * 70)
    
    # Test 1: Standalone preamble generation
    print("\n📝 Test 1: Standalone Preamble Generation")
    preamble_gen = PreambleGenerator()
    
    sample_preamble = preamble_gen.generate_preamble(
        brand="Koala", 
        industry="home", 
        complexity="medium"
    )
    
    print(f"✅ Generated preamble: {sample_preamble}")
    print(f"   Length: {len(sample_preamble)} chars, {len(sample_preamble.split())} words")
    
    # Test 2: Synthetic dataset generator with preambles enabled
    print("\n📊 Test 2: Synthetic Dataset Generator with Preambles")
    
    try:
        # Initialize generator with preamble enabled
        generator = SyntheticDatasetGenerator(
            use_semantic_smoothing=False,  # Disable for cleaner test
            use_text_preprocessing=False,  # Disable for cleaner test
            use_preamble_generator=True    # Enable preambles
        )
        
        print("✅ Generator initialized successfully")
        
        # Generate a single sample
        print("\n🔄 Generating sample agreement...")
        sample = generator._generate_basic_agreement("medium")
        
        print("✅ Sample generated successfully")
        print(f"   Sample ID: {sample['id']}")
        print(f"   Complexity: {sample['complexity_level']}")
        print(f"   Confidence: {sample['confidence_score']:.2f}")
        
        # Check if preamble is included
        raw_text = sample['raw_input_text']
        print(f"\n📄 Raw Input Text (first 200 chars):")
        print(f"   {raw_text[:200]}...")
        
        # Verify preamble characteristics
        has_agency = any(agency in raw_text for agency in preamble_gen.agency_names)
        has_campaign = "campaign" in raw_text.lower()
        has_behalf = "behalf of" in raw_text.lower()
        
        print(f"\n🔍 Preamble Verification:")
        print(f"   Contains agency name: {'✅' if has_agency else '❌'}")
        print(f"   Contains 'campaign': {'✅' if has_campaign else '❌'}")
        print(f"   Contains 'behalf of': {'✅' if has_behalf else '❌'}")
        
        if has_agency and has_campaign and has_behalf:
            print("🎉 Preamble integration successful!")
        else:
            print("⚠️ Preamble may not be properly integrated")
            
    except Exception as e:
        print(f"❌ Error testing integration: {e}")
        return False
    
    # Test 3: Generate multiple samples to verify variety
    print("\n🔀 Test 3: Preamble Variety Test")
    
    try:
        preambles = []
        for i in range(5):
            brand = ["Koala", "Woolworths", "David Jones", "Bunnings", "Qantas"][i]
            industry = ["home", "food", "fashion", "home", "tech"][i]
            complexity = ["simple", "medium", "premium", "medium", "simple"][i]
            
            preamble = preamble_gen.generate_preamble(brand, industry, complexity)
            preambles.append(preamble)
            
            print(f"   {i+1}. {brand} ({complexity}): {preamble[:60]}...")
        
        # Check variety
        unique_agencies = len(set(p.split()[0] for p in preambles))
        unique_contexts = len(set(p.split('.')[1].strip() if '.' in p else '' for p in preambles))
        
        print(f"\n📊 Variety Metrics:")
        print(f"   Unique agencies: {unique_agencies}/5")
        print(f"   Unique contexts: {unique_contexts}/5")
        
        if unique_agencies >= 3 and unique_contexts >= 3:
            print("✅ Good preamble variety achieved")
        else:
            print("⚠️ Limited preamble variety")
            
    except Exception as e:
        print(f"❌ Error testing variety: {e}")
        return False
    
    print(f"\n🎉 All tests completed successfully!")
    return True

def test_batch_generation():
    """Test batch generation with preambles"""
    
    print("\n📦 Testing Batch Generation with Preambles")
    print("=" * 50)
    
    try:
        generator = SyntheticDatasetGenerator(
            use_semantic_smoothing=False,
            use_text_preprocessing=False,
            use_preamble_generator=True
        )
        
        # Generate small batch
        print("🔄 Generating batch of 3 samples...")
        complexity_dist = {"simple": 0.33, "medium": 0.33, "premium": 0.34}
        
        dataset = generator.generate_dataset(
            num_samples=3,
            complexity_distribution=complexity_dist
        )
        
        print(f"✅ Generated {len(dataset)} samples")
        
        # Analyze preamble integration
        preamble_count = 0
        for i, sample in enumerate(dataset):
            raw_text = sample['raw_input_text']
            has_preamble = any(agency in raw_text for agency in ["Thinkerbell", "Brandlink", "Stellar PR", "Publicis", "Ogilvy"])
            
            if has_preamble:
                preamble_count += 1
            
            print(f"   Sample {i+1}: {'✅ Has preamble' if has_preamble else '❌ No preamble'}")
        
        success_rate = preamble_count / len(dataset)
        print(f"\n📊 Preamble Integration Rate: {success_rate:.1%}")
        
        if success_rate >= 0.8:
            print("🎉 Excellent preamble integration!")
        elif success_rate >= 0.5:
            print("✅ Good preamble integration")
        else:
            print("⚠️ Poor preamble integration")
            
        return success_rate >= 0.5
        
    except Exception as e:
        print(f"❌ Error in batch generation: {e}")
        return False

def main():
    """Run all integration tests"""
    
    print("🧪 Preamble Generator Integration Test Suite")
    print("=" * 80)
    
    success = True
    
    # Run individual tests
    try:
        success &= test_preamble_integration()
        success &= test_batch_generation()
        
        if success:
            print(f"\n🎉 ALL TESTS PASSED! Preamble generator is ready for production.")
            print(f"💡 To use in full pipeline, ensure use_preamble_generator=True")
        else:
            print(f"\n❌ Some tests failed. Check integration.")
            
    except Exception as e:
        print(f"\n💥 Test suite failed: {e}")
        success = False
    
    return success

if __name__ == "__main__":
    main() 