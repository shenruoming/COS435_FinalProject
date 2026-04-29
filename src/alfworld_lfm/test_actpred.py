"""
test_actpred.py - Small test to verify ACTPRED works correctly
Run this before the full training to catch errors early.
"""

import sys
import torch
from environment import VerbalizedALFWorld
from base_train import BaseExpertDataset

# Import your ACTPRED class
from actpred import ACTPREDDataset


def test_actpred_single_episode():
    """Test ACTPRED on a single episode with verbose output"""
    
    print("=" * 60)
    print("TESTING ACTPRED - Single Episode")
    print("=" * 60)
    
    # Create environment
    print("\n[1] Initializing environment...")
    env = VerbalizedALFWorld(split='train')
    
    # Create ACTPRED dataset with just 1 episode for testing
    print("\n[2] Creating ACTPRED dataset (1 episode)...")
    
    # Note: This will load Mistral (takes ~30 seconds first time)
    try:
        dataset = ACTPREDDataset(
            env=env,
            num_episodes=1,      # Just 1 episode for testing
            max_steps=10,        # Max 10 steps per episode
            context_window=5,    # Small context window
            load_path=None,      # Don't load from disk, collect fresh
            use_8bit=True        # Use 8-bit to save memory
        )
    except Exception as e:
        print(f"\n❌ FAILED to create dataset: {e}")
        import traceback
        traceback.print_exc()
        return False
    
    # Check collected examples
    print(f"\n[3] Collection complete!")
    print(f"    Total examples collected: {len(dataset.examples)}")
    print(f"    Train examples: {len(dataset.train_examples)}")
    print(f"    Val examples: {len(dataset.val_examples)}")
    
    if len(dataset.examples) == 0:
        print("\n❌ FAILED: No examples collected!")
        return False
    
    # Print first example
    print("\n[4] First example details:")
    first_ex = dataset.examples[0]
    print(f"    Input (first 500 chars):\n{first_ex['input'][:500]}...")
    print(f"\n    Target action: {first_ex['target']}")
    
    # Verify data format
    print("\n[5] Verifying data format:")
    required_keys = ['input', 'target']
    for key in required_keys:
        if key in first_ex:
            print(f"    ✅ '{key}' present")
        else:
            print(f"    ❌ '{key}' MISSING")
            return False
    
    # Check trajectory structure
    print("\n[6] Checking trajectory structure in dataset...")
    if hasattr(dataset, 'examples'):
        # The trajectory is stored internally in the dataset's collection logic
        # We can check if the dataset was created correctly
        print(f"    ✅ Dataset created with {len(dataset.examples)} examples")
    
    # Test inference with the trained policy (if we had one)
    print("\n[7] Testing action parsing with a sample response...")
    test_response = "go to desk 1"
    test_actions = ["look", "go to desk 1", "take alarmclock 1"]
    parsed = dataset._parse_action_from_response(test_response, test_actions)
    print(f"    Response: '{test_response}'")
    print(f"    Actions: {test_actions}")
    print(f"    Parsed: '{parsed}'")
    
    if parsed == "go to desk 1":
        print("    ✅ Action parsing works!")
    else:
        print(f"    ❌ Action parsing returned '{parsed}', expected 'go to desk 1'")
    
    # Test prompt building
    print("\n[8] Testing prompt building...")
    test_instruction = "put some alarmclock on desk"
    test_obs = "You are in the middle of a room. Looking quickly around you, you see a bed 1, a desk 1..."
    test_actions = ["look", "go to desk 1", "take alarmclock 1", "put alarmclock 1 on desk 1"]
    test_trajectory = [
        {'obs': 'You see a desk', 'action': 'go to desk', 'reward': 0},
        {'obs': 'On desk you see alarmclock', 'action': 'take alarmclock', 'reward': 0}
    ]
    
    prompt = dataset._build_actpred_prompt(
        test_instruction, test_obs, test_actions, test_trajectory
    )
    print(f"    Prompt length: {len(prompt)} characters")
    print(f"    Preview (first 400 chars):\n{prompt[:400]}...")
    
    if len(prompt) > 100:
        print("    ✅ Prompt building works!")
    else:
        print("    ❌ Prompt too short, may be malformed")
    
    # Memory cleanup check
    print("\n[9] Checking memory cleanup...")
    if hasattr(dataset, 'llm_model'):
        print("    ✅ Mistral model loaded (will be cleaned up when dataset is deleted)")
    
    print("\n" + "=" * 60)
    print("✅ ACTPRED TEST PASSED! Ready for full training.")
    print("=" * 60)
    
    return True


def test_actpred_with_fake_llm():
    """
    Alternative test: Use a fake LLM (no actual Mistral download)
    Useful if you don't want to download the 7B model just for testing.
    """
    import sys
    from unittest.mock import MagicMock
    
    print("\n" + "=" * 60)
    print("TESTING ACTPRED WITH FAKE LLM (Mock)")
    print("=" * 60)
    
    # Create a mock class that doesn't load the real model
    class FakeACTPREDDataset(ACTPREDDataset):
        def __init__(self, env, **kwargs):
            # Skip parent __init__ to avoid loading Mistral
            self.env = env
            self.max_new_tokens = 50
            self.temperature = 0.0
            # Initialize the shared dataset collection logic directly.
            BaseExpertDataset.__init__(self, env, **kwargs)
        
        def _query_mistral_with_retry(self, prompt):
            # Return a fake action
            return "go to desk 1"
    
    try:
        env = VerbalizedALFWorld(split='train')
        dataset = FakeACTPREDDataset(
            env=env,
            num_episodes=1,
            max_steps=5,
            context_window=5,
            load_path=None
        )
        
        print(f"✅ Fake dataset collected {len(dataset.examples)} examples")
        if len(dataset.examples) > 0:
            print(f"   First example: {dataset.examples[0]['target']}")
            return True
        else:
            print("❌ No examples collected")
            return False
            
    except Exception as e:
        print(f"❌ Failed: {e}")
        import traceback
        traceback.print_exc()
        return False


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--fake', action='store_true', 
                       help='Use fake LLM (no Mistral download)')
    args = parser.parse_args()
    
    if args.fake:
        test_actpred_with_fake_llm()
    else:
        success = test_actpred_single_episode()
        sys.exit(0 if success else 1)