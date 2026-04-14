"""
Test script for VerbalizeALFWorld environment wrapper
"""

from environment import VerbalizedALFWorld

def test_environment():
    print("Testing VerbalizedALFWorld wrapper!")

    # 1. Create environment
    print("\n[1] Creating environment...")
    env = VerbalizedALFWorld(split='train')

    # 2. Test reset
    print("\n[2] Testing reset...")
    instruction, obs, admissible_actions = env.reset()
    print("Instruction:", instruction[:100], "...")  # Print first 100 chars of instruction
    print("Observation:", obs[:100], "...")  # Print first 100 chars of observation
    print("Number of admissible actions:", len(admissible_actions))

    # 3. Test step() with a valid action
    print("\n[3] Testing step() with a valid action...")
    if admissible_actions:
        action = admissible_actions[0]  # Take the first admissible action
        print("Taking action:", action)
        instruction, obs, reward, done, admissible_actions = env.step(action)
        print("Reward:", reward)
        print("Done:", done)
        print("New Observation:", obs[:100], "...")
    else:
        print("No admissible actions available to test step()")

    # 4. Test verbalize() method
    print("\n[4] Testing verbalize() method...")
    sample_obs = "You are in a kitchen. There is a fridge and a stove."
    verbalized = env.verbalize(sample_obs)
    print("Original Observation:", sample_obs)
    print("Verbalized Observation:", verbalized)

if __name__ == "__main__":
    test_environment()