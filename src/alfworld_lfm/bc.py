"""
Behavioral Cloning baseline - uses ALFWorld's built-in expert
"""

from base_train import BaseExpertDataset, train_imitation_learning


class BCDataset(BaseExpertDataset):
    """BC uses ALFWorld's built-in expert"""

    def _get_expert_action(self, env, instruction, obs, actions, trajectory):
        """Gets expert action from ALFWorld's built-in expert"""
        return env.get_expert_action()

def train_bc():
    train_imitation_learning(
        dataset_class=BCDataset,
        model_name="google/flan-t5-large",
        dataset_path="./src/alfworld_lfm/data/bc_dataset_500eps.pkl",
        model_save_path="./models/bc_alfworld"
    )

if __name__ == "__main__":
    train_bc()