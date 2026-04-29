from base_train import BaseExpertDataset, train_imitation_learning
from bc import BCDataset

OUTPUT_PATH = 
def create_desirable_behavior_dataset():
    data = []
    output_dir = Path(OUTPUT_PATH)
    
    for f in output_dir.glob('*.json.bz2'):
        with bz2.open(f, 'rt') as file:
            traj = json.load(file)
            for ex in traj:
                if ex['llm_pred']:
                    # kept.add(fname)
                    data.append(dict(input=ex['prompt'], target=ex['action']))
    return data

def train_model_with_feedback(feedback_data):
    feedback_data = create_desirable_behavior_dataset()
    train_imitation_learning(
        dataset_class=BCDataset,
        model_name="google/flan-t5-large",
        dataset_path="./src/alfworld_lfm/data/bc_dataset_500eps.pkl",
        model_save_path="./models/lfm_alfworld",
        include_feedback=True,
        feedback_data=feedback_data
    )

if __name__ == "__main__":
    train_model_with_feedback()