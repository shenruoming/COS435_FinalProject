import yaml
import tqdm
from pathlib import Path
from environment import VerbalizedALFWorld
from google import genai
from google.genai import types
from utils import format_context
import bz2
import json
import tenacity as T
import logging
import os
from evaluate import load_trajectories
import argparse
from llm_critic import LLM_Critic


SEED = 37

PROMPT = """
You will be shown a playthrough for solving a task. Afterwards, answer some questions about how you performed.
{window}
----------
Did you see anything that makes you believe you are on the right track to solve the task? Answer yes or no. If yes, then list ALL the steps when you saw helpul things by the step number. List only in bullet form.
""".strip()

def get_windows(trajectories, win_len=20):
    windows = []
    for t in trajectories:
        traj = t['traj']
        if len(traj) == 0:
            continue
        # TODO: get instruction (how to store in trajectory?)
        task = None
        for start in range(1, len(traj), win_len):
            end = min(len(traj), start+win_len)
            window = traj[start:end]
            step_tm1 = traj[start-1]
            task = step_tm1['after']['obs']['instruction']
            replay = ['Task: {}\nBefore: {}'.format(task, step_tm1['after']['obs']['observation'])]
            for i, step in enumerate(window):
                replay.append('-' * 10)
                replay.append('Step {}'.format(start+i))
                replay.append('Your action: {}'.format(step['action']))
                replay.append('Result: {}'.format(step['after']['obs']['observation']).strip())
                step_tm1 = step
            prompt = '\n'.join(replay)
            windows.append(PROMPT.format(window=prompt))
    return windows

def get_all_feedback(trajectories, output_path):
    llm = LLM_Critic()
    limit = 10000
    window_prompts = get_windows(trajectories)
    print(window_prompts)
    window_prompts = window_prompts[:limit]
    feedback_dataset = []

    if not os.path.isdir(output_path):
        os.makedirs(output_path)

    for i, prompt in tqdm.tqdm(enumerate(window_prompts), total=len(window_prompts)):
        try:
            response = llm.query_llm(prompt)
            llm_feedback = {'prompt': prompt, 'response': response}

        except Exception as e:
            print(e)
            continue
        else:
            feedback_dataset.append(llm_feedback)
            fout = os.path.join(output_path, 'feedback.{}.json.bz2'.format(i))

            with bz2.open(fout, 'wt') as f:
                json.dump(llm_feedback, f, indent=2)
    
    print('done!')
    return feedback_dataset


def main():
    
    parser = argparse.ArgumentParser(description='Collect LLM feedback on trajectories')
    parser.add_argument('--trajectories_path', type=str, default='./eval_results/bc_alfworld_final_eval_out_of_distribution',
                        help='Path to saved trajectories')
    parser.add_argument('--no_save', action='store_true',
                        help='Do not save trajectories')
    parser.add_argument('--output_dir', type=str, default='./llm_feedback_new',
                        help='Directory to save trajectories')
    
    args = parser.parse_args()

    trajectories = load_trajectories(args.trajectories_path)
    feedback_data = get_all_feedback(trajectories, output_path=args.output_dir)


if __name__ == "__main__":
    main()