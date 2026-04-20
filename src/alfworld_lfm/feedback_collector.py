import yaml
import tqdm
import copy
import random
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


SEED = 37
GEMINI_API_KEY = "AIzaSyBZiupbLWPzSqtvJ-g9cG9gO6vDbiu0U08"

PROMPT = """
You will be shown a playthrough for solving a task. Afterwards, answer some questions about how you performed.
{window}
----------
Did you see anything that makes you believe you are on the right track to solve the task? Answer yes or no. If yes, then list ALL the steps when you saw helpul things by the step number. List only in bullet form.
""".strip()

def connect_gemini():
    # Configure your API key
    client = genai.Client(api_key=GEMINI_API_KEY)
    return client

@T.retry(stop=T.stop_after_attempt(20), wait=T.wait_fixed(10), after=lambda s: logging.error(repr(s)))
def get_gemini_response(prompt, client):
    response = client.models.generate_content(
        model='gemini-2.5-flash-lite',
        contents=prompt,
        config=types.GenerateContentConfig(
            max_output_tokens= 256,
        ),
    )
    return response.text


def get_windows(trajectories, win_len=20):
    windows = []
    for traj in trajectories:
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

def get_all_feedback(trajectories, dry_run=False):
    client = connect_gemini()
    limit = 10000
    window_prompts = get_windows(trajectories)
    window_prompts = window_prompts[:limit]
    feedback_dataset = []

    save_path = "./llm_feedback"
    if not os.path.isdir(save_path):
        os.makedirs(save_path)

    for i, prompt in tqdm.tqdm(enumerate(window_prompts), total=len(window_prompts)):
        if not dry_run:
                try:
                    response = get_gemini_response(prompt, client)
                    print(f"Gemini response: {response}")
                    llm_feedback = {'prompt': prompt, 'response': response}

                except Exception as e:
                    print(e)
                    continue
                else:
                    feedback_dataset.append(llm_feedback)
                    fout = os.path.join(save_path, 'feedback.{}.json.bz2'.format(i))

                    with bz2.open(fout, 'wt') as f:
                        json.dump(llm_feedback, f, indent=2)
    
    print('done!')
    return feedback_dataset


def main():
    TRAJ_DIR = ''

    trajectories = load_trajectories(TRAJ_DIR)
    feedback_data = get_all_feedback(trajectories)


if __name__ == "__main__":
    main()