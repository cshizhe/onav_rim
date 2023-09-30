import os
import argparse
import json
import jsonlines
import numpy as np
import gzip
import collections

def eval_result_file(
    result_file, force_stop=False,  max_steps=500,
):
    scores = collections.defaultdict(list)
    oracle_scores = collections.defaultdict(list)
    episode_ids = set()
    scene_ids = set()
    action_frac = collections.Counter()
    nwrong = 0
    with jsonlines.open(result_file, 'r') as f:
        for x in f:
            if np.isinf(x['infos'][0]['distance_to_goal']):
                nwrong += 1
                continue

            if x['episode_id'] in episode_ids:
                print('repeat', x['episode_id'])
                continue

            episode_ids.add(x['episode_id'])
            scene_ids.add(x['episode_id'].split('_')[0])
            scores['num_steps'].append(len(x['actions']))
            scores['collisions'].append(sum([info['collisions']['is_collision'] for info in x['infos']]))
            scores['start_dists'].append(x['infos'][0]['distance_to_goal'])
            action_frac.update(x['actions'])
            
            if max_steps < 500:
                x['infos'] = x['infos'][:min(max_steps, len(x['infos']))]
            if x['infos'][-1]['spl'] == 1:
                x['infos'][-1]['softspl'] = 1
            for k, v in x['infos'][-1].items():
                if force_stop and k == 'success':
                    v = x['infos'][-1]['distance_to_goal'] < 0.1
                if k != 'collisions':
                    scores[k].append(v)

            for t, info in enumerate(x['infos']):
                if info['distance_to_goal'] < 0.1:
                    oracle_scores['success'].append(1)
                    oracle_scores['num_steps'].append(t+1)
                    break
            else:
                oracle_scores['num_steps'].append(len(x['infos']))
                oracle_scores['success'].append(0)

    action_sum = np.sum(list(action_frac.values()))
    for a, c in action_frac.most_common():
        print(a, '%.2f'%(c/action_sum*100))
                
    print(sorted(list(scene_ids)))
    print('#episodes: %d'%(len(scores['num_steps'])))
    if nwrong > 0:
        print('\t#wrong start episodes', nwrong)
    print('min steps', np.min(scores['num_steps']))
    for k, v in scores.items():
        v = np.array(v)
        num_nans = np.sum(np.isnan(v))
        if num_nans > 0:
            print('\tNaN', num_nans)
            # print(v)
        print(k, np.mean(v[~np.isnan(v)]))

    print('\noracle scores')
    for k, v in oracle_scores.items():
        print(k, np.mean(v))
    print()

    out_keys = ['num_steps', 'collisions', 'distance_to_goal', 'success', 'spl', 'softspl', 'goal_vis_pixels', 'oracle_success']
    out_values = []
    for key in out_keys[:-1]:
        value = np.array(scores[key])
        value = np.mean(value[~np.isnan(value)])
        if key in ['success', 'spl', 'softspl', 'goal_vis_pixels']:
            value *= 100
        out_values.append(value)
    out_values.append(np.mean(oracle_scores['success']) * 100)
    print(','.join(out_keys))
    print(','.join(['%.2f'%v for v in out_values]))

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('result_file')
    parser.add_argument('--force_stop', action='store_true', default=False, help='force the agent to stop at the end of the episode')
    parser.add_argument('--max_steps', default=500, type=int)
    args = parser.parse_args()

    eval_result_file(
        args.result_file, force_stop=args.force_stop, 
        max_steps=args.max_steps,
    )

if __name__ == '__main__':
    main()

