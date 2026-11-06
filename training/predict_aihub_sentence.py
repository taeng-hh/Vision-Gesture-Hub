"""Offline word prediction -> rule-based sentence; input clips are pre-segmented."""
import argparse
import json
from pathlib import Path
from core.aihub_words import AIHubWordModel

RULES = {('I', 'HEAD', 'HURT'): '저는 머리가 아파요.',
         ('HEAD', 'HURT'): '머리가 아파요.',
         ('HOSPITAL', 'GO'): '병원에 가요.',
         ('FOOD', 'REQUEST'): '밥을 주세요.',
         ('HELP', 'REQUEST'): '도와주세요.'}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('clips', type=Path, nargs='+')
    parser.add_argument('--model', type=Path, default=Path('models/ksl_aihub_first.npz'))
    args = parser.parse_args()
    model = AIHubWordModel.load(args.model)
    predictions = [model.predict(json.loads(p.read_text(encoding='utf-8'))['frames']) for p in args.clips]
    tokens = tuple(p['label'] for p in predictions)
    sentence = RULES.get(tokens, '인식 불가 또는 지원하지 않는 단어 조합입니다.')
    print(json.dumps({'tokens': tokens, 'sentence': sentence,
                      'note': 'Pre-segmented clip demo; not continuous sign translation.'},
                     ensure_ascii=False, indent=2))


if __name__ == '__main__': main()
