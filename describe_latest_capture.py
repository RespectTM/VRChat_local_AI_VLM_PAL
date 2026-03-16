import os
import glob
from ollama_client import query

MODEL = 'moondream:latest'
QUESTION = 'Describe what is happening in this VRChat scene in detail.'


def latest_capture(path='captures'):
    files = glob.glob(os.path.join(path, '*.png')) + glob.glob(os.path.join(path, '*.jpg'))
    if not files:
        return None
    files.sort(key=os.path.getmtime, reverse=True)
    return files[0]


def describe_image(path: str, question: str = QUESTION) -> str:
    return query(MODEL, question, image_path=path)


def main():
    path = latest_capture()
    if not path:
        print('No captures found in captures/. Run capture_vrchat.py first.')
        return

    print('Latest capture:', path)
    print('Querying', MODEL, '...')
    answer = describe_image(path)
    print('\n--- moondream ---')
    print(answer)
    print('-----------------')


if __name__ == '__main__':
    main()
