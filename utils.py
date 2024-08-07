import requests
from pathlib import Path

if not Path('input.txt').is_file():
    print('Downloading \'input.txt\'...')
    with open('input.txt', 'wb') as f:
        request = requests.get('https://github.com/karpathy/ng-video-lecture/raw/master/input.txt')
        f.write(request.content)
else:
    print('\'input.txt\' already exists, skipping download')

# if not Path('more.txt').is_file():
#     print('Downloading \'more.txt\'...')
#     with open('more.txt', 'wb') as f:
#         request = requests.get('https://github.com/karpathy/ng-video-lecture/raw/master/more.txt')
#         f.write(request.content)
# else:
#     print('\'more.txt\' already exists, skipping download')