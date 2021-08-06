# %%
import requests
import json
from bs4 import BeautifulSoup

def get_superheros():
    url = 'http://gateway.marvel.com/v1/public/characters'
    args = {'apikey':'af18b752302271b4b7597d2abfb4c037',
            'ts': '1',
            'hash': '9d3c743d434a92951b6710ba2afa5d29'}
    headers = {'Content-Type':'application/json'}
    response = requests.get(url, params=args, headers=headers)

    print(response.status_code)
    if response.status_code== 200:
        response_json = response.json()

    return response_json

if __name__ == '__main__':
    None

superheros = get_superheros()
# print(len(superheros))
# print(type(superheros))
# print(superheros.items())
# print(superheros.keys())
# print(superheros.values())
list_sh = superheros['data']['results']
list_sh[0]

# %%    
urlHero = []
for i in range(len(list_sh)):
    for hero in list_sh[i]['urls']:
        urlHero.append(hero['url'])
urlHero    


# %%
print(type(list_sh[0]))
# %%
print(list_sh[0].keys())
# %%
print(list_sh[0].values())
# %%
beat = list_sh[0]
beat
# %%
hero = list_sh[0]['urls']
urlHero = hero[0]
urlHero    
# %%
urlHero = []
for hero in list_sh[0]['urls']:
    urlHero.append(hero['url'])
urlHero    

# %%
# 3 levels
urlHero = []
for hero in list_sh:
    for item in hero['urls']:
        print(item['url'])
        urlHero.append(item['url'])
urlHero    

# %%
# 2 levels
for hero in list_sh:
    print(hero['id'], hero['name'], hero['description'])

# %%
number = 4
# print('id           : ' + str(list_sh[number]['id']))
# print('name         : ' + list_sh[number]['name'])
# print('description  : ' + list_sh[number]['description'])
print(list_sh[number]['thumbnail'])

# %%
urlImage = []
for hero in list_sh:
    imageUrl = ''
    imageUrl += hero['thumbnail']['path']
    imageUrl += '.'
    imageUrl += hero['thumbnail']['extension']
    urlImage.append(imageUrl)
urlImage    


# %%
for hero in list_sh:
    print(hero['thumbnail'])


# %%
heros = {"data":[]}

def add_hero(hero):
    heros["data"].append(hero)

for hero in list_sh:
    imageUrl = ''
    imageUrl += hero['thumbnail']['path']
    imageUrl += '.'
    imageUrl += hero['thumbnail']['extension']
    item = {}
    item['hero_name'] = hero['name']
    item['hero_desc'] = hero['description']
    item['hero_img'] = imageUrl
    item['hero_url'] = hero['urls'][0]['url']
    add_hero(item)
    
print(heros)

# %%
heros2 = heros['data'][0]
print(heros2)
# for i in range(len(heros['data'])):
#     print(hero[i]['hero_name'])

# %%
for i in range(len(heros['data'])):
    print(heros['data'][i])
# %%
