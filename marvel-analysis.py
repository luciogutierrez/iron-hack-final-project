# %%
import requests
import json
import pandas as pd
# global variable
response_dict = {}

# %%
# construct url with dinamic folders
def url_fun(*args):
    url = 'http://gateway.marvel.com/'
    return url + '/'.join(args)

# get credentials


def get_key(key):
    with open('C:/Windows/AppsKeys/Marvel/'+key+'.key', 'r') as file_read:
        app_key = file_read.readline()
    return app_key

# get data from url


def get_data_from_url():
    # get credentials
    apikey = get_key('apikey')
    ts = get_key('ts')
    hash = get_key('hash')
    # get url full path
    url = url_fun('v1', 'public', 'characters')
    # request data from url
    args = {'apikey': apikey,
            'ts': ts,
            'hash': hash,
            'limit':50
            }
    headers = {'Content-Type': 'application/json'}
    try:
        response = requests.get(url, params=args, headers=headers)
        response.raise_for_status()
        if response.status_code == 200:
            response = response.json()
            return response
    except requests.exceptions.HTTPError as error:
        print(error.response.text)
    except requests.exceptions.RequestException as error:
        print(error.response.text)
    

# %%
response_dict = get_data_from_url()
# print(type(response_dict))
# print(len(response_dict))
# print(response_dict.items())
# print(response_dict.keys())
# print(response_dict.values())

# %%
# print(type(response_dict['data']))
# print(response_dict['data'].keys())
# print(type(response_dict['data']['results']))
# print(len(response_dict['data']['results']))
print(response_dict['data']['results'][49].keys())
# print(response_dict['data']['results'][1])

# %%
print(response_dict['data']['results'][0]['urls'])
# %%
list_data_results = response_dict['data']['results']
list_data_results[0]

# %%
heros = {"data": []}


def add_hero(hero):
    heros["data"].append(hero)


for hero in list_data_results:
    imageUrl = ''
    imageUrl += hero['thumbnail']['path']
    imageUrl += '.'
    imageUrl += hero['thumbnail']['extension']
    item = {}
    item['hero_id'] = hero['id']
    item['hero_name'] = hero['name']
    item['hero_desc'] = hero['description']
    item['modified'] = hero['modified']
    item['hero_img'] = imageUrl
    item['hero_url'] = hero['urls'][0]['url']
    add_hero(item)

print(heros)
# %%
df_headers = ('Id', 'Name', 'Description', 'Modified', 'Image URL', 'URL')
df_heros = pd.DataFrame(heros['data'][0].values()).transpose()
for i in range(1, len(list_data_results)):
    df_next = pd.DataFrame(heros['data'][i].values()).transpose()
    df_heros = df_heros.append(df_next, ignore_index=True)
df_hero.columns = df_headers
print(df_hero)

# %%
df_heros.to_csv("./outputs/marvel-heros.csv")

# %%
