# %%
import requests
from flask import Flask, render_template

# construct url with dinamic folders
def url_fun(*args):
    url = 'http://gateway.marvel.com/'
    return url + '/'.join(args)

# carga de claves
def get_key(key):
    with open('C:/Windows/AppsKeys/Marvel/'+key+'.key', 'r') as file_read:
        app_key = file_read.readline()
    return app_key

def get_superheros():
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
            'limit': 100
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


app = Flask(__name__)

@app.route('/')
def home():
    superheros = get_superheros()
    list_sh = superheros['data']['results']

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

    longitud = len(list_sh)

    return render_template('home.html', heros=heros, longitud=longitud)

@app.route('/about') 
def about():
    notas = ("nota1","nota2","nota3","nota4","nota5","nota6")
    return render_template('about.html', notas=notas)

if __name__ == '__main__':
    app.run(debug=True)
