import os
import requests
import pandas as pd
from flask import Flask, render_template

this_folder = os.path.dirname(os.path.abspath(__file__))
data = pd.read_csv(this_folder + '/outputs/marvel_data.csv')

# # construct url with dinamic folders
# def url_fun(*args):
#     url = 'http://gateway.marvel.com/'
#     return url + '/'.join(args)

# # carga de claves
# def get_key(key):
#     with open('C:/Windows/AppsKeys/Marvel/'+key+'.key', 'r') as file_read:
#         app_key = file_read.readline()
#     return app_key

# # solicitud de conexion a la api de marvel
# def get_superheros():
#     # get credentials
#     apikey = get_key('apikey')
#     ts = get_key('ts')
#     hash = get_key('hash')
#     # get url full path
#     url = url_fun('v1', 'public', 'characters')
#     # request data from url
#     args = {'apikey': apikey,
#             'ts': ts,
#             'hash': hash,
#             'limit': 100
#             }
#     headers = {'Content-Type': 'application/json'}
#     try:
#         response = requests.get(url, params=args, headers=headers)
#         response.raise_for_status()
#         if response.status_code == 200:
#             response = response.json()
#             return response
#     except requests.exceptions.HTTPError as error:
#         print(error.response.text)
#     except requests.exceptions.RequestException as error:
#         print(error.response.text)
#     return

# salvar data en archivo.csv
# def save_dictionary_as_csv(dictionary, lista):
#     # built pandas tabular data with de dictironary useful data
#     df_headers = ('Name','Description','Image_URL','URL')
#     df_heros = pd.DataFrame(dictionary['data'][0].values()).transpose()
#     for i in range(1, len(lista)):
#         df_next = pd.DataFrame(dictionary['data'][i].values()).transpose()
#         df_heros = df_heros.append(df_next, ignore_index=True)
#     df_heros.columns = df_headers
#     # save tabular data into a csv file
#     df_heros.to_csv(this_folder + "/outputs/marvel_heros.csv", index=False)
#     return

# carga data desde archivo.csv
def get_data_from_csv():
    df = pd.read_csv(this_folder + '/outputs/marvel_heros.csv')
    df
    return df

# instanciación de Flask
app = Flask(__name__)

# definición del root de la aplicación
@app.route('/')
@app.route('/home')
def home():
    return render_template('index.html')

# Get character from csv file
@app.route('/api_request') 
def api_request():
    df_characters = get_data_from_csv()

    heros = {"data":[]}
    def add_hero(hero):
        heros["data"].append(hero)

    for index, row in df_characters.iterrows():
        item = {}
        item['hero_name'] = row.Name
        item['hero_desc'] = row.Description
        item['hero_img'] = row.Image_URL
        item['hero_url'] = row.URL
        add_hero(item)
        
    longitud = len(df_characters)
    
    return render_template('api_request.html', heros=heros, longitud=longitud)

# # This section get data directly from the Marvel Api insted of from csv file
# @app.route('/api_request') 
# def api_request():
#     superheros = get_superheros()
#     list_sh = superheros['data']['results']

#     heros = {"data":[]}
#     def add_hero(hero):
#         heros["data"].append(hero)

#     for hero in list_sh:
#         imageUrl = ''
#         imageUrl += hero['thumbnail']['path']
#         imageUrl += '.'
#         imageUrl += hero['thumbnail']['extension']
#         item = {}
#         item['hero_name'] = hero['name']
#         item['hero_desc'] = hero['description']
#         item['hero_img'] = imageUrl
#         item['hero_url'] = hero['urls'][0]['url']
#         add_hero(item)
#     longitud = len(list_sh)
#     # save data into a csv file
#     save_dictionary_as_csv(heros, list_sh)
#     return render_template('api_request.html', heros=heros, longitud=longitud)

# definición de la pagina de data visualization
@app.route('/data_vis')
def data_vis():
    dictionary = []
    # wordCloud Columns Names
    cols_names = pd.DataFrame(data.columns)
    cols_names.rename(columns={0:'x'}, inplace=True)
    cols_names['value'] = 1
    dictionary.append(cols_names.to_dict('records'))
    
    labels = []
    values = []
    # Barchart gender distribution
    df = data.groupby(['Gender']).agg({'Count':'sum'}).reset_index()
    labels.append(df.Gender.to_list())
    values.append(df.Count.to_list())
    
    # Barchart Alignment distribution
    df = data.groupby(['Alignment']).agg({'Count':'sum'}).reset_index()
    labels.append(df.Alignment.to_list())
    values.append(df.Count.to_list())

    # wordCloud Race distribution
    df = data.groupby(['Race']).agg({'Count':'sum'}).reset_index()
    df.rename(columns={'Race':'x', 'Count':'value'}, inplace=True)
    dictionary.append(df.to_dict('records'))
    
    # columnChart    
    table = pd.pivot_table(data, values='Power_Rank', index='Gender', columns='Alignment', aggfunc='sum').reset_index()
    table = table.fillna(0)
    headers = list(table.columns.values)
    rows = table.values.tolist()

    # horizontalBarChart level of power
    cols = ['Alignment','Intelligence', 'Strength', 'Speed', 'Durability', 'Power', 'Combat']
    df_temp = data[cols]
    df = df_temp.groupby(['Alignment']).sum()
    dft = df.T.reset_index()
    dft.rename(columns={'index':'Alignment'}, inplace=True)
    dft.drop(columns='neutral', axis=1, inplace=True)
    levelOfPower = dft.values.tolist()
    
    return render_template('data_vis.html', labels=labels, values=values, dictionary=dictionary, headers=headers, rows=rows, levelOfPower=levelOfPower)

# definición de la pagina de modelo supervisado
@app.route('/sml_model')
def sml_model():
    return render_template('sml_model.html')

# definición de pagina acerca de...
@app.route('/about') 
def about():
    return render_template('about.html')

if __name__ == '__main__':
    app.run()
    #app.run(debug=True)
