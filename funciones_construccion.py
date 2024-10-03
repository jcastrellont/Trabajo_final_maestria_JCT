import requests
import xml.etree.ElementTree as ET
from datetime import datetime
import pandas as pd
import pandas as pd
import numpy as np
import nltk
import string
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
from nltk.stem import WordNetLemmatizer
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import silhouette_score
from keybert import KeyBERT
import warnings
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt
import networkx as nx
import community
import community.community_louvain as cl
from sklearn.metrics.pairwise import cosine_similarity

################################################################################
################################################################################
# FUNCIONES 
################################################################################
################################################################################


################################################################################
## Funciones para busqueda
################################################################################

# Función para realizar la búsqueda en Arxiv
def search_arxiv(query, max_results=1000):
    base_url = "http://export.arxiv.org/api/query?"
    params = {
        "search_query": query,  # Término de búsqueda
        "start": 0,             # Desde qué resultado empezar
        "max_results": max_results,  # Número máximo de resultados
        "sortBy": "relevance",   # Ordenar por relevancia
        "sortOrder": "descending"  # Orden descendente
    }
    
    response = requests.get(base_url, params=params)
    
    if response.status_code == 200:
        return response.text  # La respuesta en formato XML
    else:
        print(f"Error al hacer la consulta: {response.status_code}")
        return None

# Función para parsear y extraer títulos, abstracts y fechas de publicación (eliminado DOI)
def extract_paper_data(xml_data):
    root = ET.fromstring(xml_data)
    ns = {'arxiv': 'http://www.w3.org/2005/Atom'}  # Espacio de nombres en XML
    
    papers = []
    
    for entry in root.findall('arxiv:entry', ns):
        title = entry.find('arxiv:title', ns).text.strip()
        abstract = entry.find('arxiv:summary', ns).text.strip()
        published_date = entry.find('arxiv:published', ns).text.strip()
        
        papers.append({
            "title": title, 
            "abstract": abstract, 
            "published_date": published_date
        })
    
    return papers

# Función para filtrar por fecha
def filter_by_date(papers, start_date=None, end_date=None):
    if start_date:
        start_date = datetime.strptime(start_date, "%Y-%m-%d")
    if end_date:
        end_date = datetime.strptime(end_date, "%Y-%m-%d")
    
    filtered_papers = []
    for paper in papers:
        pub_date = datetime.strptime(paper['published_date'], "%Y-%m-%dT%H:%M:%SZ")
        if (not start_date or pub_date >= start_date) and (not end_date or pub_date <= end_date):
            filtered_papers.append(paper)
    
    return filtered_papers

# Función principal para buscar, filtrar y convertir a DataFrame
def get_arxiv_papers_df(query, max_results=1000, start_date=None, end_date=None):
    print(f"Buscando artículos en Arxiv sobre '{query}'...")
    xml_data = search_arxiv(query, max_results)
    
    if xml_data:
        papers = extract_paper_data(xml_data)
        print(f"Se han encontrado {len(papers)} artículos.")
        
        # Aplicar el filtro por fecha
        if start_date or end_date:
            papers = filter_by_date(papers, start_date, end_date)
            print(f"Se han encontrado {len(papers)} artículos después de aplicar el filtro de fecha.")
        
        # Convertir a DataFrame
        df = pd.DataFrame(papers)
        return df
    else:
        print("No se pudo obtener información.")
        return pd.DataFrame()

# Función para solicitar datos al usuario
def user_input():
    query = input("Introduce el tema de búsqueda: ")
    start_date = input("Introduce la fecha de inicio (YYYY-MM-DD): ")
    end_date = input("Introduce la fecha de fin (YYYY-MM-DD): ")
    max_results = int(input("Introduce el número máximo de resultados: "))
    name = input("Asigne un nombre para el Mapa de conocimiento generado: ") 
    
    return query, start_date, end_date, max_results, name

################################################################################
## Procesamiento de datos
################################################################################

def process_text(text):
    stop_words = set(stopwords.words('english'))
    ps = PorterStemmer()
    lemmatizer = WordNetLemmatizer()
    # Tokenization
    tokens = text.split()
    
    # Remove punctuation and lower casing
    tokens = [word.lower() for word in tokens if word.isalnum()]
    
    # Stop words removal
    filtered_tokens = [word for word in tokens if word not in stop_words]
    
    # Stemming
    stemmed_tokens = [ps.stem(word) for word in filtered_tokens]
    
    # Lemmatization
    lemmatized_tokens = [lemmatizer.lemmatize(word) for word in stemmed_tokens]
    
    # Convert tokens back to a single string
    processed_text = ' '.join(lemmatized_tokens)
    
    return processed_text

################################################################################
## Palabras Clave
################################################################################

### TF-IDF

def tfidf_extractor(carac,data):
    docs=data[carac+"_transformed"].tolist()
    cv=CountVectorizer(max_df=0.85,stop_words='english')
    word_count_vector=cv.fit_transform(docs)
    tfidf_transformer=TfidfTransformer(smooth_idf=True,use_idf=True)
    tfidf_transformer.fit(word_count_vector)
    def sort_coo(coo_matrix):
        tuples = zip(coo_matrix.col, coo_matrix.data)
        return sorted(tuples, key=lambda x: (x[1], x[0]), reverse=True)
    def extract_topn_from_vector(feature_names, sorted_items, topn=10):
        """get the feature names and tf-idf score of top n items"""
        
        #use only topn items from vector
        sorted_items = sorted_items[:topn]
    
        score_vals = []
        feature_vals = []
        
        # word index and corresponding tf-idf score
        for idx, score in sorted_items:
            
            #keep track of feature name and its corresponding score
            score_vals.append(round(score, 3))
            feature_vals.append(feature_names[idx])
    
        #create a tuples of feature,score
        #results = zip(feature_vals,score_vals)
        results= {}
        for idx in range(len(feature_vals)):
            results[feature_vals[idx]]=score_vals[idx]
        
        return results
    # you only needs to do this once, this is a mapping of index to 
    feature_names=cv.get_feature_names_out()
    # Añade las columnas para los tres principales keywords
    data[carac+'_keyword_TFIDF1'] = ''
    data[carac+'_keyword_TFIDF2'] = ''
    data[carac+'_keyword_TFIDF3'] = ''
    
    # Itera sobre cada fila del DataFrame
    for i in range(len(data)):
        doc = data.loc[i, carac+"_transformed"]
    
        # Genera tf-idf para el documento actual
        tf_idf_vector = tfidf_transformer.transform(cv.transform([doc]))
    
        # Ordena los vectores tf-idf por puntaje en orden descendente
        sorted_items = sort_coo(tf_idf_vector.tocoo())
    
        # Extrae los tres principales keywords
        keywords = extract_topn_from_vector(feature_names, sorted_items, 3)
    
        # Asigna los keywords a las columnas correspondientes
        try:
            data.at[i, carac+'_keyword_TFIDF1'] = list(keywords.keys())[0]
        except:
            data.at[i, carac+'_keyword_TFIDF1'] = ''
        try:
            data.at[i, carac+'_keyword_TFIDF2'] = list(keywords.keys())[1]
        except:
            data.at[i, carac+'_keyword_TFIDF2'] = ''
        try:
            data.at[i, carac+'_keyword_TFIDF3'] = list(keywords.keys())[2]
        except:
            data.at[i, carac+'_keyword_TFIDF3'] = ''
    return(data)

### KeyBERT

def keybert_keywords(carac,data):
    kw_model = KeyBERT()
    aa=kw_model.extract_keywords(docs=list(data[carac+'_transformed']), keyphrase_ngram_range=(1,3))
    results = [[key for key, prob in sorted(sublist, key=lambda x: x[1], reverse=True)[:1]] for sublist in aa]
    data[carac+'_keyword_keybert']=results
    data[carac+'_keyword_keybert']=data[carac+'_keyword_keybert'].str[0]
    return(data)

################################################################################
## KMEANS
################################################################################

def kmeans(dataframe, clusters=30):
    df_clusters=dataframe[['keywords_total']]
    df_clusters['keywords_total']=df_clusters['keywords_total'].astype(str)
    #define vectorizer parameters
    vectorizer = TfidfVectorizer(ngram_range=(1,1))
    
    # Generate matrix of word vectors
    tfidf_matrix = vectorizer.fit_transform(df_clusters['keywords_total'])
    ###############################################
    # k-means clustering
    ###############################################
    

    
    mod = KMeans(clusters, random_state=123)
    
    model=mod.fit(tfidf_matrix )
    dataframe['cluster_kmeans']=model.predict(tfidf_matrix)
    dataframe['cluster_kmeans']='cluster_'+dataframe['cluster_kmeans'].astype(str)
    return dataframe

################################################################################
## Procesamiento Final
################################################################################

def final_process(dataframe):
    data_agrup = dataframe.groupby('cluster_kmeans').agg({
                                                            'title': ' '.join,
                                                            'abstract': ' '.join
                                                        }).reset_index()
    data_agrup['total_text']=data_agrup['title']+' '+data_agrup['abstract']
    data_agrup["total_text_tr"] = data_agrup["total_text"].apply(process_text)
    kw_model = KeyBERT()
    aa=kw_model.extract_keywords(docs=list(data_agrup['total_text_tr']), keyphrase_ngram_range=(1,3))
    results = [[key for key, prob in sorted(sublist, key=lambda x: x[1], reverse=True)[:1]] for sublist in aa]
    data_agrup['Grupo_keyBERT']=results
    data_agrup['Grupo_keyBERT']=data_agrup['Grupo_keyBERT'].str[0]
    return(data_agrup)

################################################################################
## Indicadores de similitud
################################################################################

def similarities_df(data_agrup):
    similarities=data_agrup[['cluster_kmeans','Grupo_keyBERT']]
    similarities['key']=1
    cross=data_agrup[['Grupo_keyBERT']].rename(columns={'Grupo_keyBERT':'Grupo_keyBERT_2'})
    cross['key']=1
    similarities=similarities.merge(cross).drop('key',axis=1)
    similarities=similarities[similarities['Grupo_keyBERT']!=similarities['Grupo_keyBERT_2']].reset_index(drop=True)
    # Vectorización TF-IDF
    vectorizer = TfidfVectorizer()
    
    # Convertir la columna 'texto1' en una matriz TF-IDF
    tfidf_matrix1 = vectorizer.fit_transform(similarities['Grupo_keyBERT'])
    
    # Convertir la columna 'texto2' en una matriz TF-IDF
    tfidf_matrix2 = vectorizer.transform(similarities['Grupo_keyBERT_2'])
    
    # Calcular la similitud coseno entre los vectores de 'texto1' y 'texto2'
    cosine_similarities = []
    for i in range(len(similarities)):
        cosine_sim = cosine_similarity(tfidf_matrix1[i], tfidf_matrix2[i])
        cosine_similarities.append(cosine_sim[0][0])
    
    # Agregar la similitud coseno como una nueva columna en el DataFrame
    similarities['sim_coseno'] = cosine_similarities
    similarities['disim_coseno']=1-similarities['sim_coseno']
    return similarities

################################################################################
## Resultado final
################################################################################

def mapas_conocimiento(dataframe):
    dataframe["title_transformed"] = dataframe["title"].apply(process_text)
    dataframe["abstract_transformed"] = dataframe["abstract"].apply(process_text)
    dataframe=tfidf_extractor('abstract',dataframe)
    dataframe=tfidf_extractor('title',dataframe)
    dataframe=keybert_keywords('abstract',dataframe)
    dataframe=keybert_keywords('title',dataframe)
    dataframe['keywords_total']=dataframe['abstract_keyword_TFIDF1']+' '+dataframe['abstract_keyword_TFIDF2']+' '+dataframe['abstract_keyword_TFIDF3']+' '+dataframe['title_keyword_TFIDF1']\
                        +' '+dataframe['title_keyword_TFIDF2']+' '+dataframe['title_keyword_TFIDF3']+' '+dataframe['abstract_keyword_keybert']+' '+dataframe['title_keyword_keybert']
    dataframe=kmeans(dataframe)
    data_agrup=final_process(dataframe)
    data_agrup=similarities_df(data_agrup)
    return data_agrup

################################################################################
## Mapas de conocimiento
################################################################################

def mapa(similarities, name):
    top_frame=similarities[similarities['sim_coseno']!=0]
    top_frame['sim_coseno']=(top_frame['sim_coseno']-top_frame['sim_coseno'].min())/(top_frame['sim_coseno'].max()-top_frame['sim_coseno'].min())
    edges = list(zip(top_frame['Grupo_keyBERT'], top_frame['Grupo_keyBERT_2']))
    weighted_edges = list(zip(top_frame['Grupo_keyBERT'], top_frame['Grupo_keyBERT_2'], top_frame['sim_coseno']))
    nodes = list(set(top_frame['Grupo_keyBERT']).union(set(top_frame['Grupo_keyBERT_2'])))
    G = nx.Graph()
    G.add_nodes_from(nodes)
    G.add_edges_from(edges)
    G.add_weighted_edges_from(weighted_edges)
    partition = cl.best_partition(G)
    modularity = cl.modularity(partition, G)
    pos = nx.spring_layout(G, dim=2)
    community_id = [partition[node] for node in G.nodes()]
    fig = plt.figure(figsize=(30,30))
    nx.draw(G, pos,with_labels=True, edge_color = ['black']*len(G.edges()), cmap=plt.cm.tab20,
            node_color=community_id, node_size=20550)
    plt.savefig(str(name)+'.png')
    return plt
