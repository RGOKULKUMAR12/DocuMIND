from sentence_transformers import SentenceTransformer 

try :
    model =SentenceTransformer ("sentence-transformers/all-MiniLM-L6-v2",local_files_only =True )
except Exception :
    model =SentenceTransformer ("sentence-transformers/all-MiniLM-L6-v2")

def embed (chunks ):

    if not chunks :
        return []

    texts =[c ["content"]if isinstance (c ,dict )else c for c in chunks ]

    embeddings =model .encode (
    texts ,
    convert_to_numpy =True ,
    normalize_embeddings =True ,
    show_progress_bar =False 
    )
    return embeddings 

def embed_query (query ):

    query_embedding =model .encode (
    query ,
    convert_to_numpy =True ,
    normalize_embeddings =True ,
    show_progress_bar =False 
    )
    return query_embedding 

def calculate_similarity (query_embedding ,chunk_embeddings ):

    similarities =model .similarity (
    query_embedding ,
    chunk_embeddings 
    )
    return similarities
