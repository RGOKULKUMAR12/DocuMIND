from pathlib import Path 
from typing import List ,Dict ,Any ,Optional 
import numpy as np 
import chromadb 
from chromadb .config import Settings 

DEFAULT_DB_PATH =Path ("./vector_db")
COLLECTION_NAME ="documind"

def get_client (persist_dir :str |Path =DEFAULT_DB_PATH )->chromadb .PersistentClient :

    db_path =Path (persist_dir ).resolve ()
    db_path .mkdir (parents =True ,exist_ok =True )
    return chromadb .PersistentClient (path =str (db_path ))

def get_collection (persist_dir :str |Path =DEFAULT_DB_PATH ):

    client =get_client (persist_dir )
    collection =client .get_or_create_collection (
    name =COLLECTION_NAME ,
    metadata ={"hnsw:space":"cosine"}
    )
    return collection 

def add_documents (
chunks :List [Dict [str ,Any ]],
embeddings :Any ,
source :str ="unknown",
persist_dir :str |Path =DEFAULT_DB_PATH 
)->int :

    if not chunks :
        return 0 

    collection =get_collection (persist_dir )

    if isinstance (embeddings ,np .ndarray ):
        embeddings_list =embeddings .tolist ()
    else :
        embeddings_list =[
        e .tolist ()if isinstance (e ,np .ndarray )else list (e )
        for e in embeddings 
        ]

    ids :List [str ]=[]
    documents :List [str ]=[]
    metadatas :List [Dict [str ,Any ]]=[]

    for i ,chunk in enumerate (chunks ):
        if isinstance (chunk ,dict ):
            content =chunk .get ("content","")
            page_number =int (chunk .get ("page_number",1 ))
            chunk_id =int (chunk .get ("chunk_id",i +1 ))
            chunk_source =str (chunk .get ("source",source ))
        else :

            content =str (chunk )
            page_number =1 
            chunk_id =i +1 
            chunk_source =source 

        unique_id =f"{chunk_source}_p{page_number}_c{chunk_id}"

        ids .append (unique_id )
        documents .append (content )
        metadatas .append ({
        "source":chunk_source ,
        "page_number":page_number ,
        "chunk_id":chunk_id 
        })

    collection .upsert (
    ids =ids ,
    embeddings =embeddings_list ,
    documents =documents ,
    metadatas =metadatas 
    )

    total_count =collection .count ()
    return total_count 

def query_chunks (
query_embedding :Any ,
top_k :int =3 ,
source :Optional [str ]=None ,
persist_dir :str |Path =DEFAULT_DB_PATH 
)->Dict [str ,Any ]:

    collection =get_collection (persist_dir )

    if isinstance (query_embedding ,np .ndarray ):

        if query_embedding .ndim ==2 and query_embedding .shape [0 ]==1 :
            query_list =query_embedding [0 ].tolist ()
        else :
            query_list =query_embedding .tolist ()
    else :
        query_list =list (query_embedding )

    if query_list and not isinstance (query_list [0 ],list ):
        query_embeddings =[query_list ]
    else :
        query_embeddings =query_list 

    where_filter ={"source":source }if source else None 

    available_docs =collection .count ()
    actual_k =min (top_k ,available_docs )if available_docs >0 else top_k 

    results =collection .query (
    query_embeddings =query_embeddings ,
    n_results =actual_k ,
    where =where_filter ,
    include =["documents","metadatas","distances"]
    )

    return results 

def count_documents (persist_dir :str |Path =DEFAULT_DB_PATH )->int :

    collection =get_collection (persist_dir )
    return collection .count ()

def reset_collection (persist_dir :str |Path =DEFAULT_DB_PATH )->None :

    client =get_client (persist_dir )
    try :
        client .delete_collection (name =COLLECTION_NAME )
    except Exception :
        pass
