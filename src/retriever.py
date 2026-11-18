from typing import List ,Dict ,Any ,Optional 
from src .embeddings import embed_query 
from src .vector_store import query_chunks ,count_documents 

def retrieve (
query :str ,
top_k :int =3 ,
source :Optional [str ]=None 
)->List [Dict [str ,Any ]]:

    if not query or not query .strip ():
        return []

    total_docs =count_documents ()
    if total_docs ==0 :
        return []

    query_embedding =embed_query (query .strip ())

    results =query_chunks (
    query_embedding =query_embedding ,
    top_k =top_k ,
    source =source 
    )

    retrieved_chunks :List [Dict [str ,Any ]]=[]

    documents_list =results .get ("documents",[[]])[0 ]
    metadatas_list =results .get ("metadatas",[[]])[0 ]
    distances_list =results .get ("distances",[[]])[0 ]

    for i in range (len (documents_list )):
        doc_text =documents_list [i ]
        meta =metadatas_list [i ]if i <len (metadatas_list )else {}
        distance =distances_list [i ]if i <len (distances_list )else 0.0 

        similarity_score =max (0.0 ,1.0 -float (distance ))

        retrieved_chunks .append ({
        "content":doc_text ,
        "source":meta .get ("source","unknown"),
        "page_number":meta .get ("page_number",1 ),
        "chunk_id":meta .get ("chunk_id",i +1 ),
        "similarity_score":round (similarity_score ,4 )
        })

    return retrieved_chunks
