import os 
from typing import List ,Dict ,Any ,Union 
from dotenv import load_dotenv 
from google import genai 
from google .genai .errors import APIError 

load_dotenv ()

DEFAULT_MODEL ="gemini-3.8-flash"

def get_gemini_client ()->genai .Client :

    api_key =os .getenv ("GEMINI_API_KEY")or os .getenv ("GOOGLE_API_KEY")
    if not api_key :
        raise ValueError (
        "Gemini API key not found. Please set GEMINI_API_KEY in your .env file."
        )
    return genai .Client (api_key =api_key )

def format_context (context :Union [str ,List [Dict [str ,Any ]]])->str :

    if not context :
        return ""

    if isinstance (context ,str ):
        return context .strip ()

    formatted_parts =[]
    for chunk in context :
        if isinstance (chunk ,dict ):
            content =chunk .get ("content","").strip ()
            source =chunk .get ("source","Document")
            page =chunk .get ("page_number","?")
            formatted_parts .append (f"[Source: {source}, Page: {page}]\n{content}")
        else :
            formatted_parts .append (str (chunk ).strip ())

    return "\n\n".join (formatted_parts ).strip ()

def generate_answer (
question :str ,
context :Union [str ,List [Dict [str ,Any ]]],
model_name :str =DEFAULT_MODEL 
)->str :

    if not question or not question .strip ():
        return "Please provide a valid question."

    context_str =format_context (context )
    if not context_str :
        return "The provided documents do not contain sufficient information to answer this question."

    prompt =f

    try :
        client =get_gemini_client ()
        response =client .models .generate_content (
        model =model_name ,
        contents =prompt 
        )

        if response and response .text :
            return response .text .strip ()
        return "No answer could be generated from the model."

    except ValueError as ve :

        return f"Configuration Error: {str(ve)}"
    except APIError as ae :

        return f"Gemini API Error: {str(ae)}"
    except Exception as e :
        return f"Unexpected Error: {str(e)}"
