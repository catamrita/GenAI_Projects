"""
This program is build with Flan-T5-XL LLM to be able to answer the final question using the output from the previous questions as in-context learning/few-shot learning. 

Consider three related questions from a search session: Question 1, Question 2, Question 3
1. Answer to Question 1 needs to be generated. 
2. Answer to Question 2 needs to be generated with the answer to Question 1 as one-shot example / context. 
3. Answer to Question 3 needs to be generated with the answer to Question 2 as one-shot example / context.
4. Answer to Question 3 will be either YES or NO and nothing else.


> The program accepts three parameters provided as a command line input. 
> The three inputs represent the questions.
> The output of the first two question is Generation based whereas the last question output is deterministic i.e. its either YES or NO.
> Output should be in upper-case: YES or NO
> There should be no additional output including any warning messages in the terminal.
> Remember that your output will be tested against test cases, therefore any deviation from the test cases will be considered incorrect during evaluation.


Syntax: python template.py <string> <string> <string> 

The following example is given for your reference:

 Terminal Input: python template.py "Who is Rabindranath Tagore?" "Where was he born?" "Is it in America?"
Terminal Output: NO

 Terminal Input: python template.py "Who is Rabindranath Tagore?" "Where was he born?" "Is it in India?"
Terminal Output: YES

You are expected to create some examples of your own to test the correctness of your approach.

ALL THE BEST!!
"""

"""
ALERT: * * * No changes are allowed to import statements  * * *
"""
import sys
import torch
import transformers
from transformers import T5Tokenizer, T5ForConditionalGeneration
import re

##### You may comment this section to see verbose -- but you must un-comment this before final submission. ######
transformers.logging.set_verbosity_error()
transformers.utils.logging.disable_progress_bar()
#################################################################################################################

"""
* * * Changes allowed from here  * * * 
"""
#PLEASE SET GOOGLE COLAB RUNTIME to T4 GPU 
# THE FIRST EXECUTION TAKES LONGER as THE MODEL IS BEING DOWNLOADED FRESH FROM HF. 
# ON  SUBSEQUENT COLAB RUNS IT IS GETTING COMPLETED IN 1 MINUTES 
#Supress HuggingFace Hub warning
import os
os.environ["HF_HUB_VERBOSITY"] = 'error'
if 'huggingface_hub' in sys.modules:
  sys.modules['huggingface_hub'].logging.set_verbosity_error()


def llm_function(model,tokenizer,questions):
    '''
    The steps are given for your reference:

    1. Generate answer for the first question.
    2. Generate answer for the second question use the answer for first question as context.
    3. Generate a deterministic output either 'YES' or 'NO' for the third question using the context from second question.  
    5. Clean output and return.
    6. Output is case-sensative: YES or NO
    Note: The model (Flan-T5-XL) and tokenizer is already initialized. Do not modify that section.
    '''
    q1, q2, q3 = questions 

    #eval() + inference mode : faster inference , no grad tracking 
    model.eval()
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model.to(device)
        
    _MAX_LEN = 64
  

    def generate_answer(prompt, max_new_tokens=5):
        enc = tokenizer(prompt, return_tensors="pt",
                     truncation=True, max_length=_MAX_LEN).to(device)
        with torch.inference_mode():
          
            ids = model.generate(**enc, max_new_tokens=max_new_tokens,
                                 num_beams=1, do_sample=False,
                                 use_cache=True)          
        return tokenizer.decode(ids[0], skip_special_tokens=True).strip()

    a1 = generate_answer(f"Question: {q1}\nAnswer:")

    a2= generate_answer(
        f"Question: {q1}\nAnswer: {a1}\n\nQuestion: {q2}\nAnswer:"
        )

    prompt_q3 = f"Question: {q2}\nAnswer: {a2}\n\nQuestion: {q3}\nAnswer:"
    enc_q3 = tokenizer(prompt_q3, return_tensors="pt",
                          truncation=True, max_length=_MAX_LEN).to(device)
     
    
#collect token IDs for YES/NO
    yes_ids = list({tokenizer.encode(v,add_special_tokens=False)[0]
                    for v in ['YES','yes', 'Yes']})
    no_ids = list({tokenizer.encode(v,add_special_tokens=False)[0]
                    for v in ['NO','no', 'No']})
    
    dec_start = torch.tensor(
        [[model.config.decoder_start_token_id]], device=device
    )
    with torch.inference_mode():
      
      out = model(**enc_q3,decoder_input_ids=dec_start)
     
    logits = out.logits[0,0]
    yes_score = max(logits[i].item() for i in yes_ids)
    no_score  = max(logits[i].item() for i in no_ids)

    final_output = 'YES' if yes_score >= no_score else 'NO'
    return final_output
"""
ALERT: * * * No changes are allowed below this comment  * * *
"""

if __name__ == '__main__':

    question_a = sys.argv[1].strip()
    question_b = sys.argv[2].strip()
    question_c = sys.argv[3].strip()

    questions = [question_a, question_b, question_c]
    ##################### Loading Model and Tokenizer ########################
    tokenizer = T5Tokenizer.from_pretrained("google/flan-t5-xl")
    model = T5ForConditionalGeneration.from_pretrained("google/flan-t5-xl")
    ##########################################################################

    """  Call to function that will perform the computation. """
    torch.manual_seed(42)
    out = llm_function(model,tokenizer,questions)
    print(out.strip())

    """ End to call """