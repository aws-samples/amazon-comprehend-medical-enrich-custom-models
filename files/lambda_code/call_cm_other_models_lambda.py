import json
import boto3
import logging
import os
import copy
import swalign

#get environment variables
#name of bucket lambda gets notifications from
NOTIFICATION_BUCKET_NAME = os.environ['NOTIFICATION_BUCKET_NAME']
OUTPUT_BUCKET_NAME=os.environ['OUTPUT_BUCKET_NAME']
COMPREHEND_CUSTOM_ENDPOINT=os.environ['COMPREHEND_CUSTOM_ENDPOINT']

#hard coded list of breast cancer genes
list_of_genes=["ATM", "BARD1", "BRCA1", "BRCA2", "BRIP1", "CDH1", "CHEK2", "NBN", "NF1", "PALB2", "PTEN", "RAD51C", "RAD51D", "STK11", "TP53", "ATM serine/threonine kinase", "BRCA1 associated RING domain 1", "BRCA1 DNA repair associated", "BRCA2 DNA repair associated", "BRCA1 interacting protein C-terminal helicase 1", "cadherin 1", "checkpoint kinase 2", "nibrin", "neurofibromin 1", "partner and localizer of BRCA2", "phosphatase and tensin homolog", "RAD51 paralog C", "RAD51 paralog D", "serine/threonine kinase 11", "tumor protein p53"]


def get_match_score(string_1=None,string_2=None,normalize_max=True):
    '''
    Get the score of the match between 2 strings. This uses the Smith-Waterman (SW) algorithm.
    SW computes local alighments between the two strings, and returns the match percentage between the two.
    If normalize_max=True, the score is normalized based on the maximum string length. If not, the score is normalized that by length of string_1.
    Note: The normalized SW score may be not symmetric.
    '''
    match = 1
    mismatch = -1
    #set lower gap penalty to encourage longer matches.
    gap_penalty = -.5
    scoring = swalign.NucleotideScoringMatrix(match, mismatch)
    sw = swalign.LocalAlignment(scoring,gap_penalty)
    z=sw.align(string_1,string_2)
    total_score=z.score

    len_to_divide=len(string_1)

    if normalize_max==True:
        len_to_divide=max(len(string_1),len(string_2))
    score=total_score/len_to_divide
    return(score)


def read_in_file_from_s3(bucketname,filename):
    '''reads in the file from S3 and returns the content from the body of the file'''
    s3 = boto3.resource('s3')
    obj = s3.Object(bucketname, filename)
    body = obj.get()['Body'].read()
    return(body)


def put_file_in_s3(filename,filecontent,output_bucketname):
    '''add file to s3 bucket, return response of operation'''
    s3 = boto3.client('s3')
    response = s3.put_object(
        Bucket=output_bucketname,
        Key=filename,
        Body=filecontent,
    )
    return (response)


def call_comprehehend_medical(the_input=None,call_type="detect_entities_v2"):
    '''pass the input data to comprehend medical
    call_type controls what NLP operation comprehend medical should do.
    call_type must be a valid method for CM.
    '''
    structured_content=None
    client = boto3.client('comprehendmedical')
    if call_type=='detect_entities_v2':
        structured_content = client.detect_entities_v2(Text=the_input)
    elif call_type=='infer_icd10_cm':
        structured_content = client.infer_icd10_cm(Text=the_input)
    elif call_type=='infer_rx_norm':
        structured_content = client.infer_rx_norm(Text=the_input)
    else:
        logging.warning(f'Something is Wrong. Comprehend Medical call type {call_type} may be invalid.')
    try:
        response=structured_content['ResponseMetadata']['HTTPStatusCode']
        if response==200:
            pass
    except:
        logging.warning('Something is wrong. Perhaps there is a problem calling Comprehend Medical?')
        structured_content=None

    return(structured_content)


def call_custom_comprehend_model(the_input=None,endpoint_arn=None):
    '''call the custom comprehend model that has been previously trained to classify the document according to its medical specialty type.'''
    client = boto3.client('comprehend')
    response = client.classify_document(
        Text=the_input,
        EndpointArn=endpoint_arn
        )
    return(response)


def search_dict_for_breast_cancer_genes(the_dict=None):
    '''search the comprehend medical output for breast cancer genes and augment the comprehend medical output. This modifies the input dictionary in place'''
    list_to_search=the_dict['Entities'] #only search the entities found by comprehend medical.
    for i in range(0,len(list_to_search)):
        try:
            dict_to_examine=list_to_search[i]
            the_text=dict_to_examine['Text']
            max_score=0
            #get score of best matching gene
            for j in range(0,len(list_of_genes)):
                the_score=get_match_score(string_1=list_of_genes[j],string_2=the_text,normalize_max=False)
                if the_score >= max_score:
                    max_score=the_score
            gene_score=max_score
            #hard code a score threshold. Do not report scores less than this.
            if gene_score >=.75:
                dict_to_append={"Name":"BREAST_CANCER_GENE",'Score':gene_score}
                dict_to_examine['Traits'].append(dict_to_append)
        except: #if the dictionary entry doesn't match, just skip it.
            pass
    return(the_dict)


def search_raw_text_for_breast_cancer_genes(raw_text=None):
    ''''search the raw output for breast cancer genes. This searches the entire text; not just Comprehend Medical identified entities.'''
    list_to_search=raw_text.split()
    dict_of_breast_cancer_genes={}
    dict_of_breast_cancer_genes['BREAST_CANCER_GENES_FOUND']=[]
    for i in range(0,len(list_to_search)):
        try:
            the_text=list_to_search[i]
            max_score=0
            #get score of best matching gene
            for j in range(0,len(list_of_genes)):
                the_score=get_match_score(string_1=list_of_genes[j],string_2=the_text,normalize_max=True)
                if the_score >= max_score:
                    max_score=the_score
                    the_index=j
            gene_score=max_score
            if gene_score >=.75:   #hard code a score threshold. Do not report scores less than this.
                dict_of_breast_cancer_genes['BREAST_CANCER_GENES_FOUND'].append({the_text:list_of_genes[the_index]})
        except: #if the dictionary entry doesn't match, just skip it.
            pass
    return(dict_of_breast_cancer_genes)


def lambda_handler(event, context):
    #uncomment to log event info
    #logging.info(json.dumps(event))

    filename=event['Records'][0]['s3']['object']['key']
    filename_basename=os.path.basename(filename)

    content=read_in_file_from_s3(NOTIFICATION_BUCKET_NAME,filename)
    content_2=call_comprehehend_medical(the_input=content.decode("utf-8") ,call_type='detect_entities_v2') #decode to prevent error
    custom_predictions=call_custom_comprehend_model(the_input=content.decode("utf-8"),endpoint_arn=COMPREHEND_CUSTOM_ENDPOINT)
    content_3=copy.deepcopy(content_2) #make copy to avoid modifying original dictionary
    content_3['Medical_Specialty_Prediction']=custom_predictions
    content_4=search_dict_for_breast_cancer_genes(the_dict=content_3)
    breast_cancer_genes_found=search_raw_text_for_breast_cancer_genes(raw_text=content.decode("utf-8"))
    content_4['BREAST_CANCER_GENE_PREDICTIONS']=breast_cancer_genes_found

    #export final output
    #logging.info(json.dumps(content_4))
    put_file_in_s3(f'''{filename_basename}_out''',json.dumps(content_4),OUTPUT_BUCKET_NAME)


    return {
        'statusCode': 200,
        'body': json.dumps('Hello from Lambda!')
    }
