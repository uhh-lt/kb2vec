from diffbot_api import query_and_save, ENTITY_TYPES

query_and_save('allUris:"barackobama.com"', "data/all-uris.json")
query_and_save('wikipediaUri:"en.wikipedia.org/wiki/Barack_Obama"', "data/wiki-uri.json")
query_and_save('allUris:"en.wikipedia.org/wiki/Barack\_Obama"', "data/all-uris-wiki.json")
query_and_save('origins:"en.wikipedia.org/wiki/Barack_Obama"', "data/origins.json")

for entity_type in ENTITY_TYPES:
    query_and_save(
        query='type:{}'.format(entity_type),
        output_fpath="data/{}.json".format(entity_type))
    
query_and_save(
    query='type:Person name:"Alexander Panchenko"',
    output_fpath="data/ap.json")


query_and_save(
    query='type:Person employments.employer.name:"Diffbot"',
    output_fpath="data/diffbot-employees.json")


query_and_save(
    query='type:Person employments.{title:"CEO" employer.name:"Diffbot"}',
    output_fpath="data/diffbot-ceo.json")

query_and_save(
    query='type:Person employments.{employer.name:"Diffbot" isCurrent:true}',
    output_fpath="data/diffbot-current-employees.json")

query_and_save(
    query='type:Person name:"Angela Merkel"',
    output_fpath="data/am.json")

query_and_save(
    query='type:Person name:"Barack Obama"',
    output_fpath="data/bo.json")

query_and_save(
    query='type:Person name:"Nicolas Sarkozy"',
    output_fpath="data/ns.json")

query_and_save(
    query='type:Person name:"Diego Maradona"',
    output_fpath="data/dm.json")
