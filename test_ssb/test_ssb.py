import requests
from datetime import datetime
import pandas as pd

# url = "https://data.ssb.no/api/v0/no/table/10748/"

# query_fast = """{
#   "query": [
#     {
#       "code": "Utlanstype",
#       "selection": {
#         "filter": "item",
#         "values": [
#           "04"
#         ]
#       }
#     },
#     {
#       "code": "Sektor",
#       "selection": {
#         "filter": "item",
#         "values": [
#           "04b"
#         ]
#       }
#     },
#     {
#       "code": "Rentebinding",
#       "selection": {
#         "filter": "item",
#         "values": [
#           "06"
#         ]
#       }
#     }
#   ],
#   "response": {
#     "format": "json-stat2"
#   }
# }"""

# query_flytende = """{
#   "query": [
#     {
#       "code": "Utlanstype",
#       "selection": {
#         "filter": "item",
#         "values": [
#           "04"
#         ]
#       }
#     },
#     {
#       "code": "Sektor",
#       "selection": {
#         "filter": "item",
#         "values": [
#           "04b"
#         ]
#       }
#     },
#     {
#       "code": "Rentebinding",
#       "selection": {
#         "filter": "item",
#         "values": [
#           "08"
#         ]
#       }
#     }
#   ],
#   "response": {
#     "format": "json-stat2"
#   }
# }"""


url_index = "https://data.ssb.no/api/v0/no/table/07221/"
query_index = """{
  "query": [
    {
      "code": "Region",
      "selection": {
        "filter": "item",
        "values": [
          "TOTAL"
        ]
      }
    },
    {
      "code": "Boligtype",
      "selection": {
        "filter": "item",
        "values": [
          "00"
        ]
      }
    },
    {
      "code": "ContentsCode",
      "selection": {
        "filter": "item",
        "values": [
          "Boligindeks"
        ]
      }
    }
  ],
  "response": {
    "format": "json-stat2"
  }
}"""



# def request2list(r) :
#     data = r.json()

#     rente = data["value"]
#     months_d = data["dimension"]["Tid"]["category"]["index"]
#     dates = []
#     for date_str, value in months_d.items():
#         dates.append( datetime.strptime(date_str, "%YM%m") )
#     return dates,rente




r = requests.post(url_index, data=query_index)

data = r.json()


rente = data["value"]