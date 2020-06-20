
import json

version_json = '''
{
 "last_update": "2020-02-02",
 "dirty": false,
 "error": null,
 "contact": "apalarcon1991@gmail.com",
 "version": "1.0.3",
 "author":"Albenis Pérez Alarcón"
}
'''  # END VERSION_JSON


def get_versions():
    return json.loads(version_json)
