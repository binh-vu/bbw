#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Library for semantic annotation of tabular data with the Wikidata knowledge graph"""
import functools
from pathlib import Path
from typing import List, Dict, Mapping, Tuple
from traceback import print_exc
import pandas as pd
import requests
from collections import Counter
import re
import difflib
from datetime import date
from bs4 import BeautifulSoup
import time
import ftfy
import numpy as np
import random
import string
import os
import langid
from jdcal import jd2gcal, jcal2jd
from loguru import logger

from sm.misc import serialize_json, deserialize_json
from kgdata.wikidata.models import DataValue, QNode, WDProperty
from bbw_baseline.file_cache import EagerSingleFileCache, key_jsondump, key_unordered_list
from sm.namespaces.wikidata import WikidataNamespace


url_query = "https://query.wikidata.org/sparql" # default URL for SPARQL endpoint
url_front = "http://www.wikidata.org" # default URL for Wikibase frontend
ptype = "P31" # default property for 'instance of'


def get_parallel(a, n):
    """Get input for GNU parallel based on a-list of filenames and n-chunks.
    The a-list is split into n-chunks. Offset and amount are provided."""
    k, m = divmod(len(a), n)
    # chunked = list((a[i * k + min(i, m):(i + 1) * k + min(i + 1, m)] for i in range(n)))
    offset = ' '.join(list((str(i * k + min(i, m)) for i in range(n))))
    amount = ' '.join(list((str(k + min(i + 1, m) - min(i, m)) for i in range(n))))
    parallel = "parallel --delay 1 --linebuffer --link python3 bbw_cli.py "
    input_4_gnu_parallel = parallel + "--amount ::: " + amount + " ::: --offset  ::: " + offset
    return input_4_gnu_parallel


def random_user_agent(agent='bot4bbw-'):
    """Add random strings from the left and right sides to an input user agent."""
    letters = string.ascii_lowercase
    random_agent = '-' + ''.join(random.choice(letters) for i in range(random.randrange(4, 9)))
    return random.choice(letters) + agent + random_agent


def get_language(string):
    """ https://github.com/IBCNServices/CSV2KG/blob/master/csv2kg/util.py#L15-L19 """
    return 'en'
    try:
        return langid.classify(string)[0]
    except Exception:
        return 'en'


def get_datatype(prop, url=url_query):
    """
    Parameters
    ----------
    prop : str
        URL of a property in Wikidata or its PID.
    url : str, optional
        URL of a SPARQL endpoint. The default is "https://query.wikidata.org/sparql".
    Returns
    -------
    output : str
        Datatype corresponding to the property in Wikidata.
        See https://www.mediawiki.org/wiki/Wikibase/DataModel#Datatypes.
    """
    try:
        prop = prop.split('/')[-1]
        query = """SELECT ?datatype WHERE {
            ?x wikibase:directClaim wdt:""" + prop + """;
            wikibase:propertyType ?datatype.}"""
        r = requests.get(url,
                         params={'format': 'json', 'query': query},
                         headers={'User-Agent': random_user_agent()},
                         timeout=2)
        results = r.json().get('results').get('bindings')
        datatype = results[0].get('datatype').get('value')
        if datatype:
            output = datatype
        else:
            output = ''
    except Exception:
        output = ''
    return output


def get_SPARQL_dataframe_item(name, language, 
                              url=url_query, ptype=ptype):
    """
    Parameters
    ----------
    name : str
        Possible item in wikidata.
    url : str, optional
        SPARQL-endpoint. The default is "https://query.wikidata.org/sparql".
    Returns
    -------
    output : pd.DataFrame
        Dataframe created from the json-file returned by SPARQL-endpoint.
    """
    name = name.replace('"', '\\\"')
    if language:
        lang = language
    else:
        lang = get_language(name)
    query = """SELECT REDUCED ?value ?valueType ?p2 ?item ?itemType ?itemLabel WHERE {
                ?value rdfs:label """ + '"' + name + '"@' + lang + """;
                wdt:""" + ptype + """ ?valueType.
                ?item ?p2 [ ?x """ + '"' + name + '"@' + lang + """].
                ?item wdt:""" + ptype + """ ?itemType.
                ?item rdfs:label ?itemLabel.
                FILTER((LANG(?itemLabel)) = "en").
            }
            LIMIT 10000
            """
    try:
        r = requests.get(url,
                         params={'format': 'json', 'query': query},
                         headers={'User-Agent': random_user_agent()},
                         timeout=2.5)
        if r.status_code == 429:
            time.sleep(int(r.headers["Retry-After"]))
            r = requests.get(url,
                             params={'format': 'json', 'query': query},
                             headers={'User-Agent': random_user_agent()},
                             timeout=2.5)
        results = r.json().get('results').get('bindings')
        for prop in results:
            prop.update((key, value.get('value')) for key, value in prop.items())
        if len(results) > 0:
            output = pd.DataFrame(results, dtype=str)
        else:
            output = None
    except Exception:
        output = None

    return output


def get_SPARQL_dataframe_prop(prop, value, url=url_query, ptype=ptype):
    value = [val.replace('"', '\\\"') for val in value]
    subquery = []
    subquery.extend([""" wdt:""" + str(prop) + """ [ ?p """ + '"' + str(value) + '"' + """@en ] ;
        wdt:""" + str(prop) + " ?value" + str(ind) + ";" for ind, (prop, value) in enumerate(zip(prop, value))])
    subquery = ' '.join(subquery)
    # wdt:"""+ str(prop) + """ [ ?p """ + '"' + str(value) + '"' + """@en ] ;
    #    wdt:"""+ str(prop) + """ ?value0;
    query = """
    SELECT REDUCED ?item ?itemType ?itemLabel ?p2 ?value ?valueType ?valueLabel ?psvalueLabel WHERE {
  ?item """ + subquery + """
        ?p2 ?value.
  ?item wdt:""" + ptype + """ ?itemType;
        rdfs:label ?itemLabel.
  FILTER (lang(?itemLabel) = "en").
  OPTIONAL {
  ?value wdt:""" + ptype + """ ?valueType .}
  OPTIONAL {?wdproperty wikibase:claim ?p2 ;
                        wikibase:statementProperty ?psproperty .
            ?value ?psproperty ?psvalue .}
   SERVICE wikibase:label { bd:serviceParam wikibase:language "en". }
   }
    LIMIT 50000
    """
    try:
        r = requests.get(url,
                         params={'format': 'json', 'query': query},
                         headers={'User-Agent': random_user_agent()},
                         timeout=5)  # To avoid 1 min. timeouts.
        if r.status_code == 429:
            time.sleep(int(r.headers["Retry-After"]))
            r = requests.get(url,
                             params={'format': 'json', 'query': query},
                             headers={'User-Agent': random_user_agent()},
                             timeout=5)
        results = r.json().get('results').get('bindings')
        for prop in results:
            if 'psvalueLabel' in prop and prop.get('psvalueLabel').get('value') is not None:
                prop['valueLabel']['value'] = prop.get('psvalueLabel').get('value')
            prop.update((key, value.get('value')) for key, value in prop.items())
        if len(results) > 0:
            output = pd.DataFrame(results, dtype=str)
        else:
            output = None
    except Exception:
        output = None

    return output


def get_openrefine_bestname(name):
    """
    Parameters
    ----------
    name : str
        Possible entity label in wikidata.
    Returns
    -------
    bestname : str
        The best suggestion returned by OpenRefine-Reconciliation API-service.
    """
    # Alternative url: "https://openrefine-wikidata.toolforge.org/en/api"
    url = "https://wikidata.reconci.link/en/api"
    params = {"query": name}

    try:
        r = requests.get(url=url, params=params, headers={'User-Agent': random_user_agent()}, timeout=1)
        results = r.json().get('result')
        bestname = results[0].get('name')
    except Exception:
        bestname = None
    return bestname


def get_wikidata_URL(name, url=url_front):
    """
    Parameters
    ----------
    name : str
        Possible entity label in wikidata.
    Returns
    -------
    bestname : str
        The best suggestion returned by Wikidata API-service.
    """

    url = url_front + "/w/api.php"
    params = {"action": "query",
              "srlimit": "1",
              "format": "json",
              "list": "search",
              "srqiprofile": "wsum_inclinks_pv",
              "srsearch": name}

    try:
        r = requests.get(url=url, params=params, headers={'User-Agent': random_user_agent()}, timeout=1)
        results = r.json()
        if len(results) != 0:
            query = results.get('query')
            if query:
                search = query.get('search')
                if len(search) > 0:
                    bestname = search[0].get("title")
                    if bestname:
                        URL = url_front + '/entity/' + bestname
        if not URL:
            URL = None
    except Exception:
        URL = None
    return URL


def get_wikidata_title(url, url_front=url_front):
    """
    Parameters
    ----------
    url : str
        URL of a Wikidata page.
    Returns
    -------
    title: str
        Title of a Wikidata page.
    """
    try:
        url = url.replace(url_front+'/prop/direct/', url_front+'/entity/')
        params = {"action": "wbgetentities",
                  "format": "json",
                  "props": "labels",
                  "ids": url.split('/')[-1]}
        r = requests.get(url, params=params, headers={'User-Agent': random_user_agent()}, timeout=5).json()
        title = r.get('entities').get(url.split('/')[-1]).get('labels').get('en').get('value')
    except Exception:
        title = ''
    return title


def get_title(url):
    """
    Parameters
    ----------
    url : str
        URL of a web-page.
    Returns
    -------
    title: str
        Title of a web-page.
    """
    try:
        r = requests.get(url, headers={'User-Agent': random_user_agent()}, timeout=1)
        title = BeautifulSoup(r.text, features="lxml").title.text
        title = title.replace(' - Wikidata', '')
    except Exception:
        title = None
    return title


def get_wikimedia2wikidata_title(wikimedia_url):
    """
    Parameters
    ----------
    wikimedia_url : str
        URL of a Wikimedia Commons page.
    Returns
    -------
    title : str
        The title of the corresponding Wikidata page.
    """
    try:
        r = requests.get(wikimedia_url, headers={'User-Agent': random_user_agent()}, timeout=1)
        soup = BeautifulSoup(r.content, 'html.parser')
        redirect_url = soup.find(class_="category-redirect-header")
        if redirect_url:
            redirect_url = redirect_url.find("a").get("href")
            r = requests.get("https://commons.wikimedia.org" + redirect_url,
                             headers={'User-Agent': random_user_agent()}, timeout=1)
            soup = BeautifulSoup(r.content, 'html.parser')
        wikidata_url = soup.find('a', title="Edit infobox data on Wikidata").get('href')
        # time.sleep(0.25)
        title = get_title(wikidata_url)
    except Exception:
        title = None
    return title


def get_wikipedia2wikidata_title(wikipedia_title, url_front=url_front):
    """
    Parameters
    ----------
    wikipedia_title : str
        Title of a Wikipedia article.
    Returns
    -------
    bestname : str
        The title of the corresponding entity in Wikidata.
    """
    url = url_front + "/w/api.php"
    params = {"action": "query",
              "prop": "pageprops",
              "ppprop": "wikibase_item",
              "redirects": "1",
              "titles": wikipedia_title,
              "format": "json"}

    try:
        r = requests.get(url=url, params=params, headers={'User-Agent': random_user_agent()}, timeout=1)
        pages = r.json().get('query').get('pages')
        if pages.get('-1'):
            bestname = None
        else:
            # bestname = [k for k in pages.values()][0].get('title')
            wikidataID = [k for k in pages.values()][0].get('pageprops').get('wikibase_item')
            bestname = get_title(url_front + "/wiki/" + wikidataID).replace(' - Wikidata', '')
    except Exception:
        bestname = None
    return bestname


def get_searx_bestname(name):
    """
    Parameters
    ----------
    name : str
        Possible entity label in wikidata.
    Returns
    -------
    bestname : str
        A few best suggestions returned by the Searx metasearch engine.
    """
    name_cleaned = name.replace('!', ' ').replace('#', ' ').replace(':-', ' -')
    url = os.getenv("BBW_SEARX_URL", "http://localhost:80")
    engines = "!yh !ddd !eto !bi !ew !et !wb !wq !ws !wt !wv !wy !tl !qw !mjk !nvr !wp !cc !wd !ddg !sp !yn !dc "
    data = {"q": engines + name_cleaned, "format": "json"}
    try:
        results = requests.get(url, data=data, headers={'User-Agent': random_user_agent()}).json()
        if 'results' not in locals():
            raise Exception
        bestname = []
        medianame = []
        # Process infoboxes
        if len(results.get('infoboxes')) > 0:
            bestname.extend([x.get('infobox') for x in results.get('infoboxes')])
        # Process suggestions
        if len(results.get('suggestions')) > 0:
            bestname.extend([k for k in results.get('suggestions') if not re.search("[\uac00-\ud7a3]", k)])
            for sugg in results.get('suggestions'):
                splitsugg = sugg.split()
                if len(splitsugg) > 2 and not re.search("[\uac00-\ud7a3]", sugg):
                    bestname.extend([' '.join(splitsugg[:-1])])
            best_sugg = difflib.get_close_matches(name, [k for k in results.get('suggestions') if
                                                         not re.search("[\uac00-\ud7a3]", k)], n=1, cutoff=0.65)
            try:
                data2 = {"q": engines + best_sugg[0], "format": "json"}
                results2 = requests.get(url, data=data2, headers={'User-Agent': random_user_agent()}).json()
                if results2:
                    if len(results2.get('infoboxes')) > 0:
                        bestname.extend([x.get('infobox') for x in results2.get('infoboxes')])
            except Exception:
                pass
        # Process corrections
        if len(results.get('corrections')) > 0:
            corrections = [corr for corr in results.get('corrections') if '"' not in corr]
            if len(corrections) > 0:
                for correction in corrections:
                    try:
                        data3 = {"q": engines + correction, "format": "json"}
                        results3 = requests.get(url, data=data3, headers={'User-Agent': random_user_agent()}).json()
                        if results3:
                            if len(results3.get('infoboxes')) > 0:
                                bestname.extend([x.get('infobox') for x in results3.get('infoboxes')])
                    except Exception:
                        pass
                bestname.extend(corrections)
        # Process search results
        if len(results.get('results')) > 0:
            for i, result in enumerate(results.get('results')):
                url = result.get('url')
                parsed_url=result.get('parsed_url')
                if len(parsed_url) > 1:
                    hostname = parsed_url[1]
                raw_title = result.get('title')
                if i == 1:
                    bestname.append(
                        raw_title.split(' | ')[0].split(" - ")[0].split(" ? ")[0].split(" ? ")[0].split(' \u2014 ')[
                            0].split(' \u2013 ')[0].replace('Talk:', '').replace('Category:', '').replace('...', ''))
                if ("wiki" in url) and not raw_title.endswith('...'):
                    bestname.append(
                        raw_title.split(" - ")[0].split(" ? ")[0].split(" ? ")[0].split(' \u2014 ')[0].split(
                            ' \u2013 ')[0].replace('Talk:', '').replace('Category:', ''))
                if ("wiki" in url) and raw_title.endswith('...') and ("Wikidata:SPARQL" not in result.get('title')):
                    title = get_title(url)
                    if title:
                        bestname.append(
                            title.split(" - ")[0].split(" ? ")[0].split(" ? ")[0].split(' \u2014 ')[0].split(
                                ' \u2013 ')[0].replace('Talk:', '').replace('Category:', ''))
                if hostname:
                    if hostname.endswith('.wikimedia.org'):
                        title = get_wikimedia2wikidata_title(url)
                        if title:
                            medianame.append(
                                title.split(" - ")[0].split(" ? ")[0].split(" ? ")[0].split(' \u2014 ')[0].split(
                                    ' \u2013 ')[0])
                if "dict" in url:
                    bestname.append(raw_title.split(' : ')[0].split(' | ')[0])
                raw_match = difflib.get_close_matches(name, [
                    raw_title.replace(' ...', '').replace(' ?', '').split(' | ')[0].split(" - ")[0].split(' \u2014 ')[
                        0].split(' \u2013 ')[0]], n=1, cutoff=0.7)
                if len(raw_match) == 1:
                    bestname.append(raw_match[0])
        if len(bestname) > 0:
            bestname = [best for best in bestname if best != name]
        suggestions = list(set(difflib.get_close_matches(name, bestname, n=3, cutoff=0.41)))
        suggestions = suggestions + [get_openrefine_bestname(best) for best in suggestions]
        suggestions = suggestions + [get_wikipedia2wikidata_title(best) for best in suggestions]
        suggestions = list(set([best for best in suggestions if best]))
        bestname = difflib.get_close_matches(name, suggestions, n=3, cutoff=0.7)
        if 'results' in locals():
            if len(results.get('infoboxes')) > 0:
                bestname.extend([x.get('infobox') for x in results.get('infoboxes')])
        if 'results2' in locals():
            if len(results2.get('infoboxes')) > 0:
                bestname.extend([x.get('infobox') for x in results2.get('infoboxes')])
        if 'results3' in locals():
            if len(results3.get('infoboxes')) > 0:
                bestname.extend([x.get('infobox') for x in results3.get('infoboxes')])
        if len(bestname) == 0:
            bestname = difflib.get_close_matches(name, suggestions, n=3, cutoff=0.41)
            if len(bestname) == 0:
                bestname = None
        if len(medianame) > 0:
            bestname.extend(medianame)
        if len(bestname) > 0:
            bestname = list(set([best for best in bestname if best != name]))
            if len(bestname) == 0:
                bestname = None
    except Exception:
        bestname = None
    return bestname


def isfloat(value):
    try:
        float(value.replace(',', ''))
        return True
    except ValueError:
        return False


def get_common_class(classes, url=url_query, url_front=url_front):
    """
    Parameters
    ----------
    classes : list
        List of Wikidata entities.
    url : str, optional
        SPARQL-endpoint. The default is "https://query.wikidata.org/sparql".
    Returns
    -------
    output : str
        The common Wikidata class for a list of Wikidata entities.
    """
    if not isinstance(classes, list):
        print("Error:", classes, "is not a list of classes. ")
        return
    classes = [entity.replace(url_front + '/entity/', '') for entity in classes]
    lengths = ['?len' + entity for entity in classes]
    length = '(' + ' + '.join(lengths) + ' as ?length)'
    subquery = []
    for entity, Qlength in zip(classes, lengths):
        subquery.append("""
    SERVICE gas:service {
        gas:program gas:gasClass "com.bigdata.rdf.graph.analytics.SSSP" ;
                    gas:in wd:""" + entity + """ ;
                    gas:traversalDirection "Forward" ;
                    gas:out ?super ;
                    gas:out1 """ + Qlength + """ ;
                    gas:maxIterations 10 ;
                    gas:linkType wdt:P279 .
      }
    """)
    subquery = ' '.join(subquery)
    query = """PREFIX gas: <http://www.bigdata.com/rdf/gas#>
    SELECT ?super """ + length + """ WHERE {""" + subquery + """
    } ORDER BY ?length
    LIMIT 1"""
    try:
        r = requests.get(url,
                         params={'format': 'json', 'query': query},
                         headers={'User-Agent': random_user_agent()})
        results = r.json().get('results').get('bindings')
        output = results[0].get('super').get('value')
    except Exception as e:
        output = classes[0]

    return output


def get_one_class(classes):
    """
    Takes a list of two tuples with a class and the number of times it has appeared in a column.
    Returns a common class.
    """
    if not classes:
        return None
    if len(classes) == 1 or (len(classes) == 2 and classes[0][1] > classes[1][1]):
        return classes[0][0]
    if len(classes) == 2 and classes[0][1] == classes[1][1]:
        one_class = BBWSearchFn.get_instance().get_common_class([classes[0][0], classes[1][0]])
        if one_class == "http://www.wikidata.org/entity/Q35120":
            return classes[0][0]
        else:
            return one_class
    if len(classes) > 2:
        print('ERROR: More than two classes in get_one_class().')
        return None


class BBWSearchFn:
    """Create the class so that we can override the function we need or restore it"""
    instance = None

    ENABLE_FLAG = "enable"
    THROW_ERROR_ON_INVOKE_FLAG = "throw_error_on_invoke"
    USE_OUR_IMPL_FLAG = "new_implementation"

    class SpecialException(Exception):
        pass

    def __init__(self):
        self.lookup = None
        self.get_SPARQL_dataframe = None
        self.get_searx_bestname = None
        self.get_openrefine_bestname = None
        self.get_SPARQL_dataframe_type = None

    @staticmethod
    def get_instance():
        if BBWSearchFn.instance is None:
            BBWSearchFn.instance = BBWSearchFn()
        return BBWSearchFn.instance

    def setup(self, 
                qnodes: Mapping[str, QNode], 
                table2links: Dict[str, Dict[Tuple[int, int], List[str]]],
                cache_dir, enable_override: bool,
                get_SPARQL_dataframe_type_flag: str,
                get_SPARQL_dataframe_type2_flag: str):
        self.qnodes = qnodes
        self.table2links = table2links
        self.cache_dir = Path(cache_dir).absolute()
        self.cache_dir.mkdir(exist_ok=True, parents=True)

        get_searx_bestname_outfile = str((self.cache_dir / "get_searx_bestname.pkl").absolute())
        get_openrefine_bestname_outfile = str((self.cache_dir / "get_openrefine_bestname.pkl").absolute())
        get_common_class_outfile = str((self.cache_dir / "get_common_class.pkl").absolute())

        get_SPARQL_dataframe_outfile = str((self.cache_dir / "get_SPARQL_dataframe.pkl").absolute())
        get_SPARQL_dataframe_type_outfile = str((self.cache_dir / "get_SPARQL_dataframe_type.pkl").absolute())

        self.get_searx_bestname = EagerSingleFileCache.cache_func(get_searx_bestname_outfile)(get_searx_bestname)
        self.get_openrefine_bestname = EagerSingleFileCache.cache_func(get_openrefine_bestname_outfile)(get_openrefine_bestname)

        self._original_get_common_class = EagerSingleFileCache.cache_func(get_common_class_outfile, key_fn=key_unordered_list)(get_common_class)
        self._original_get_SPARQL_dataframe = EagerSingleFileCache.cache_func(get_SPARQL_dataframe_outfile)(self._original_get_SPARQL_dataframe)
        self._original_get_SPARQL_dataframe_type = EagerSingleFileCache.cache_func(get_SPARQL_dataframe_type_outfile)(self._original_get_SPARQL_dataframe_type)
        
        if get_SPARQL_dataframe_type2_flag == self.ENABLE_FLAG:
            self.get_SPARQL_dataframe_type2 = self.big_cache_get_SPARQL_dataframe_type2(self.cache_dir / "class_instances")
        elif get_SPARQL_dataframe_type2_flag == self.THROW_ERROR_ON_INVOKE_FLAG:
            self.get_SPARQL_dataframe_type2 = self.big_cache_get_SPARQL_dataframe_type2(self.cache_dir / "class_instances", self.always_fail)
        else:
            self.get_SPARQL_dataframe_type2 = self.always_none

        if enable_override:
            self.lookup = self._lookup
            self.get_SPARQL_dataframe = self._get_SPARQL_dataframe
            self.get_common_class = self._get_common_class
            if get_SPARQL_dataframe_type_flag == self.ENABLE_FLAG:
                self.get_SPARQL_dataframe_type = self.wrap_cell_based_function(self._original_get_SPARQL_dataframe_type)
            elif get_SPARQL_dataframe_type_flag == self.THROW_ERROR_ON_INVOKE_FLAG:
                self.get_SPARQL_dataframe_type = self.always_fail
            else:
                self.get_SPARQL_dataframe_type = self._get_SPARQL_dataframe_type
        else:
            self.lookup = self.wrap_cell_based_function(self._original_lookup)
            self.get_common_class = self._original_get_common_class
            self.get_SPARQL_dataframe = self.wrap_cell_based_function(self._original_get_SPARQL_dataframe)
            self.get_SPARQL_dataframe_type = self.wrap_cell_based_function(self._original_get_SPARQL_dataframe_type)
        return self

    def always_fail(self, *kargs, **kwargs):
        logger.error("Calling a function you shouldn't call")
        raise self.SpecialException("ALWAYS FAIL")

    def always_none(self, *kargs, **kwargs):
        return None

    def _original_get_SPARQL_dataframe(self, name, language,
                                       url=url_query, extra='', ptype=ptype):
        """
        Parameters
        ----------
        name : str
            Possible mention in wikidata.
        url : str, optional
            SPARQL-endpoint. The default is "https://query.wikidata.org/sparql".
        extra : str
            An extra parameter that will be also SELECTed in the SPARQL query.
        Returns
        -------
        output : pd.DataFrame
            Dataframe created from the json-file returned by SPARQL-endpoint.
        """
        name = name.replace('"', '\\\"')
        if language:
            lang = language
        else:
            lang = get_language(name)
        if extra:
            subquery = """
            ?item rdfs:label ?itemLabel.
            FILTER (lang(?itemLabel) = """ + '"' + lang + '").'
        else:
            subquery = ""
        query = "SELECT DISTINCT ?item " + extra + """?itemType ?p1 ?p2 ?value ?valueType ?valueLabel ?psvalueLabel WHERE {
                    ?item ?p1 """ + '"' + name + '"' + "@" + lang + """;
                    ?p2 ?value.""" + subquery + """
                    OPTIONAL { ?item wdt:""" + ptype + """ ?itemType. }
                    OPTIONAL { ?value wdt:""" + ptype + """ ?valueType. }
                    OPTIONAL {
                        ?wdproperty wikibase:claim ?p2 ;
                            wikibase:statementProperty ?psproperty .
                        ?value ?psproperty ?psvalue .
                    }
                    SERVICE wikibase:label { bd:serviceParam wikibase:language """ + '"' + lang + '"' + """. }
                }
                LIMIT 100000
                """
        try:
            r = requests.get(url,
                             params={'format': 'json', 'query': query},
                             headers={'User-Agent': random_user_agent()},
                             timeout=12.5)
            if r.status_code == 429:
                time.sleep(int(r.headers["Retry-After"]))
                r = requests.get(url,
                                 params={'format': 'json', 'query': query},
                                 headers={'User-Agent': random_user_agent()},
                                 timeout=12.5)
            results = r.json().get('results').get("bindings")
            for prop in results:
                if 'psvalueLabel' in prop and prop.get('psvalueLabel').get('value') is not None:
                    prop['valueLabel']['value'] = prop.get('psvalueLabel').get('value')
                prop.update((key, value.get('value')) for key, value in prop.items())
            if len(results) != 0:
                output = pd.DataFrame(results, dtype=str)
            else:
                output = None
        except Exception:
            output = None

        return output

    def _original_get_SPARQL_dataframe_type(self, name, datatype, language, url=url_query, ptype=ptype):
        name = name.replace('"', '\\\"')
        if language:
            lang = language
        else:
            lang = get_language(name)
        query = """SELECT DISTINCT ?item ?itemLabel WHERE {
            {?item  (rdfs:label|skos:altLabel) """ + '"' + name + '"@' + lang + """.}
            ?item wdt:""" + ptype + """ wd:""" + datatype + """.
            SERVICE wikibase:label { bd:serviceParam wikibase:language """ + '"' + lang + '"' + """. }
            }
            LIMIT 10000"""
        try:
            r = requests.get(url,
                             params={'format': 'json', 'query': query},
                             headers={'User-Agent': random_user_agent()},
                             timeout=2)
            if r.status_code == 429:
                time.sleep(int(r.headers["Retry-After"]))
                r = requests.get(url,
                                 params={'format': 'json', 'query': query},
                                 headers={'User-Agent': random_user_agent()},
                                 timeout=2)
            results = r.json().get('results').get('bindings')
            for prop in results:
                prop.update((key, value.get('value')) for key, value in prop.items())
            if len(results) > 0:
                output = pd.DataFrame(results, dtype=str)
            else:
                output = None
        except Exception:
            output = None

        return output

    def _original_get_SPARQL_dataframe_type2(self, datatype, language, url=url_query, ptype=ptype):
        """This function query for all instances of a class???? I think we just need to cache it.
        as they used it in their code to find a proper name
        """
        if datatype == "Q5":
            limit = "LIMIT 350000"
        else:
            limit = "LIMIT 1000000"
        if language:
            lang = language
        else:
            lang = "en"
        query = """SELECT REDUCED ?itemLabel WHERE {
            hint:Query hint:maxParallel 50 .
            hint:Query hint:chunkSize 1000 .
            []  wdt:""" + ptype + """ wd:""" + datatype + """;
                  (rdfs:label|skos:altLabel) ?itemLabel.
            FILTER (lang(?itemLabel) = """ + '"' + lang + '"' + """).
            }
            """ + limit
        try:
            r = requests.get(url,
                             params={'format': 'json', 'query': query},
                             headers={'User-Agent': random_user_agent()},
                             timeout=59)
            if r.status_code == 429:
                time.sleep(int(r.headers["Retry-After"]))
                r = requests.get(url,
                                 params={'format': 'json', 'query': query},
                                 headers={'User-Agent': random_user_agent()},
                                 timeout=59)
            results = r.json().get('results').get('bindings')
            for prop in results:
                prop.update((key, value.get('value')) for key, value in prop.items())
            if len(results) > 0:
                # MODIFIED return list instead of dataframe as they call to list repeatedly????
                output = pd.DataFrame(results, dtype=str).itemLabel.to_list()
            else:
                output = None
        except Exception:
            output = None

        return output

    def _original_lookup(self, name_in_data, language, metalookup=True, openrefine=False):
        """
        Parameters
        ----------
        name_in_data : str
            Search string.
        Returns
        -------
        list = [WDdf, how_matched]
            WDdf is a dataframe with SPARQL-request
            how_matched shows how the string was matched
                0: SPARQL-Wikidata
                1: OpenRefine Suggest API
                2: Searx-metasearch
        """
        how_matched = ''
        proper_name = ''
        # Search entity using WD SPARQL-endpoint
        WDdf = self.get_SPARQL_dataframe(name_in_data, language)
        if isinstance(WDdf, pd.DataFrame):
            proper_name = name_in_data
            how_matched = 'SPARQL'  # This means we have found a mention of 'name_in_data' in Wikidata using single SPARQL-query
        if isinstance(WDdf, pd.DataFrame):
            if 'item' in WDdf.columns:
                if all(WDdf.item.str.contains('wikipedia')):
                    WDdf = None
        # Searx-metasearch-engine API
        if metalookup:
            if not isinstance(WDdf, pd.DataFrame):
                proper_name = self.get_searx_bestname(name_in_data)
                if proper_name:
                    test_list = []
                    for proper in proper_name:
                        e = self.get_SPARQL_dataframe(proper, language)
                        if isinstance(e, pd.DataFrame):
                            test_list.append(e)
                    if len(test_list) > 0:
                        WDdf = pd.concat(test_list)
                        how_matched = 'SearX'  # proper_name is found in Wikidata
        # OpenRefine-Reconciliation API
        if openrefine:
            if not isinstance(WDdf, pd.DataFrame):
                proper_name = self.get_openrefine_bestname(name_in_data)
                if proper_name:
                    WDdf = self.get_SPARQL_dataframe(proper_name, language)
                    how_matched = 'OpenRefine'  # proper_name is found in Wikidata
        return [WDdf, how_matched, proper_name]

    def _get_SPARQL_dataframe(self,
                              name, language,
                              url=url_query, extra='', ptype=ptype,
                              _ri_: int=None, _ci_: int=None, _table_id_: str=None):
        """
        Parameters
        ----------
        name : str
            Possible mention in wikidata.
        url : str, optional
            SPARQL-endpoint. The default is "https://query.wikidata.org/sparql".
        extra : str
            An extra parameter that will be also SELECTed in the SPARQL query.
        Returns
        -------
        output : pd.DataFrame
            Dataframe created from the json-file returned by SPARQL-endpoint.
        """
        name = name.replace('"', '\\\"')
        if language:
            lang = language
        else:
            lang = get_language(name)

        ents: List[QNode] = [self.qnodes[qnode_id] for qnode_id in self.table2links[_table_id_].get((_ri_, _ci_), [])]
        if len(ents) == 0:
            return None

        results = []
        for ent in ents:
            item = self.get_ent_uri(ent.id)
            item_types = [self.get_ent_uri(stmt.value.as_entity_id()) for stmt in ent.props.get("P31", [])]
            if len(item_types) == 0:
                item_types = [None]

            for p, stmts in ent.props.items():
                for stmt in stmts:
                    value_label = self.extract_datavalue_label(stmt.value, lang)
                    if stmt.value.is_qnode():
                        if stmt.value.as_entity_id().startswith("P"):
                            value = self.get_prop_uri(stmt.value.as_entity_id())
                        elif stmt.value.as_entity_id().startswith("Q"):
                            value = self.get_ent_uri(stmt.value.as_entity_id())
                        else:
                            value = stmt.value.as_entity_id()
                        if stmt.value.as_entity_id() not in self.qnodes:
                            # maintaining a list of qnodes we have know this error due to not in wikidata
                            # assert stmt.value.as_entity_id() in {"Q71963720", "Q9681307", "Q91051973"}, stmt.value.as_entity_id()
                            value_types = [None]
                        else:
                            value_types = [self.get_ent_uri(x.value.as_entity_id()) for x in
                                        self.qnodes[stmt.value.as_entity_id()].props.get("P31", [])]
                            if len(value_types) == 0:
                                value_types = [None]
                    else:
                        value = value_label
                        value_types = [None]
                    for item_type in item_types:
                        for value_type in value_types:
                            result = {
                                "item": item,
                                "p1": "http://www.w3.org/2000/01/rdf-schema#label",
                                "p2": self.get_prop_uri(p),
                                "value": value,
                                "valueLabel": value_label,
                                "psValueLabel": value_label
                            }
                            if value_type is not None:
                                result['valueType'] = value_type
                            if item_type is not None:
                                result['itemType'] = item_type
                            results.append(result)

        output = pd.DataFrame(results, dtype=str)
        return output

    def _get_SPARQL_dataframe_type(self, name, datatype, language, url=url_query, ptype=ptype, _ri_: int=None, _ci_: int=None, _table_id_: str=None):
        # their function is just find an entity has the name and satisfy the type
        assert datatype[0] == 'Q' and datatype[1:].isdigit()
        name = name.replace('"', '\\\"')
        if language:
            lang = language
        else:
            lang = get_language(name)
        ents: List[QNode] = [self.qnodes[qnode_id] for qnode_id in self.table2links[_table_id_].get((_ri_, _ci_), [])]
        ents = [
            ent
            for ent in ents
            if any(stmt.value.as_entity_id() == datatype for stmt in ent.props.get("P31", []))
        ]
        if len(ents) == 0:
            return None

        results = []
        for ent in ents:
            item = self.get_ent_uri(ent.id)
            item_label = ent.label.as_lang(lang, '')
            if item_label == '':
                item_label = ent.label.as_lang('en', '')
                assert item_label != ''
            result = {
                "item": item,
                "itemLabel": item_label,
            }
            results.append(result)
        output = pd.DataFrame(results, dtype=str)
        return output

    def _lookup(self, name_in_data, language, metalookup=True, openrefine=False, _ri_: int=None, _ci_: int=None, _table_id_: str=None):
        """
        Parameters
        ----------
        name_in_data : str
            Search string.
        Returns
        -------
        list = [WDdf, how_matched]
            WDdf is a dataframe with SPARQL-request
            how_matched shows how the string was matched
                0: SPARQL-Wikidata
                1: OpenRefine Suggest API
                2: Searx-metasearch
        """
        assert _ri_ is not None and _ci_ is not None
        how_matched = ''
        proper_name = ''
        # Search entity using WD SPARQL-endpoint
        WDdf = self.get_SPARQL_dataframe(name_in_data, language, _ri_=_ri_, _ci_=_ci_, _table_id_=_table_id_)
        if isinstance(WDdf, pd.DataFrame):
            proper_name = name_in_data
            how_matched = 'SPARQL'  # This means we have found a mention of 'name_in_data' in Wikidata using single SPARQL-query
        if isinstance(WDdf, pd.DataFrame):
            if 'item' in WDdf.columns:
                if all(WDdf.item.str.contains('wikipedia')):
                    WDdf = None
        # Searx-metasearch-engine API
        if metalookup:
            if not isinstance(WDdf, pd.DataFrame):
                # MODIFIED
                proper_name = self.get_searx_bestname(name_in_data)
                if proper_name:
                    test_list = []
                    for proper in proper_name:
                        e = self.get_SPARQL_dataframe(proper, language)
                        if isinstance(e, pd.DataFrame):
                            test_list.append(e)
                    if len(test_list) > 0:
                        WDdf = pd.concat(test_list)
                        how_matched = 'SearX'  # proper_name is found in Wikidata
        # OpenRefine-Reconciliation API
        if openrefine:
            if not isinstance(WDdf, pd.DataFrame):
                # MODIFIED
                proper_name = self.get_openrefine_bestname(name_in_data)
                if proper_name:
                    WDdf = self.get_SPARQL_dataframe(proper_name, language)
                    how_matched = 'OpenRefine'  # proper_name is found in Wikidata
        return [WDdf, how_matched, proper_name]

    def get_wikidata_title(self, url: str):
        """A re-implementation of the function get_wikidata_title. Instead of using the wikidata-api as in the original
        function, we read the label directly from the database. We return the label in english as in the original function
        
        """
        if url.startswith("http://www.wikidata.org/prop/direct/"):
            url = url.replace("http://www.wikidata.org/prop/direct/", "http://www.wikidata.org/entity/")

        if WikidataNamespace.is_abs_uri_entity(url):
            ent_id = WikidataNamespace.get_entity_id(url)
            ent = self.qnodes[ent_id]
            return ent.label.lang2value.get('en', '')
        
        raise NotImplementedError(url)

    def _get_common_class(self, classes: List[str]):
        """A re-implementation of the function get_common_class. The function find common parent classes
        that has the minimum total length to given classes. If there is no common class, return the first class in the list
        """
        assert all(WikidataNamespace.is_abs_uri_entity(c) for c in classes)
        ents = [self.qnodes[WikidataNamespace.get_entity_id(class_id)] for class_id in classes]
        # parent -> entity & their distance
        parents = {}

        # in their method it means the maximum level of the tree or distance
        # see: https://blazegraph.com/database/apidocs/com/bigdata/rdf/graph/impl/bd/GASService.html
        maxIterations = 10

        # discover all parents
        for ent in ents:
            queue: List[Tuple[str, int]] = [(ent.id, 0)]
            visited = {}
            while len(queue) > 0:
                ptr, distance = queue.pop(0)
                visited[ptr] = min(distance, visited.get(ptr, float('inf')))

                if distance >= maxIterations:
                    break

                for stmt in self.qnodes[ptr].props.get("P279", []):
                    if stmt.value.is_entity_id():
                        pid = stmt.value.as_entity_id()
                        if pid in visited and visited[pid] >= distance + 1:
                            continue
                        queue.append((pid, distance + 1))
            
            for ptr, distance in visited.items():
                if ptr not in parents:
                    parents[ptr] = []
                parents[ptr].append(distance)
            
        commonParents = [
            (p, sum(distances))
            for p, distances in parents.items()
            if len(distances) == len(ents)
        ]
        if len(commonParents) == 0:
            output = classes[0]
        else:
            output = WikidataNamespace.get_entity_abs_uri(min(commonParents, key=lambda x: x[1])[0])

        # below check verifies that we implement it correctly, for a few cases it fails due to differences in
        # the local db & current wikidata.
        # if self._original_get_common_class(classes) != output:
        #     # the only way the result is different is that it's another class of the same distance
        #     assert len(commonParents) > 0
        #     _tmp1 = WikidataNamespace.get_entity_id(output)
        #     _tmp2 = WikidataNamespace.get_entity_id(self._original_get_common_class(classes))
        #     dist1 = [x for x in commonParents if x[0] == _tmp1][0][1]
        #     dist2 = [x for x in commonParents if x[0] == _tmp2]
        #     if len(dist2) == 0 or dist1 != dist2[0][1]:
        #         print("Diff", classes, get_common_class(classes), output)
        #         assert False
        return output

    def wrap_cell_based_function(self, fn):
        """Wrap those function so that we can call them with row index and column index"""
        @functools.wraps(fn)
        def wrap_impl(*kargs, **kwargs):
            for k in ['_ri_', '_ci_', '_table_id_']:
                if k in kwargs:
                    kwargs.pop(k)
            return fn(*kargs, **kwargs)
        return wrap_impl

    def big_cache_get_SPARQL_dataframe_type2(self, big_cache_dir: Path, override_fn=None):
        big_cache_dir.mkdir(exist_ok=True, parents=True)
        def fn(datatype, language):
            assert datatype[0] == 'Q' and datatype[1:].isdigit()
            outfile = big_cache_dir / f"{datatype}_{language}.json"
            if outfile.exists():
                return deserialize_json(outfile)
            if override_fn is not None:
                result = override_fn(datatype, language)
            else:
                result = self._original_get_SPARQL_dataframe_type2(datatype, language)
            serialize_json(result, outfile)
            return result
        return fn

    def get_prop_uri(self, id):
        assert id[0] == 'P' and id[1:].isdigit(), id
        return 'http://www.wikidata.org/prop/' + id

    def get_ent_uri(self, id):
        assert id[0] == 'Q' and id[1:].isdigit(), id
        return 'http://www.wikidata.org/entity/' + id

    def extract_datavalue_label(self, value: DataValue, language):
        # implementation of this follow wikibase:label service
        # example query: https://query.wikidata.org/#SELECT%20DISTINCT%20%3Fitem%20%3FitemType%20%3FitemLabel%20%3Fp%20%3Fvalue%20%3FvalueType%20%3FvalueLabel%20%3FpsvalueLabel%20WHERE%20%7B%0A%20%20%3Fitem%20%3Fp%20%3Fvalue%20.%0A%20%20OPTIONAL%20%7B%20%3Fitem%20wdt%3AP31%20%3FitemType.%20%7D%0A%20%20OPTIONAL%20%7B%20%3Fvalue%20wdt%3AP31%20%3FvalueType.%20%7D%0A%20%20OPTIONAL%20%7B%0A%20%20%20%20%3Fwdproperty%20wikibase%3Aclaim%20%3Fp2%20%3B%0A%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20wikibase%3AstatementProperty%20%3Fpsproperty%20.%0A%20%20%20%20%3Fvalue%20%3Fpsproperty%20%3Fpsvalue%20.%0A%20%20%7D%0A%20%20SERVICE%20wikibase%3Alabel%20%7B%20bd%3AserviceParam%20wikibase%3Alanguage%20%22vn%22.%20%7D%0A%20%20VALUES%20%3Fitem%20%7B%20wd%3AQ5806%20%7D%20%0A%20%20VALUES%20%3Fp%20%7B%20p%3AP571%20%7D%0A%7D%0ALIMIT%20100000
        if value.is_entity_id():
            if value.as_entity_id() not in self.qnodes:
                label = value.as_entity_id()
            else:
                label = self.qnodes[value.as_entity_id()].label
                label = label.as_lang(language)
                if label == '':
                    label = value.as_entity_id()
            return label
        if value.is_globe_coordinate():
            return f"Point({value.value['longitude']} {value.value['latitude']})"
        if value.is_quantity():
            amount = value.value['amount']
            if amount[0] == '+':
                amount = amount[1:]
            return amount
        if value.is_string():
            return value.value
        if value.is_time():
            # okay, this will be different from what you get in the wikibase:label service
            # they have bug in processing BC julian date that they subtract 1 year
            # and we need to convert 00-00 to 01-01 as that's what they do
            # their code: https://gerrit.wikimedia.org/r/c/mediawiki/extensions/Wikibase/+/198419/17/repo/includes/rdf/JulianDateTimeValueCleaner.php#11
            timestr = value.value['time']
            if value.value['calendarmodel'] == 'http://www.wikidata.org/entity/Q1985786':
                first, second = timestr.split("T")
                y, m, d = first.rsplit("-", 2)
                y, m, d = jd2gcal(*jcal2jd(int(y), int(m), int(d)))[:3]
                timestr = f"{y}-{m}-{d}T{second}"
            else:
                if timestr[0] == '+':
                    if timestr.find("-00-00T00:00:00") != -1:
                        timestr = timestr.replace("-00-00T00:00:00", "-01-01T00:00:00")
                    timestr = timestr[1:]
                else:
                    assert timestr[0] == '-'
                    if timestr.find("-00-00T00:00:00") != -1:
                        timestr = timestr.replace('-00-00T00:00:00', '-01-01T00:00:00')
            return timestr
        if value.is_mono_lingual_text():
            return value.value['text']
        assert False, ("unreachable!", value)

    def test(self):
        WDdf = pd.DataFrame([])
        for i in range(len(WDdf)):
            r = WDdf.iloc[i]
            if not r['item'].endswith("Q2399866"): continue
            if not r['p2'].endswith("P17"): continue
            print(r)


def detect_name(value):
    """
    This is an extended function from https://github.com/IBCNServices/CSV2KG/blob/master/csv2kg/util.py
    It detects names like 'V. Futter' or 'Ellen V. Futter' and returns 'Futter' or 'Ellen Futter'
    """
    match2 = re.match("^(\w\. )+([\w\-']+)$", value, re.UNICODE)
    match3 = re.match("^([\w\-']+ )+(\w\. )+([\w\-']+)$", value, re.UNICODE)
    if match2 is not None:
        return match2.group(2)
    if match3 is not None:
        return match3.group(1) + match3.group(3)
    return None


def match(WDdf, target_value):
    """Performs contextual matching for input dataframe and input target_value.
    Returns the dataframe constrained to the objects equal to target_value."""
    if isfloat(target_value):
        target_value = target_value.replace(',', '')
    isdate = re.match(r"^\d{4}-\d{2}-\d{2}", target_value)
    # 0. Normalize dates
    match_date = re.match(r"^(\d{4})/(\d{2})/(\d{2})$", target_value)
    if match_date:
        isdate = True
        target_value = match_date[1] + "-" + match_date[2] + "-" + match_date[3]
    # 1. exact matching of valueLabels
    df = WDdf[WDdf.valueLabel == target_value]
    # 2a. case-insensitive exact matching of valueLabels
    if df.empty and not isfloat(target_value):
        df = WDdf[WDdf.valueLabel.str.lower() == str.lower(target_value)]
        # 2b. inexact matching of valueLabels with high cuttoff=0.95
        if df.empty:
            approx_matches = difflib.get_close_matches(target_value, WDdf.valueLabel.to_list(), n=3, cutoff=0.95)
            if len(approx_matches) == 0:
                approx_matches = difflib.get_close_matches(target_value, WDdf.valueLabel.to_list(), n=3, cutoff=0.5)
            if len(approx_matches) > 0:
                df = WDdf[WDdf.valueLabel.isin(approx_matches)]
            else:
                if detect_name(target_value):
                    df = WDdf[WDdf.valueLabel.apply(
                        lambda x: all(word in x.lower() for word in detect_name(target_value).lower()))]
    # 3. approximate date matching
    if df.empty and isdate:
        wd_dates = [x for x in WDdf.valueLabel.to_list() if re.match(r"^\d{4}-\d{2}-\d{2}", x)]
        target_datetime = date.fromisoformat(target_value)
        approximate_match = min(wd_dates, key=lambda x: abs(date.fromisoformat(x[:10]) - target_datetime))
        # check that approximate date is within 6 months
        delta = date.fromisoformat(approximate_match[:10]) - target_datetime
        if abs(delta.days) < 183:
            df = WDdf[WDdf.valueLabel == approximate_match]
    # 4. approximate floating numbers matching
    if df.empty and isfloat(target_value):
        wd_floats = [x for x in WDdf.valueLabel.to_list() if isfloat(x)]
        # get only one approximate match
        # approximate_match = min(wd_floats, key=lambda x: abs(float(x) - float(target_value)))
        # get all approximate matches within a 2% range
        all_approximate_matches = [x for x in wd_floats if
                                   (abs(float(x) - float(target_value)) <= 0.02 * abs(float(x)))]
        if len(all_approximate_matches) > 0:
            df = WDdf[WDdf.valueLabel.isin(all_approximate_matches)]

    return df


def preprocessing(filecsv):
    """Simple preprocessing of a dataframe using ftfy.fix_text()."""
    filecsv = filecsv.fillna("")
    if len(filecsv.columns) == 1:  # Data augmentation for single-column tables
        filecsv[1] = filecsv[0]
    filecsv = filecsv.applymap(lambda x: ftfy.fix_text(x))  # fix encoding and clean text
    return filecsv


def contextual_matching(filecsv, filename='', language='', semtab = False,
                        default_cpa=None, default_cea=None, default_nomatch=None,
                        step3=False, step4=False, step5=True, step6=True, url=url_front):
    """Five-steps contextual matching for an input dataframe filecsv.
    Step 2 is always executed. Steps 3-6 are optional.
    The lists cpa_list and cea_list with annotations are returned.
    If semtab=True, a property must have URL at www.wikidata.org and col0=1.
    If semtab=False, a property may have URL at www.w3.org and col0=0.
    """
    if semtab:
        col0 = 1
    else:
        col0 = 0
    if default_cpa:
        cpa_list = default_cpa
    else:
        cpa_list = []
    if default_cea:
        cea_list = default_cea
    else:
        cea_list = []
    if default_nomatch:
        nomatch = default_nomatch
    else:
        nomatch = []
    (rows, cols) = filecsv.shape
    nomatch_row = []
    fullymatched_rows = []
    cpa_ind = len(cpa_list)
    cea_ind = len(cea_list)
    bbw_search_fn = BBWSearchFn.get_instance()
    # STEP 2 in the workflow
    step2 = True  # Step 2 is always executed
    if step2:
        for row in range(1, rows):  # We start here from row=1, because there are "col0" and "col1" in row=0
            name_in_data = filecsv.iloc[row, 0]
            [WDdf, how_matched, proper_name] = bbw_search_fn.lookup(name_in_data, language, _ri_=row, _ci_=0, _table_id_=filename)  # Lookup using the value from the 0-column
            this_row_item = []
            matches_per_row = 0
            cpa_row_ind = len(cpa_list)
            # for each other column look for a match of the value within the wikidata dataframe
            if isinstance(WDdf, pd.DataFrame):
                if not WDdf.empty:
                    for col in range(col0, cols):
                        try:
                            df = match(WDdf, filecsv.iloc[row, col])
                            if semtab:
                                df_prop = df[(df.p2.str.contains(url)) & (
                                    ~df.item.str.contains('/statement/'))]
                            else:
                                df_prop = df
                            properties = [
                                x.replace("/prop/P", "/prop/direct/P").replace("/direct-normalized/", "/direct/") for x
                                in df_prop.p2.to_list()]
                            properties = list(set(zip(properties, df_prop.item.to_list())))
                            if len(properties) > 0:
                                matches_per_row += 1
                                if matches_per_row == cols - 1:
                                    fullymatched_rows.append(row)
                            item = list(set(df_prop.item.to_list()))
                            if 'itemType' in df_prop:
                                itemType = list(set([k for k in df_prop.itemType.to_list() if k is not np.nan]))
                            else:
                                itemType = []
                            df_value = df[
                                (~df.value.str.contains('/statement/')) & (df.value.str.contains(url))]
                            if not df_value.empty:
                                value, valueType = list(set(df_value.value.to_list())), list(
                                    set([k for k in df_value.valueType.to_list() if k is not np.nan]))
                            else:
                                value, valueType = [], []
                            if properties and item:
                                cpa_list.append(
                                    [filename, row, 0, col, properties, item, itemType, how_matched, proper_name])
                            if item:
                                cea_list.append([filename, row, 0, item, itemType, how_matched, proper_name])
                                this_row_item.extend(item)
                            if value:
                                cea_list.append(
                                    [filename, row, col, value, valueType, 'Step 2: ' + how_matched, proper_name])
                        except Exception:
                            pass
            else:
                nomatch.append([filename, row, name_in_data, proper_name])
            # Take the most possible item for this row and remove the properties which are not taken from this item
            if len(this_row_item) > 0:
                this_row_item = Counter(this_row_item).most_common(1)[0][0]
                for i, cpa_row in enumerate(cpa_list[cpa_row_ind:]):
                    if len(cpa_row[4]) > 0:
                        cpa_list[cpa_row_ind + i][4] = [prop for prop in cpa_row[4] if prop[1] == this_row_item]
            # Define the unannotated rows
            if row == rows - 1:  # After the last row
                nomatch_row = [r for r in range(1, rows) if r not in fullymatched_rows]
    # Choose only entity columns, not the literal columns
    entity_columns = list(set([k[2] for k in cea_list[cea_ind:] if k[2] != 0 and k[3]]))

    # STEP 3 in the workflow
    if step3:
        # MATCHING item,itemType,value and valueType via properties and values in the entity-columns
        # Calculate the properties and find the item, itemType, value and valueType:
        col_prop = {}
        for row_prop in cpa_list[cpa_ind:]:
            if len(row_prop[4]) > 0:
                if col_prop.get(row_prop[3]):
                    col_prop[row_prop[3]].extend([cprop[0].split('/')[-1] for cprop in row_prop[4]])
                else:
                    col_prop[row_prop[3]] = [cprop[0].split('/')[-1] for cprop in row_prop[4]]
        col_prop.update((key, Counter(value).most_common(1)[0][0]) for key, value in col_prop.items())
        if len(entity_columns) > 0:
            for nrow in nomatch_row or []:
                try:  # Try to use ALL entity columns AT ONCE and their property-relations to the main column
                    WDdf = get_SPARQL_dataframe_prop(prop=[col_prop[ncol] for ncol in entity_columns],
                                                     value=[filecsv.iloc[nrow, ncol] for ncol in entity_columns])
                    bestname = list(set(
                        difflib.get_close_matches(filecsv.iloc[nrow, 0], WDdf.itemLabel.to_list(), n=3, cutoff=0.81)))
                    WD = WDdf[WDdf.itemLabel.isin(bestname)]
                    for col in range(col0, cols):
                        try:
                            df = match(WD, filecsv.iloc[nrow, col])
                            item = list(set(df.item.to_list()))
                            if 'itemType' in df.columns:
                                itemType = list(set([k for k in df.itemType.to_list() if k is not np.nan]))
                            else:
                                itemType = []
                            df_value = df[
                                (~df.value.str.contains('/statement/')) & (df.value.str.contains(url))]
                            if not df_value.empty:
                                value, valueType = list(set(df_value.value.to_list())), list(
                                    set([k for k in df_value.valueType.to_list() if k is not np.nan]))
                            else:
                                value, valueType = [], []
                            if item:
                                cea_list.append([filename, nrow, 0, item, itemType, 'Step 3', bestname])
                            if value:
                                cea_list.append([filename, nrow, col, value, valueType, 'Step 3', bestname])
                        except Exception:
                            pass
                except Exception:
                    pass

    # STEP 4 in the workflow
    if step4:
        # # MATCHING via the tail-entity-label and main-column-label
        for row in nomatch_row or []:
            for col in entity_columns or []:
                value_to_match = filecsv.iloc[row, col]
                if not isfloat(value_to_match) and not re.match(r"^(\d{4})/(\d{2})/(\d{2})$", value_to_match):
                    try:
                        WDitem = get_SPARQL_dataframe_item(value_to_match, language)
                        bestname = difflib.get_close_matches(filecsv.iloc[row, 0], WDitem.itemLabel.to_list(), n=2,
                                                             cutoff=0.95)
                        if len(bestname) == 0:
                            bestname = difflib.get_close_matches(filecsv.iloc[row, 0], WDitem.itemLabel.to_list(), n=2,
                                                                 cutoff=0.905)
                        if len(bestname) > 0:
                            WD = WDitem[WDitem.itemLabel.isin(bestname)]
                            item = list(set(WD.item.to_list()))
                            if 'itemType' in WD.columns:
                                itemType = list(set([k for k in WD.itemType.to_list() if k is not np.nan]))
                            else:
                                itemType = []
                            properties = [
                                x.replace("/prop/P", "/prop/direct/P").replace("/direct-normalized/", "/direct/") for x
                                in WD.p2.to_list()]
                            properties = list(set(zip(properties, WD.item.to_list())))
                            WD = WD[(~WD.value.str.contains('/statement/')) & (WD.value.str.contains(url))]
                            if not WD.empty:
                                value, valueType = list(set(WD.value.to_list())), list(
                                    set([k for k in WD.valueType.to_list() if k is not np.nan]))
                            else:
                                value, valueType = [], []
                            if properties and item:
                                cpa_list.append(
                                    [filename, row, 0, col, properties, item, itemType, 'tail-entity-label main-label',
                                     bestname])
                            if item:
                                cea_list.append([filename, row, 0, item, itemType, 'Step 4', bestname])
                            if value:
                                cea_list.append([filename, row, col, value, valueType, 'Step 4', bestname])
                    except Exception:
                        pass

    # MATCHING via column types in Steps 5 and 6
    if step5 or step6:
        # Estimate the types of columns in this table
        col_type = {}
        for row_type in cea_list[cea_ind:]:
            if len(row_type[4]) > 0:
                if col_type.get(row_type[2]):
                    col_type[row_type[2]].extend([etype.split('/')[-1] for etype in row_type[4]])
                else:
                    col_type[row_type[2]] = [etype.split('/')[-1] for etype in row_type[4]]
        col_type.update((key, [ct[0] for ct in Counter(value).most_common(2)]) for key, value in col_type.items())

    # STEP 5 in the workflow
    if step5:
        # We match tail-entities using its type and itemLabel.
        for nrow in nomatch_row or []:
            for ncol in entity_columns or []:
                try:
                    for column_type in col_type[ncol]:
                        WDtype = bbw_search_fn.get_SPARQL_dataframe_type(filecsv.iloc[nrow, ncol], column_type, language)
                        item = list(set(WDtype.item.to_list()))
                        if item:
                            cea_list.append(
                                [filename, nrow, ncol, item, [url+"/entity/" + column_type],
                                 'Step 5', list(set(WDtype.itemLabel.to_list()))])
                except Exception:
                    pass

    # STEP 6 in the workflow
    if step6:
        # We match entities in the main column using its datatype
        if col_type.get(0) and len(nomatch_row) > 0:
            for column_type in col_type.get(0):
                try:
                    WDtype_itemLabel_to_list = bbw_search_fn.get_SPARQL_dataframe_type2(column_type, language)
                    for row in nomatch_row or []:
                        proper_name = difflib.get_close_matches(filecsv.iloc[row, 0], WDtype_itemLabel_to_list, n=15,
                                                                cutoff=0.95)
                        if len(proper_name) == 0:
                            proper_name = difflib.get_close_matches(filecsv.iloc[row, 0], WDtype_itemLabel_to_list,
                                                                    n=15, cutoff=0.9)
                            if len(proper_name) == 0:
                                proper_name = difflib.get_close_matches(filecsv.iloc[row, 0],
                                                                        WDtype_itemLabel_to_list, n=15, cutoff=0.8)
                                if len(proper_name) == 0:
                                    proper_name = difflib.get_close_matches(filecsv.iloc[row, 0],
                                                                            WDtype_itemLabel_to_list, n=15,
                                                                            cutoff=0.7)
                        this_row_item = []
                        cpa_row_ind = len(cpa_list)
                        if len(proper_name) > 0:
                            test_list = []
                            for proper in proper_name:
                                e = bbw_search_fn._original_get_SPARQL_dataframe(proper, language, extra='?itemLabel ')
                                if isinstance(e, pd.DataFrame):
                                    test_list.append(e)
                            if len(test_list) > 0:
                                WDdf = pd.concat(test_list)
                            else:
                                # MODIFIED: add here otherwise, may be undefined (bug in their code)
                                WDdf = None
                            if isinstance(WDdf, pd.DataFrame):
                                for col in range(col0, cols):
                                    try:
                                        df = match(WDdf, filecsv.iloc[row, col])
                                        if semtab:
                                            df_prop = df[df.p2.str.contains(url)]
                                        else:
                                            df_prop = df
                                        properties = [
                                            x.replace("/prop/P", "/prop/direct/P").replace("/direct-normalized/",
                                                                                           "/direct/") for x in
                                            df_prop.p2.to_list()]
                                        properties = list(set(zip(properties, df_prop.item.to_list())))
                                        item = list(set(df_prop.item.to_list()))
                                        if 'itemType' in df_prop.columns:
                                            itemType = list(
                                                set([k for k in df_prop.itemType.to_list() if k is not np.nan]))
                                        else:
                                            itemType = []
                                        df_value = df[(~df.value.str.contains('/statement/')) & (
                                            df.value.str.contains(url))]
                                        if not df_value.empty:
                                            value, valueType = list(set(df_value.value.to_list())), list(
                                                set([k for k in df_value.valueType.to_list() if k is not np.nan]))
                                        else:
                                            value, valueType = [], []
                                        if properties and item:
                                            cpa_list.append(
                                                [filename, row, 0, col, properties, item, itemType, 'Step 6',
                                                 list(set(df_prop.itemLabel.to_list()))])
                                        if item:
                                            cea_list.append([filename, row, 0, item, itemType, 'Step 6',
                                                             list(set(df_prop.itemLabel.to_list()))])
                                            this_row_item.extend(item)
                                        if value:
                                            cea_list.append([filename, row, col, value, valueType, 'Step 6',
                                                             list(set(df_value.itemLabel.to_list()))])
                                    except Exception:
                                        pass
                        # Take the most possible item for this row and remove the properties which are not taken from this item
                        if len(this_row_item) > 0:
                            this_row_item = Counter(this_row_item).most_common(1)[0][0]
                            for i, cpa_row in enumerate(cpa_list[cpa_row_ind:]):
                                if len(cpa_row[4]) > 0:
                                    cpa_list[cpa_row_ind + i][4] = [prop for prop in cpa_row[4] if
                                                                    prop[1] == this_row_item]
                except BBWSearchFn.SpecialException:
                    raise
                except Exception:
                    pass
    return [cpa_list, cea_list, nomatch]


def postprocessing(cpa_list, cea_list, filelist=None, target_cpa=None, target_cea=None, target_cta=None, gui=False):
    """Postprocessing is performed for input lists cpa_list and cea_list.
    The target-dataframes are optional. If they are given,
    only target-annotations are returned in """
    # Create dataframe using the non-matched names from the main column (0), the corresponding filename and row
    # nm = pd.DataFrame(nomatch)
    # Create CPA-dataframe from the list and find the most frequent property
    bbw_cpa_few = pd.DataFrame(cpa_list, columns=['file', 'row', 'column0', 'column', 'property', 'item', 'itemType',
                                                  'how_matched', 'what_matched'])
    bbw_cpa_sub = bbw_cpa_few.groupby(['file', 'column0', 'column']).agg(
        {'property': lambda x: tuple(x)}).reset_index()  # because a list is unhashable
    bbw_cpa_sub['property'] = bbw_cpa_sub['property'].apply(lambda x: [y[0] for subx in x for y in subx])  # flatten
    bbw_cpa_sub['property'] = bbw_cpa_sub['property'].apply(lambda x: Counter(x).most_common(2))
    bbw_cpa_sub['property'] = bbw_cpa_sub['property'].apply(lambda x: None if len(x) == 0 else x[0][0])
    bbw_cpa_sub = bbw_cpa_sub.dropna()
    # Keep only the target columns for CPA-challenge
    if filelist and isinstance(target_cpa, pd.DataFrame):
        bbw_cpa_sub = pd.merge(
            target_cpa[target_cpa.file.isin(filelist)].astype({"file": str, "column0": int, "column": int}),
            bbw_cpa_sub.astype({"file": str, "column0": int, "column": int, "property": str}),
            on=['file', 'column0', 'column'], how='inner')
    # Create CEA-dataframe from the list and drop rows with None or empty lists
    bbw_few = pd.DataFrame(cea_list,
                           columns=['file', 'row', 'column', 'item', 'itemType', 'how_matched', 'what_matched'])
    # Prepare dataframe for CEA-challenge
    bbw_cea_sub = bbw_few.groupby(['file', 'row', 'column']).agg({'item': lambda x: tuple(x)}).reset_index()
    bbw_cea_sub['item'] = bbw_cea_sub['item'].apply(lambda x: [y for subx in x for y in subx])
    bbw_cea_sub['item'] = bbw_cea_sub['item'].apply(lambda x: Counter(x).most_common(2))
    bbw_cea_sub['item'] = bbw_cea_sub['item'].apply(lambda x: None if len(x) == 0 else x[0][0])
    bbw_cea_sub = bbw_cea_sub.dropna()
    # Keep only the target columns for CEA-challenge
    if filelist and isinstance(target_cea, pd.DataFrame):
        bbw_cea_sub = pd.merge(
            target_cea[target_cea.file.isin(filelist)].astype({"file": str, "row": int, "column": int}),
            bbw_cea_sub.astype({"file": str, "row": int, "column": int, "item": str}),
            on=['file', 'row', 'column'], how='inner')
    # Drop None-rows from bbw_few before getting itemType for CTA:
    bbw_few = bbw_few.dropna()
    bbw_few = bbw_few[bbw_few['itemType'].map(lambda x: len(x)) > 0]
    # Prepare dataframe for CTA-challenge
    bbw_cta_one = bbw_few.groupby(['file', 'column']).agg({'itemType': lambda x: tuple(x)}).reset_index()
    bbw_cta_one['itemType'] = bbw_cta_one['itemType'].apply(lambda x: [y for subx in x for y in subx])
    # bbw_cta_one['itemType'] = bbw_cta_one['itemType'].apply(lambda x: get_common_class(x) if len(x)>1 else x[0])
    bbw_cta_one['itemType'] = bbw_cta_one['itemType'].apply(lambda x: Counter(x).most_common(2))
    bbw_cta_one['itemType'] = bbw_cta_one['itemType'].apply(lambda x: get_one_class(x))
    bbw_cta_sub = bbw_cta_one.dropna()
    # Keep only the target columns for CTA-challenge
    if filelist and isinstance(target_cta, pd.DataFrame):
        bbw_cta_sub = pd.merge(target_cta[target_cta.file.isin(filelist)].astype({"file": str, "column": int}),
                               bbw_cta_sub.astype({"file": str, "column": int, "itemType": str}),
                               on=['file', 'column'], how='inner')
    # Print statistics
    if filelist and not gui:
        stat_cpa_matched = len(bbw_cpa_sub)
        if isinstance(target_cpa, pd.DataFrame):
            stat_cpa_target = len(target_cpa[target_cpa.file.isin(filelist)])
        stat_cea_matched = len(bbw_cea_sub)
        if isinstance(target_cea, pd.DataFrame):
            stat_cea_target = len(target_cea[target_cea.file.isin(filelist)])
        stat_cta_matched = len(bbw_cta_sub)
        if isinstance(target_cta, pd.DataFrame):
            stat_cta_target = len(target_cta[target_cta.file.isin(filelist)])
        print('\n*** Internal statistics ***')
        print('Task', 'Coverage', 'Matched', 'Total', 'Unmatched', sep='\t')
        try:
            print('CEA', round(stat_cea_matched / stat_cea_target, 4), stat_cea_matched, stat_cea_target,
                  stat_cea_target - stat_cea_matched, sep='\t')
        except Exception:
            pass
        try:
            print('CTA', round(stat_cta_matched / stat_cta_target, 4), stat_cta_matched, stat_cta_target,
                  stat_cta_target - stat_cta_matched, sep='\t')
        except Exception:
            pass
        try:
            print('CPA', round(stat_cpa_matched / stat_cpa_target, 4), stat_cpa_matched, stat_cpa_target,
                  stat_cpa_target - stat_cpa_matched, sep='\t')
        except Exception:
            pass
    return [bbw_cpa_sub, bbw_cea_sub, bbw_cta_sub]


def annotate(filecsv, filename='', language='', semtab=False, target_cpa=None, target_cta=None):
    """
    Parameters
    ----------
    filecsv : pd.DataFrame
        Input dataframe.
    filename : str
        A filename.
    Returns
    -------
    list
        [
        bbwtable - dataframe containing annotated Web Table,
        urltable - dataframe containing URLs of annotations,
        labeltable - dataframe containing labels of annotations,
        cpa_sub - dataframe with annotations for CPA task,
        cea_sub - dataframe with annotations for CEA task,
        cta_sub - dataframe with annotations for CTA task
        ].

    """
    filename = filename.replace('.csv', '')
    filecsv = preprocessing(filecsv)
    [cpa, cea, nomatch] = contextual_matching(filecsv, filename, language,
                                              semtab=semtab,
                                              step3=False, step4=False, step5=True,
                                              step6=True)
    [cpa_sub, cea_sub, cta_sub] = postprocessing(cpa, cea, [filename], target_cpa=target_cpa, target_cta=target_cta, gui=True)
    bbwtable = filecsv
    urltable = pd.DataFrame(columns=filecsv.columns)
    labeltable = pd.DataFrame(columns=filecsv.columns)
    bbw_search_fn = BBWSearchFn.get_instance()

    if not cea_sub.empty:
        for row in set(cea_sub.row.to_list()) or []:
            for column in set(cea_sub.column.to_list()) or []:
                _tmp = cea_sub.item[(cea_sub.row == row) & (cea_sub.column == column)].to_list()
                if len(_tmp) > 0:
                    link = _tmp[0]
                    if link:
                        label = bbw_search_fn.get_wikidata_title(link)
                        urltable.loc[row, column] = link
                        labeltable.loc[row, column] = label
                        bbwtable.loc[row, column] = '<a target="_blank" href="' + link + '">' + label + '</a>'
    if not cpa_sub.empty:
        for column in set(cpa_sub.column.to_list()) or []:
            _tmp = cpa_sub.property[cpa_sub.column == column].to_list()
            if len(_tmp) > 0:
                link = str(_tmp[0])
                label = bbw_search_fn.get_wikidata_title(link)
                bbwtable.loc['index', column] = '<a target="_blank" href="' + link + '">' + label + '</a>'
                urltable.loc['index', column] = link
                labeltable.loc['index', column] = label
    if not cta_sub.empty:
        for column in set(cta_sub.column.to_list()) or []:
            _tmp = cta_sub.itemType[cta_sub.column == column].to_list()
            if len(_tmp) > 0:
                link = str(_tmp[0])
                label = bbw_search_fn.get_wikidata_title(link)
                bbwtable.loc['type', column] = '<a target="_blank" href="' + link + '">' + label + '</a>'
                urltable.loc['type', column] = link
                labeltable.loc['type', column] = label
    bbwtable = bbwtable.rename(index={'index': 'property'})
    bbwtable = bbwtable.replace({np.nan: ''})
    bbwtable.columns = bbwtable.iloc[0]
    bbwtable = bbwtable[1:]
    urltable = urltable.rename(index={'index': 'property'})
    urltable = urltable.replace({np.nan: ''})
    urltable.columns = bbwtable.columns
    labeltable = labeltable.rename(index={'index': 'property'})
    labeltable = labeltable.replace({np.nan: ''})
    labeltable.columns = bbwtable.columns
    if len(urltable) > 0 and 'property' in urltable:
        urltable.loc['datatype'] = [get_datatype(prop) for prop in urltable.loc['property',:].to_list()]
        labeltable.loc['datatype'] = urltable.loc['datatype'].apply(lambda x: x.split('#')[-1] if x else '')
        bbwtable.loc['datatype'] = '<a target="_blank" href="' + urltable.loc['datatype'] + '">' + labeltable.loc['datatype'] + '</a>'
    return [bbwtable, urltable, labeltable, cpa_sub, cea_sub, cta_sub]
