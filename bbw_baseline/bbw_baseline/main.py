import os
from pathlib import Path
from typing import Dict, List, Mapping, Optional, Set, Tuple, TypedDict
from kgdata.wikidata.models import QNode
from sm.inputs.table import ColumnBasedTable
import pandas as pd
from bbw_baseline.bbw_modified import BBWSearchFn, annotate
from sm.misc import Parallel


Example = TypedDict(
    "Example",
    {
        "table": ColumnBasedTable,
        "links": Dict[Tuple[int, int], List[str]],
        "subj_col": Optional[Tuple[int, str]],
    },
)


class AnnotateFn:
    def __init__(
        self,
        is_semtab: bool,
        target_cpa_file: Optional[str],
        target_cta_file: Optional[str],
    ):
        self.is_semtab = is_semtab
        self.target_cpa_file = target_cpa_file
        self.target_cta_file = target_cta_file

        if self.is_semtab:
            assert self.target_cpa_file is not None
            assert self.target_cta_file is not None
            self.target_cpa = pd.read_csv(
                self.target_cpa_file, names=["file", "column0", "column"]
            )
            self.target_cta = pd.read_csv(
                self.target_cta_file, names=["file", "column"]
            )
        else:
            self.target_cpa = None
            self.target_cta = None

    def annotate(self, df: pd.DataFrame, table: ColumnBasedTable):
        annotation = annotate(
            df,
            semtab=self.is_semtab,
            filename=table.table_id,
            target_cpa=self.target_cpa,
            target_cta=self.target_cta,
        )

        return annotation
        # except:
        #     return
        # [web_table, url_table, label_table, cpa, cea, cta] = annotation
        # M.serialize_pkl({"table": table, "annotation": annotation}, outfile)
        # return annotation


def predict(
    qnodes: Mapping[str, QNode],
    examples: List[Example],
    cache_dir: Path,
    is_semtab: bool,
    target_cpa_file: Optional[str],
    target_cta_file: Optional[str],
):
    # convert examples to bbw format
    annotated_args = [
        make_dataframe(example["table"], example["links"], example["subj_col"])
        for example in examples
    ]
    table2newlinks = {
        examples[i]["table"].table_id: annotated_args[i][1]
        for i in range(len(examples))
    }

    BBWSearchFn.get_instance().setup(
        qnodes=qnodes,
        table2links=table2newlinks,
        cache_dir=cache_dir,
        enable_override=True,
        get_SPARQL_dataframe_type_flag=BBWSearchFn.USE_OUR_IMPL_FLAG,
        get_SPARQL_dataframe_type2_flag=BBWSearchFn.USE_OUR_IMPL_FLAG,
    )

    annotator = AnnotateFn(
        is_semtab=is_semtab,
        target_cpa_file=target_cpa_file,
        target_cta_file=target_cta_file,
    )
    pp = Parallel()
    annotations = pp.map(
        annotator.annotate,
        [
            (ann_args[0], examples[i]["table"])
            for i, ann_args in enumerate(annotated_args)
        ],
        show_progress=True,
        progress_desc="bbw annotates",
        is_parallel=True,
        n_processes=min(16, os.cpu_count() or 4),
    )
    outputs = []
    for i, example in enumerate(examples):
        annotation = annotations[i]
        newcol2oldcol = annotated_args[i][2]
        [web_table, url_table, label_table, cpa, cea, cta] = annotation
        cpa_pred = [
            (newcol2oldcol[row[1]], newcol2oldcol[row[2]], process_uri(row[3]))
            for row in cpa.values.tolist()
            if row[1] != row[2]
        ]
        cta_pred = {
            newcol2oldcol[row[1]]: process_uri(row[2]) for row in cta.values.tolist()
        }

        outputs.append(
            {
                "cpa": cpa_pred,
                "cta": cta_pred,
                "table": example["table"].table_id,
            }
        )

    return outputs


def make_dataframe(
    table: ColumnBasedTable,
    links: Dict[Tuple[int, int], List[str]],
    subj_col: Optional[Tuple[int, str]],
):
    """When we add the header as the first row, then the link need to adjust accordingly (+1)"""
    size = len(table.columns[0].values)
    columns = list(range(len(table.columns)))
    if subj_col is not None:
        ci = subj_col[0]
        # swap the subject column first
        columns[0], columns[ci] = columns[ci], columns[0]
    ci_old2new = {ci: i for i, ci in enumerate(columns)}
    lst = [[table.columns[ci].name for ci in columns]]
    for ri in range(size):
        row = []
        for ci in columns:
            row.append(table.columns[ci].values[ri])
        lst.append(row)
    df = pd.DataFrame(lst)
    new_links = {}
    for k, v in links.items():
        ri = k[0] + 1
        # cause we swap
        ci = ci_old2new[k[1]]
        new_links[ri, ci] = v
    ci_new2old = {i: ci for i, ci in enumerate(columns)}
    return df, new_links, ci_new2old


def process_uri(uri):
    if uri.startswith("http://www.wikidata.org/prop/direct/"):
        uri = uri.replace("http://www.wikidata.org/prop/direct/", "")
        assert uri[0] == "P" and uri[1:].isdigit()
        return uri
    if uri.startswith("http://www.wikidata.org/entity/"):
        uri = uri.replace("http://www.wikidata.org/entity/", "")
        assert uri[0] == "Q" and uri[1:].isdigit()
        return uri
    if uri[0] == "Q" and uri[1:].isdigit():
        return uri
    assert False
