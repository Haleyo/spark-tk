# vim: set encoding=utf-8

#  Copyright (c) 2016 Intel Corporation 
#
#  Licensed under the Apache License, Version 2.0 (the "License");
#  you may not use this file except in compliance with the License.
#  You may obtain a copy of the License at
#
#       http://www.apache.org/licenses/LICENSE-2.0
#
#  Unless required by applicable law or agreed to in writing, software
#  distributed under the License is distributed on an "AS IS" BASIS,
#  WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
#  See the License for the specific language governing permissions and
#  limitations under the License.
#

from sparktk.tkcontext import TkContext
from sparktk.dtypes import dtypes

def import_hbase(table_name, schema, start_tag=None, end_tag=None, tc=TkContext.implicit):
    """
    Import data from hbase table into frame

    :param table_name: (str) hbase table name
    :param schema: (list[list[str, str, type]]) hbase schema as a List of List(string) (columnFamily, columnName,
                   dataType for cell value)
    :param start_tag: (Optional(str)) optional start tag for filtering
    :param end_tag: (Optional(str)) optional end tag for filtering
    :return: (Frame) frame with data from hbase table

    Example
    ---------
    Load data into frame from a hbase table

    <skip>
        >>> frame = tc.frame.import_hbase("demo_test_hbase", [["test_family", "a", int],["test_family", "b", float], ["test_family", "c", int],["test_family", "d", int]])
        -etc-
        >>> frame.inspect()
        [#]  test_family_a  test_family_b  test_family_c  test_family_d
        ===============================================================
        [0]              1            0.2             -2              5
        [1]              2            0.4             -1              6
        [2]              3            0.6              0              7
        [3]              4            0.8              1              8

        Use of start_tag and end_tag. (Hbase creates a unique row id for data in hbase tables)
        start_tag: It is the unique row id from where row scan should start
        end_tag: It is the unique row id where row scan should end

        Assuming you already have data on hbase table "test_startendtag" under "startendtag" family name with single column named "number".
        data: column contains values from 1 to 99. Here rowid is generated by hbase.

        Sample hbase data. Few rows from hbase table looks as below.
        hbase(main):002:0> scan "test_startendtag"
        ROW             COLUMN+CELL
         0          column=startendtag:number, timestamp=1465342524846, value=1
         1          column=startendtag:number, timestamp=1465342524846, value=25
         10         column=startendtag:number, timestamp=1465342524847, value=51
         103        column=startendtag:number, timestamp=1465342524851, value=98
         107        column=startendtag:number, timestamp=1465342524851, value=99
         11         column=startendtag:number, timestamp=1465342524851, value=75
         12         column=startendtag:number, timestamp=1465342524846, value=4
         13         column=startendtag:number, timestamp=1465342524846, value=28
         14         column=startendtag:number, timestamp=1465342524847, value=52
         15         column=startendtag:number, timestamp=1465342524851, value=76
         16         column=startendtag:number, timestamp=1465342524846, value=5
         17         column=startendtag:number, timestamp=1465342524846, value=29
         18         column=startendtag:number, timestamp=1465342524847, value=53
         19         column=startendtag:number, timestamp=1465342524851, value=77
         2          column=startendtag:number, timestamp=1465342524847, value=49
         20         column=startendtag:number, timestamp=1465342524846, value=6
         21         column=startendtag:number, timestamp=1465342524846, value=30

        >>> frame = tc.frame.import_hbase("test_startendtag", [["startendtag", "number", int]], start_tag="20", end_tag="50")
        -etc-
        >>> frame.count()
        33
        >>> frame.inspect(33)
        [##]  startendtag_number
        ========================
        [0]                    6
        [1]                   30
        [2]                   54
        [3]                   78
        [4]                    7
        [5]                   31
        [6]                   55
        [7]                   79
        [8]                    8
        [9]                   32
        [10]                  73
        [11]                  56
        [12]                  80
        [13]                   9
        [14]                  33
        [15]                  57
        [16]                  81
        [17]                  10
        [18]                  34
        [19]                  58


        [##]  startendtag_number
        ========================
        [20]                  82
        [21]                   2
        [22]                  11
        [23]                  35
        [24]                  59
        [25]                  83
        [26]                  12
        [27]                  36
        [28]                  60
        [29]                  84
        [30]                  13
        [31]                  37
        [32]                  26

    </skip>

    """

    if not isinstance(table_name, basestring):
        raise ValueError("table name parameter must be a string, but is {0}.".format(type(table_name)))
    if not isinstance(schema, list):
        raise ValueError("schema parameter must be a list, but is {0}.".format(type(table_name)))
    TkContext.validate(tc)

    inner_lists=[tc._jutils.convert.to_scala_list([item[0], item[1], dtypes.to_string(item[2])]) for item in schema]
    scala_final_schema = tc.jutils.convert.to_scala_list(inner_lists)

    scala_frame = tc.sc._jvm.org.trustedanalytics.sparktk.frame.internal.constructors.Import.importHbase(tc.jutils.get_scala_sc(),
                                                                                                         table_name, scala_final_schema,
                                                                                                         tc._jutils.convert.to_scala_option(start_tag),
                                                                                                         tc._jutils.convert.to_scala_option(end_tag))

    from sparktk.frame.frame import Frame
    return Frame(tc, scala_frame)
