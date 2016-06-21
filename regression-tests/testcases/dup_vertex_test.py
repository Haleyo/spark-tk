##############################################################################
# INTEL CONFIDENTIAL
#
# Copyright 2014, 2015 Intel Corporation All Rights Reserved.
#
# The source code contained or described herein and all documents related to
# the source code (Material) are owned by Intel Corporation or its suppliers
# or licensors. Title to the Material remains with Intel Corporation or its
# suppliers and licensors. The Material may contain trade secrets and
# proprietary and confidential information of Intel Corporation and its
# suppliers and licensors, and is protected by worldwide copyright and trade
# secret laws and treaty provisions. No part of the Material may be used,
# copied, reproduced, modified, published, uploaded, posted, transmitted,
# distributed, or disclosed in any way without Intel's prior express written
# permission.
#
# No license under any patent, copyright, trade secret or other intellectual
# property right is granted to or conferred upon you by disclosure or
# delivery of the Materials, either expressly, by implication, inducement,
# estoppel or otherwise. Any license under such intellectual property rights
# must be express and approved by Intel in writing.
##############################################################################
"""
    Usage:  python2.7 dup_vertex_test.py
    Tests readding vertices with new information
"""
__author__ = 'Prune Wickart'
__credits__ = ["Prune Wickart"]
__version__ = "2015.06.09"

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.realpath(__file__))))

import unittest

import trustedanalytics as ia

from qalib import frame_utils
from qalib import atk_test


class DupVertex(atk_test.ATKTestCase):

    def setUp(self):
        super(DupVertex, self).setUp()

        self.example_data = "example_count.csv"
        self.example_schema = [("idnum", ia.int32)]

        self.employee_data = "employee_example.csv"
        self.employee_schema = [("Employee", str),
                                ("Manager", str),
                                ("Title", str),
                                ("Years", ia.int32)]

        self.dot_product_schema = [("col_0", ia.int32)]
        self.flatten_schema = [("a", ia.int32)]

    def test_duplicating_vertices(self):
        """Verify that updating vertices properly updates and is sane """
        expected_vertex = [
            ['Anup', 'Associate'],
            ['Bob', 'Associate'],
            ['David', 'VP of Sales'],
            ['Jane', 'Sn Associate'],
            ['Larry', 'Product Manager'],
            ['Mohit', 'Associate'],
            ['Rob', None],      # Rob is the top manager; no employee record
            ['Steve', 'Marketing Manager'],
            ['Sue', 'Market Analyst']
        ]
        # expected_edge = [[],]

        updated_vertex = [
            ['Anup', '1'],
            ['Bob', '2'],
            ['David', '3'],
            ['Jane', '4'],
            ['Larry', '5'],
            ['Mohit', '0'],
            ['Rob', '6'],
            ['Steve', 'Marketing Manager'],
        ]

        updated_frame = frame_utils.build_frame(
            updated_vertex, [("Employee", str), ("Title", str)],
            file_format="list")
        employees_frame = frame_utils.build_frame(
            self.employee_data, self.employee_schema, skip_header_lines=1)

        my_graph = ia.Graph()
        my_graph.define_vertex_type(u'Employee')
        my_graph.define_edge_type(
            u'worksunder', u'Employee', u'Employee', directed=True)

        my_graph.vertices['Employee'].add_vertices(
            employees_frame, 'Employee', ['Title'])
        print "VERTEX FRAME:\n", my_graph.vertices['Employee'].inspect(20)
        print "EDGE FRAME:\n", my_graph.edges['worksunder'].inspect(20)

        my_graph.vertices['Employee'].add_vertices(
            updated_frame, 'Employee', ['Title'])
        my_graph.edges['worksunder'].add_edges(
            employees_frame, 'Employee', 'Manager', ['Years'],
            create_missing_vertices=True)

        print my_graph.vertex_count, "vertices"
        print my_graph.edge_count, "edges"
        print "VERTEX FRAME:\n", my_graph.vertices['Employee'].inspect(20)
        print "EDGE FRAME:\n", my_graph.edges['worksunder'].inspect(20)

        # Checking edge validity is too complex for simple regressions;
        #   just check the vertex list.
        my_graph.vertices['Employee'].sort('Employee')
        print my_graph.vertices['Employee'].inspect(20)
        vertex_take = my_graph.vertices['Employee'].download(20)
        edges_take = my_graph.edges['worksunder'].download(20)
        print "VERTEX TAKE:\n", vertex_take
        print "EDGE TAKE:\n", edges_take
        vid = vertex_take["_vid"].tolist()
        edge_src = edges_take["_src_vid"].tolist()
        edge_dest = edges_take["_dest_vid"].tolist()
        updated_vertex.append([u'Sue', u'Market Analyst'])
        for _, i in vertex_take[["Employee", "Title"]].iterrows():
            self.assertIn(i.tolist(), updated_vertex)

        for i in edge_src:
            self.assertIn(i, vid)

        for i in edge_dest:
            self.assertIn(i, vid)


if __name__ == '__main__':
    unittest.main()
