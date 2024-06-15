#!/usr/bin/env python3

from pytregex.condition import NODE_ID, NODE_REGEX, Condition, NodeDescription, NodeDescriptions
from pytregex.relation import *

from .base_tmpl import BaseTmpl


class TestNodeDescriptions(BaseTmpl):
    def test_repr(self):
        desc_1 = NodeDescription(NODE_ID, "S")
        desc_2 = NodeDescription(NODE_ID, "NN")
        node_descs1 = NodeDescriptions([desc_1, desc_2])
        self.assertEqual(str(node_descs1), "S|NN")
        node_descs1.add_description(NodeDescription(NODE_REGEX, "/V/"))
        self.assertEqual(str(node_descs1), "S|NN|/V/")
        node_descs1.negate()
        self.assertEqual(str(node_descs1), "!S|NN|/V/")
        node_descs1.enable_basic_cat()
        self.assertEqual(str(node_descs1), "!@S|NN|/V/")

        node_descs2 = NodeDescriptions([desc_1, desc_2])
        reldata = RelationData(CHILD_OF, "<")
        cond = Condition(relation_data=reldata, node_descriptions=node_descs2)
        node_descs1.set_condition(cond)

        self.assertEqual(str(node_descs1), "(!@S|NN|/V/ < S|NN)")
