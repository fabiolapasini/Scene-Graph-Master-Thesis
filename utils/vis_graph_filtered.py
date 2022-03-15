#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Nov  9 18:38:55 2020

@author: sc
"""

if __name__ == '__main__' and __package__ is None:
    from os import sys

    sys.path.append('../')
import graphviz
import json
# import util
# import labels_utils
import numpy as np
# from plot_confusion_matrix import plot_confusion_matrix
import os
import operator
# import util_merge_same_part
import trimesh
# import operator
import math

# dot = graphviz.Digraph(comment='The Round Table',engine='circo') # neato, circo, dot

# dot.node('A', 'King Arthur')
# dot.node('B', 'Sir Bedevere the Wise')
# dot.node('L', 'Sir Lancelot the Brave')

# dot.edges(['AB', 'AL'])
# dot.edge('B', 'L', constraint='false',label='A')
# dot.view()
color_correct = '#0CF369'  # green
color_wrong = '#FF0000'  # red
color_missing_pd = '#A9A2A2'  # gray
color_missing_gt = '#0077FF'  # blue

debug = False
# debug = True
b_merge_floor = True


def process_node_pd(nodes):
    ns = dict()
    for k, v in nodes.items():
        if isinstance(v, dict):  # full output
            vv = max(v.items(), key=operator.itemgetter(1))[0]
            ns[k] = vv
        else:
            ns[k] = v
    return ns


def process_edge_pd(edges):
    es = dict()
    for k, v in edges.items():
        if isinstance(v, dict):
            vv = max(v.items(), key=operator.itemgetter(1))[0]
            es[k] = vv
        else:
            es[k] = v
    return es


def draw_evaluation(scan_id, node_pds, edge_pds, node_gts, edge_gts, none_name='UN',
                    pd_only=False, gt_only=False, inst_groups: dict = None, label_color=None, seg_colors: dict = None):
    g = graphviz.Digraph(comment=scan_id, format='png',
                         node_attr={'shape': 'circle',
                                    'style': 'filled',
                                    'fontname': 'helvetica',
                                    'color': 'lightblue2',
                                    'width': '1',
                                    'fontsize': '24',
                                    },
                         edge_attr={
                             'fontsize': '18',
                         },
                         graph_attr={'rankdir': 'LR',
                                     # 'center':'true',
                                     'splines': 'compound',
                                     'margin': '0.01',
                                     'fontsize': '24',
                                     'ranksep': '0.1',
                                     'nodesep': '0.1',
                                     'width': '1',
                                     # 'height':'20',
                                     },
                         # graph_attr={'rankdir': 'TB'},
                         engine='fdp'
                         )

    nodes = set(node_pds.keys()).union(node_gts.keys())

    selected_insts = []
    exclude_selected_insts = []
    selected_nodes = []
    # selected_insts = ['103','11','174','298','184','107','316']
    # selected_nodes = ['129','131','137','159','160','239','174']
    # exclude_selected_insts=['588','909','197','400','877']
    drawed_nodes = set()
    drawed_insts = set()
    if inst_groups is None:
        for idx in nodes:
            name_gt = none_name
            name_pd = none_name
            if idx not in node_pds:
                color = color_missing_pd
            else:
                name_pd = node_pds[idx]
            if idx not in node_gts:
                color = color_missing_gt
            else:
                name_gt = node_gts[idx]
            if idx in node_pds and idx in node_gts:
                color = color_correct if node_gts[idx] == node_pds[idx] else color_wrong
            g.node(idx, str(idx) + '_' + name_pd + '(' + name_gt + ')', color=color)
            drawed_nodes.add(idx)
    '''else:
        seg_to_inst = dict()
        wrong_segs = list()
        for k, v in inst_groups.items():
            if len(selected_insts) > 0:
                if k not in selected_insts: continue
            if len(exclude_selected_insts) > 0:
                if k in exclude_selected_insts: continue
            inst_name = 'cluster' + k
            with g.subgraph(name=inst_name) as c:
                name = node_pds[k]
                color = None
                # if seg_colors is not None:
                color = '#%02x%02x%02x' % label_color[labels_utils.nyu40_name_to_id(name) + 1]
                # color = '#%02x%02x%02x' % tuple(seg_colors[int(k)][:3])

                c.attr(label=name + '_' + k)
                if not debug: c.attr(label=name)
                c.attr(style='filled', color=color)
                # c.attr(fillcolor='white')
                # c.attr(style='filled')

                pre = []
                for idx in v:
                    if int(idx) not in seg_colors: continue
                    if len(selected_nodes) > 0:
                        if idx not in selected_nodes: continue

                    name_gt = none_name
                    name_pd = none_name
                    if idx in node_pds:
                        name_pd = node_pds[idx]
                    if idx in node_gts:
                        name_gt = node_gts[idx]

                    color = None
                    if label_color is not None:
                        # color = '#%02x%02x%02x' % label_color[labels_utils.nyu40_name_to_id(name)+1]
                        color = '#%02x%02x%02x' % tuple(seg_colors[int(idx)][:3])
                    # name = nodes[idx]
                    # color = '#%02x%02x%02x' % colors[labels_utils.nyu40_name_to_id(name)+1]

                    # c.node(idx, idx,{'shape':'circle'})

                    seg_to_inst[idx] = inst_name
                    node_label = '' if name_pd == name_gt else name_gt
                    if debug: node_label = str(idx) + '_' + node_label
                    # node_label = name_pd+'_'+name_gt
                    c.node(idx, node_label, color='white', fillcolor=color)
                    drawed_nodes.add(idx)
                    drawed_insts.add(inst_name)
                    if node_label == '':
                        wrong_segs.append(idx)

                    if len(pre) > 0:
                        g.edge(pre[-1], idx)
                        # return g'''

    edges = set(edge_pds.keys()).union(edge_gts.keys())
    i = 0
    for edge in edges:
        '''
        For each edge there may have multiple labels. 
        If non ground truth labels are given, set to missing_gt
        If gt labels are given, find the union and difference.
        '''
        f = edge.split('_')[0]
        t = edge.split('_')[1]

        # if f in seg_to_inst:
        #     f = seg_to_inst[f]
        # if t in seg_to_inst:
        #     t = seg_to_inst[t]
        # if f not in drawed_nodes or t not in drawed_nodes: continue
        # if f not in drawed_insts or t not in drawed_insts: continue
        # if f not in seg_to_inst or t not in seg_to_inst: continue
        # f = seg_to_inst[f]
        # t = seg_to_inst[t]

        if f == 'cluster901' and t == 'cluster312':
            print('helo')
            # continue
        if f == 'cluster527':
            print('2')
            # continue
        # elif f == 'cluster901':

        names_gt = list()
        names_pd = list()
        if edge in edge_pds:
            names_pd = edge_pds[edge] if isinstance(edge_pds[edge], list) else [edge_pds[edge]]

        if edge in edge_gts:
            names_gt = edge_gts[edge]

        names_pd = set(names_pd).difference([none_name, 'same part'])

        names_gt = set(names_gt).difference([none_name, 'same part'])

        if len(names_gt) > 0:
            intersection = set(names_gt).intersection(names_pd)  # match prediction
            diff_gt = set(names_gt).difference(intersection)  # unmatched gt
            diff_pd = set(names_pd).difference(intersection)  # unmatched pd

            ''' same part is in a box alearly only use outline color to indicate right or wrong'''

            for name in intersection:
                g.edge(f, t, label=name, color=color_correct)

            pds = ''
            if not gt_only:
                for name_pd in diff_pd:  # in pd but not in gt
                    if pds == '':
                        pds = name_pd
                    else:
                        pds = pds + ',' + name_pd
                    # g.edge(f,t,label=name_pd,color=color_wrong)
            gts = ''
            if not pd_only:
                for name_gt in diff_gt:  # in gt but not in pd
                    gts = name_gt if gts == '' else gts + ',' + name_gt
                    # g.edge(f,t,label=none_name+'\n('+name_gt+')',color=color_wrong) # color_missing_pd
            if pds != '' or gts != '':
                if pds != '': pds + '\n'
                g.edge(f, t, label=pds + '(' + gts + ')', color=color_wrong)  # color_missing_pd
        elif len(names_gt) == 0:  # missing gt
            if not gt_only:
                for name_pd in names_pd:
                    g.edge(f, t, label=name_pd, color=color_missing_gt)  # color_missing_pd
        i += 1
        # if i>30: break
    return g


def process_pd(nodes, edges):
    ns = dict()
    es = dict()
    for k, v in nodes.items():
        if isinstance(v, dict):  # full output
            vv = max(v.items(), key=operator.itemgetter(1))[0]
            ns[k] = vv
        else:
            ns[k] = v

    for k, v in edges.items():
        if isinstance(v, dict):
            vv = max(v.items(), key=operator.itemgetter(1))[0]
            es[k] = vv
        else:
            es[k] = v
    return ns, es


if __name__ == '__main__':
    '''
    Use find_correspondeing_gt.py to generate pd_gt_pairs.json
    this program loads that and geneate graph
    '''
    pth_base = '/home/sc/research/PersistentSLAM/c++/GraphSLAM/bin/draw_graph/'
    scan_id = '0a4b8ef6-a83a-21f2-8672-dce34dd0d7ca'
    pth_pd = pth_base + scan_id

    pth_data = os.path.join(pth_pd, 'pd_gt_pairs.json')

    pth_ply = os.path.join(pth_pd, 'node_segment.ply')
    # pth_ply = os.path.join(pth_pd,'node_panoptic.ply')
    cloud_pd = trimesh.load(pth_ply, process=False)
    colors = cloud_pd.colors
    segments_pd = cloud_pd.metadata['ply_raw']['vertex']['data']['label'].flatten()
    seg_colors = dict()
    segment_ids = np.unique(segments_pd)
    min_size = 512 * 8
    for idx in segment_ids:
        indices = np.where(segments_pd == idx)[0]
        if min_size > 0:
            if len(indices) > min_size:
                seg_colors[idx] = colors[indices[0]].tolist()


    pth_out = os.path.dirname(pth_data)
    scan_id = os.path.splitext(pth_pd)[0]
    ''' load predictions '''
    with open(pth_data, 'r') as f:
        data = json.load(f)

    node_pds = data['node_pds']
    node_gts = data['node_gts']
    edge_pds = data['edge_pds']
    edge_gts = data['edge_gts']

    # for k, v in data['edge_pds'].items():
    #     if k not in edge_gts:
    #         raise RuntimeError('')

    ''' filter small segments '''
    node_pds = {k: v for k, v in node_pds.items() if int(k) in seg_colors}
    tmp = dict()
    for k, v in edge_pds.items():
        sp = k.split('_')
        source, target = sp
        source, target = int(source), int(target)
        if source not in seg_colors or target not in seg_colors: continue
        tmp[k] = v
    edge_pds = tmp

    ''' Get instance groups '''
    node_pds = process_node_pd(node_pds)
    edge_pds = process_edge_pd(edge_pds)

    ''' merge floors '''
    if b_merge_floor:
        tmp = dict()
        tmp = set()
        for k, v in node_pds.items():
            if v == 'floor':
                tmp.add(k)
            # if k in node_gts:
            #     if node_gts[k] == 'floor':
            #         tmp.add(k)
        for t in tmp:
            for tt in tmp:
                if t == tt: continue
                name = str(t) + '_' + str(tt)
                if name in edge_pds: continue
                edge_pds[name] = 'same part'

    # inst_groups = util_merge_same_part.collect(node_pds, edge_pds)
    inst_groups = None
    seg_to_inst = dict()
    for k, v in inst_groups.items():
        for vv in v:
            seg_to_inst[vv] = k

    ''' collect predictions on inst level '''
    inst_edge_pred = dict()
    inst_edge_gt = dict()
    counter = dict()
    for k, v in data['edge_pds'].items():
        sp = k.split('_')
        source, target = sp
        if source not in seg_to_inst or target not in seg_to_inst: continue

        ''' check if prediction is correct'''
        s, t = seg_to_inst[source], seg_to_inst[target]

        ''' set name to cluster if the node on that cluster has the same label'''
        s = 'cluster' + str(s) if source in node_pds and source in node_gts and node_pds[source] == node_gts[source] \
            else source
        # t = 'cluster'+str(t) if target in node_pds and target in node_gts and node_pds[target] == node_gts[target] \
        #     else target
        t = 'cluster' + str(t)

        # if source in node_pds and source in node_gts and node_pds[source] == node_gts[source]:
        #     name = 'cluster'+str(s)+'_'+'cluster'+str(t)

        # else:
        name = str(s) + '_' + str(t)
        # name = str(s)+'_'+'cluster'+str(t)

        if name not in inst_edge_pred:
            inst_edge_pred[name] = dict()
            counter[name] = dict()
        for v_k, v_v in v.items():
            if v_k not in inst_edge_pred[name]:
                inst_edge_pred[name][v_k] = 0
                counter[name][v_k] = 0
            inst_edge_pred[name][v_k] += math.exp(v_v)
            counter[name][v_k] += 1

    for k, v in data['edge_gts'].items():
        sp = k.split('_')
        source, target = sp
        if source not in seg_to_inst or target not in seg_to_inst: continue

        ''' check if prediction is correct'''
        s, t = seg_to_inst[source], seg_to_inst[target]

        ''' set name to cluster if the node on that cluster has the same label'''
        s = 'cluster' + str(s) if source in node_pds and source in node_gts and node_pds[source] == node_gts[source] \
            else source
        # t = 'cluster'+str(t) if target in node_pds and target in node_gts and node_pds[target] == node_gts[target] \
        #     else target
        t = 'cluster' + str(t)
        name = str(s) + '_' + str(t)
        if name in inst_edge_gt:
            if inst_edge_gt[name] != edge_gts[k]:
                inst_edge_gt[name] = inst_edge_gt[name].union(set(edge_gts[k]))  # = inst_edge_gt[name]+','+
                # raise RuntimeError('duplicate',inst_edge_gt[name], edge_gts[k])
        else:
            inst_edge_gt[name] = set(edge_gts[k])

    for k, v in inst_edge_pred.items():
        for v_k, v_v in v.items():
            inst_edge_pred[k][v_k] /= counter[k][v_k]

    edge_pds = process_edge_pd(inst_edge_pred)
    edge_gts = inst_edge_gt

    g = draw_evaluation(scan_id, node_pds, edge_pds, node_gts, edge_gts)
    save_pth = os.path.join(pth_out, scan_id + '_graph')
    print(save_pth)
    g.render(save_pth, view=False)

