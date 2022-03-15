#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Created on Thu Oct  1 13:12:44 2020
@author: sc
"""

if __name__ == '__main__' and __package__ is None:
    from os import sys
    sys.path.append('../')

import graphviz
import json
# import labels_utils   # this file doesn t exist
# import util
# import numpy as np
from plot_confusion_matrix import plot_confusion_matrix
import util_eva
import os
import operator

# dot = graphviz.Digraph(comment='The Round Table',engine='circo') # neato, circo, dot

# dot.node('A', 'King Arthur')
# dot.node('B', 'Sir Bedevere the Wise')
# dot.node('L', 'Sir Lancelot the Brave')

# dot.edges(['AB', 'AL'])
# dot.edge('B', 'L', constraint='false',label='A')
# dot.view()

color_correct = '#0CF369' # green
color_wrong = '#FF0000' # red
color_missing_pd = '#A9A2A2' # gray
color_missing_gt = '#0077FF' #blue


'''def draw_evaluation(scan_id, node_pds, edge_pds, node_gts, edge_gts, none_name = 'UN', pd_only=False, gt_only=False):
    g = graphviz.Digraph(comment=scan_id,format='png',
                         node_attr={'shape': 'circle',
                                    'ratio': 'compress',
                                    'size': '4500, 8000',
                                    'fixedsize': 'false',
                                    'style': 'filled',
                                    'fontname':'helvetica',
                                    'fontsize' : '30pt',
                                    'color': 'lightblue2'},
                         edge_attr={'splines':'spline',
                                    'fontname': 'helvetica',
                                    'fontsize': '30pt' },
                         graph_attr={'rankdir': 'TB'})
    nodes = set(node_pds.keys()).union(node_gts.keys())
    for idx in nodes:
        name_gt=none_name
        name_pd=none_name
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
        g.node(idx,str(idx) + '_' + name_pd+'('+name_gt+')',color=color)


    edges = set(edge_pds.keys()).union(edge_gts.keys())
    for edge in edges:
    
        # For each edge there may have multiple labels. 
        # If non ground truth labels are given, set to missing_gt
        # If gt labels are given, find the union and difference.
        
        f = edge.split('_')[0]
        t = edge.split('_')[1]
        names_gt=list()
        names_pd=list()
        if edge in edge_pds:
            names_pd = edge_pds[edge]

        if edge in edge_gts:
            names_gt = edge_gts[edge]

        names_pd = set(names_pd).difference([none_name])
        names_gt = set(names_gt).difference([none_name])

        if len(names_gt) > 0:
            intersection = set(names_gt).intersection(names_pd) # match prediction
            diff_gt = set(names_gt).difference(intersection) # unmatched gt
            diff_pd = set(names_pd).difference(intersection) # unmatched pd
            for name in intersection:
                g.edge(f,t,label=name+'('+name+')',color=color_correct)
            if not gt_only:
                for name_pd in diff_pd: # in pd but not in gt
                    g.edge(f,t,label=name_pd+'('+none_name+')',color=color_wrong)
            if not pd_only:
                for name_gt in diff_gt: # in gt but not in pd
                    g.edge(f,t,label=none_name+'('+name_gt+')',color=color_wrong) # color_missing_pd
        elif len(names_gt) == 0: # missing gt
            if not gt_only:
                for name_pd in names_pd:
                    g.edge(f,t,label=name_pd+'('+none_name+')',color=color_missing_gt) # color_missing_pd
    return g'''


def draw_evaluation(scan_id, node_pds, edge_pds, node_gts, edge_gts, none_name='UN',
                    pd_only=False, gt_only=False, inst_groups: dict = None, label_color=None, seg_colors: dict = None):
    '''g = graphviz.Digraph(comment=scan_id, format='png',
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
                         )'''
    g = graphviz.Digraph(comment=scan_id, format='png',
                         node_attr={'shape': 'circle',
                                    'ratio': 'compress',
                                    'size': '4500, 8000',
                                    'fixedsize': 'false',
                                    'style': 'filled',
                                    'fontname': 'helvetica',
                                    'fontsize': '30pt',
                                    'color': 'lightblue2'},
                         edge_attr={'splines': 'spline',
                                    'fontname': 'helvetica',
                                    'fontsize': '30pt'},
                         graph_attr={'rankdir': 'TB'})

    nodes = set(node_pds.keys()).union(node_gts.keys())

    selected_insts = []
    exclude_selected_insts = []
    selected_nodes = []
    # selected_insts = ['103','11','174','298','184','107','316']
    # selected_nodes = ['5','14','15','16','9','10','13','2','7','21']
    # exclude_selected_insts=['588','909','197','400','877']
    drawed_nodes = set()
    drawed_insts = set()
    if inst_groups is None:
        selected_nodes = ['5','14','15','16','9','10','13','2','7','21']
        for idx in nodes:
            if idx not in selected_nodes: continue
            name_gt = none_name
            name_pd = none_name
            if idx not in node_pds:
                print("IDX not in the selected ones, ", idx)
                # color = color_missing_pd
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
        for k,v in inst_groups.items():
            if len(selected_insts)>0:
                if k not in selected_insts:continue
            if len(exclude_selected_insts)>0:
                if k in exclude_selected_insts:continue
            inst_name = 'cluster'+k
            with g.subgraph(name = inst_name) as c:
                name = node_pds[k]
                color = None
                # if seg_colors is not None:
                color = '#%02x%02x%02x' % label_color[labels_utils.nyu40_name_to_id(name)+1]
                    # color = '#%02x%02x%02x' % tuple(seg_colors[int(k)][:3])

                c.attr(label=name+'_'+k)          
                if not debug: c.attr(label=name)          
                c.attr(style='filled', color=color)
                # c.attr(fillcolor='white')
                # c.attr(style='filled')

                pre = []
                for idx in v:
                    if int(idx) not in seg_colors: continue
                    if len(selected_nodes)>0:
                        if idx not in selected_nodes: continue

                    name_gt=none_name
                    name_pd=none_name
                    if idx in node_pds: 
                        name_pd = node_pds[idx]
                    if idx in node_gts:
                        name_gt = node_gts[idx]

                    color=None
                    if label_color is not None:
                        # color = '#%02x%02x%02x' % label_color[labels_utils.nyu40_name_to_id(name)+1]
                        color = '#%02x%02x%02x' % tuple(seg_colors[int(idx)][:3])
                    # name = nodes[idx]
                    # color = '#%02x%02x%02x' % colors[labels_utils.nyu40_name_to_id(name)+1]

                    # c.node(idx, idx,{'shape':'circle'})

                    seg_to_inst[idx] = inst_name
                    node_label = '' if name_pd == name_gt else name_gt
                    if debug: node_label =str(idx)+'_'+node_label
                    # node_label = name_pd+'_'+name_gt
                    c.node(idx, node_label, color='white',fillcolor=color)
                    drawed_nodes.add(idx)
                    drawed_insts.add(inst_name)
                    if node_label == '':
                        wrong_segs.append(idx)

                    if len(pre)>0:
                        g.edge(pre[-1],idx)    
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

        if f not in selected_nodes or t not in selected_nodes: continue

        names_gt = list()
        names_pd = list()
        if edge in edge_pds:
            names_pd = edge_pds[edge] if isinstance(edge_pds[edge], list) else [edge_pds[edge]]

        if edge in edge_gts:
            names_gt = edge_gts[edge]

        names_pd = set(names_pd).difference([none_name, 'same part'])

        names_gt = set(names_gt).difference([none_name, 'same part'])

        if len(names_gt) > 0:
            intersection = set(names_gt).intersection(names_pd)     # match prediction
            diff_gt = set(names_gt).difference(intersection)        # unmatched gt
            diff_pd = set(names_pd).difference(intersection)        # unmatched pd

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


def process_gt(nodes,edges):
    nodes_gts = dict()
    for idx, pd in nodes.items():
        nodes_gts[str(idx)] = pd
        pass
    edges_gts = dict()
    for edge in edges:
        name = str(edge[0])+'_'+str(edge[1])
        if name not in edges_gts:
            edges_gts[name] = list()
        edges_gts[name].append(edge[3])
    return nodes_gts, edges_gts


def process_pd(nodes,edges):
    ns = dict()
    es = dict()
    for k, v in nodes.items():
        if isinstance(v,dict):      # full output
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


def read_classes(read_file):
    relationships = []
    with open(read_file, 'r') as f:
        for line in f:
            relationship = line.rstrip().lower()
            relationships.append(relationship)
    return relationships


if __name__ == '__main__':
    pth_out = "..\\SGPN\\SGPN\\results\\GCN_600\\graph\\"
    DATA_PATH = '..\\gen_data\\'
    RESULT_PATH = "..\\SGPN\\SGPN\\results\\GCN_600\\"
    RESULT_PATH = os.path.join(RESULT_PATH, 'predictions.json')

    ''' load predictions '''
    with open(RESULT_PATH, 'r') as f:
        predictions = json.load(f)

    def load_scans(pth):
        data=dict()
        with open(pth, 'r') as f:
            data_ = json.load(f)
        for g in data_['scans']:
            data[g['scan']] = dict()
            data[g['scan']]['split'] = g['split']
            data[g['scan']]['objects'] = g['objects']
            data[g['scan']]['relationships'] = g['relationships']
        return data

    def load_both(pth):
        pth_gt = pth+ 'relationships_validation.json'
        gts = load_scans(pth_gt)

        pth_gt = pth+ 'relationships_train.json'
        gts_ = load_scans(pth_gt)

        pth_gt = pth + 'relationships_test.json'
        gts__ = load_scans(pth_gt)

        gts = {**gts, **gts_, **gts__}

        pth_class = os.path.join(pth,'classes.txt')
        pth_relation = os.path.join(pth,'relationships.txt')
        classNames = sorted(read_classes(pth_class))
        predicateNames = sorted(read_classes(pth_relation))

        return gts, classNames, predicateNames


    gts, classNames, predicateNames = load_both(DATA_PATH)
    gts_, classNames_, predicateNames_ = load_both(DATA_PATH)
    classNames=list(set(classNames).union(classNames_))
    predicateNames=list(set(predicateNames).union(predicateNames_))
    gts = {**gts,**gts_}

    for scan_id, prediction in predictions.items():
        """ in all the predictions.json file the scan id is the name f the scan follwed by '_0' """
        scan_id = scan_id.rsplit('_',1)[0]
        print('scan_id: ', scan_id)
        gt_scan = gts[scan_id]

        if 'pd' in prediction:
            node_pds, edge_pds = prediction['pd']['nodes'],prediction['pd']['edges']
            node_gts, edge_gts = prediction['gt']['nodes'],prediction['gt']['edges']
        else:
            node_pds, edge_pds = prediction['nodes'],prediction['edges']
            node_gts, edge_gts = process_gt(gt_scan['objects'],gt_scan['relationships'])

        node_pds, edge_pds = process_pd(node_pds, edge_pds)

        # check nodes and edges
        # node_ids = [int(id) for id in nodes.keys()]
        for edge in edge_pds:
            e = edge.split('_')
            if e[0] not in node_pds:
                print(edge[0])
            if e[1] not in node_pds:
                print(edge[1])


        # break
        '''Filter'''
        # choices = list(np.random.choice(np.unique(list(node_gts.keys())),20))
        # node_gts = {i:node_gts[i]  for i in choices}

        ''' draw '''
        g = draw_evaluation(scan_id, node_pds, edge_pds, node_gts, edge_gts)
        g.render(os.path.join(pth_out,scan_id+'_graph'),view=False)
        break


