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
# from plot_confusion_matrix import plot_confusion_matrix
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

'''
def draw_prediction(scan_id, nodes,edges, colors):
    g = graphviz.Digraph(comment=scan_id,format='png',
                         node_attr={'shape': 'circle',
                                    'style': 'filled',
                                    'fontname':'helvetica',
                                    'color': 'lightblue2'},
                         edge_attr={'splines':'spline'},
                         rankdir = 'TB')
    for idx, name in nodes.items():
        g.node(idx, idx + '_' + name, color=colors[labels_utils.nyu40_name_to_id(name)])
        
    for edge, name in edges.items():
        f = edge.split('_')[0]
        t = edge.split('_')[1]
        if f not in nodes or t not in nodes: continue
        g.edge(f,t, label=name)
    return g 
'''


def draw_evaluation(scan_id, node_pds, edge_pds, node_gts, edge_gts, none_name = 'UN', pd_only=False, gt_only=False):
    g = graphviz.Digraph(comment=scan_id,format='png',
                         node_attr={'shape': 'circle',
                                    'style': 'filled',
                                    'fontname':'helvetica',
                                    'color': 'lightblue2'},
                         edge_attr={'splines':'spline'},
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
        '''
        For each edge there may have multiple labels. 
        If non ground truth labels are given, set to missing_gt
        If gt labels are given, find the union and difference.
        '''
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

        # if len(names_gt) == 0: # missing gt
        #     for name in names_pd:
        #         g.edge(f,t,label=name+'('+none_name+')',color=color_missing_gt)            
        # else:
        #     corrects = set(names_gt).intersection(names_pd)
        #     wrongs = set(names_gt).difference(names_pd)
        #     for name in corrects:
        #         g.edge(f,t,label=name_pd+'('+name_gt+')',color=color_correct)
        #     for name in wrongs:
        #         name_gt = name if name in names_gt else none_name
        #         name_pd = name if name in names_pd else none_name
        #         g.edge(f,t,label=name_pd+'('+name_gt+')',color=color_wrong)    
                
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
    return g
    

'''       
def evaluate_prediction(pd:map, gt:map, names:list, title:str, gt_only = False):
    c_mat = np.zeros([len(names)+1,len(names)+1])
    c_UNKNOWN = len(names)

    # For every predicted entry, check their corresponding gt.
    # if predicted idx it not in gt and gt_only is one, continue. Otherwise
    # count it as wrong prediction.

    for idx, name in pd.items():
        pd_idx = names.index(name)
        if idx not in gt:
            if gt_only: continue
            else: gt_idx = c_UNKNOWN
        else:
            gt_idx = names.index(gt[idx])
        c_mat[gt_idx][pd_idx] += 1

    # For every gt, if gt not in pd consider it as a wrong prediction.
    # The condition of gt present in pd is covered in the previous stage.

    for idx, name in gt.items():
        gt_idx = names.index(name)
        if idx not in pd:
            pd_idx = c_UNKNOWN
        c_mat[gt_idx][pd_idx] += 1    
        
    names_ = names + ['missed']
    plot_confusion_matrix(c_mat, 
                          target_names=names_, 
                          title=title,
                          plot_text=False,)
    return c_mat
'''


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


def read_classes(read_file):        # load list of relationships
    relationships = [] 
    with open(read_file, 'r') as f: 
        for line in f: 
            relationship = line.rstrip().lower() 
            relationships.append(relationship) 
    return relationships 


if __name__ == '__main__':
    pth_out = 'C:\\Users\\fabio\\Documents\\GitHub\\Scene-Graph-Master-Thesis\\SGPN\\SGPN\\results\\119\\graph\\'
    SCAN_PATH = 'C:\\Users\\fabio\\Documents\GitHub\\Scene-Graph-Master-Thesis\\3RScan\\'  # '/path/to/3RScan/'
    DATA_PATH = 'C:\\Users\\fabio\\Documents\\GitHub\\Scene-Graph-Master-Thesis\\Data\\'
    RESULT_PATH = 'C:\\Users\\fabio\\Documents\\GitHub\\Scene-Graph-Master-Thesis\\SGPN\\SGPN\\results\\119\\'  # '/path/to/result/'
    RESULT_PATH = os.path.join(RESULT_PATH, 'predictions.json')
    pd_only = False
    gt_only = False
    
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

        gts = {**gts,**gts_}

        pth_class = os.path.join(pth,'classes.txt')
        pth_relation = os.path.join(pth,'relationships.txt')
        classNames = sorted(read_classes(pth_class))
        predicateNames = sorted(read_classes(pth_relation))

        return gts, classNames, predicateNames


    gts, classNames, predicateNames = load_both(DATA_PATH)
    # print(gts)
    '''gts_, classNames_, predicateNames_ = load_both(pth_data_scannet)
    classNames=list(set(classNames).union(classNames_))
    predicateNames=list(set(predicateNames).union(predicateNames_))
    gts = {**gts,**gts_}'''
    
    ''' Generate random colors '''
    '''unique_colors = set()
    while len(unique_colors) < len(classNames):
        unique_colors.add('#'+util.color_hex(num=util.rand_24_bit()))
    unique_colors = list(unique_colors)'''

    eva_o_cls = util_eva.EvaClassification(classNames)
    eva_r_cls = util_eva.EvaClassification(predicateNames)

    for scan_id, prediction in predictions.items():
        """ in all the predictions.json file the scan id is the name f the scan follwed by '_0' """
        scan_id = scan_id.rsplit('_',1)[0]
        # if scan_id != '0a4b8ef6-a83a-21f2-8672-dce34dd0d7ca': continue
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
        
        # draw_prediction(scan_id, node_pds, edge_pds, unique_colors)
        
        eva_o_cls.update(node_pds, node_gts)
        eva_r_cls.update(edge_pds, edge_gts)
        
        # break
        '''Filter'''
        # choices = list(np.random.choice(np.unique(list(node_gts.keys())),20))
        # node_gts = {i:node_gts[i]  for i in choices}
        
        ''' draw '''
        # draw_prediction(scan_id, node_gts, edge_gts, unique_colors)
        
        g = draw_evaluation(scan_id, node_pds, edge_pds, node_gts, edge_gts, pd_only=pd_only,gt_only=gt_only)
        g.render(os.path.join(pth_out,scan_id+'_graph'),view=False)
        break

    eva_o_cls.draw('obj cmatrix')
    eva_r_cls.draw('rel cmarix')
    c_cmat = eva_o_cls.c_mat
    r_cmat = eva_r_cls.c_mat
    util_eva.write_result_file(c_cmat, pth_out + 'tmp_cls.txt', [], classNames)
    util_eva.write_result_file(r_cmat, pth_out + 'tmp_rel.txt', [], predicateNames)


