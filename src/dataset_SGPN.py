'''if __name__ == '__main__' or __package__ is None:
    from os import sys
    sys.path.append('../')'''

import os, torch, json, trimesh
os.sys.path.append('../')
from utils import util_ply, util_data, util
from data_processing import compute_weight_occurrences
import numpy as np
import torch.utils.data as data
import multiprocessing as mp

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print("Running on", device)

import platform
if (platform.system() == "Windows"):
    from utils import define_win as define
    print("Os: Windows")
elif (platform.system() != "Windows"):
    from utils import define
    print("Os: Ubuntu")

def dataset_loading_3RScan(root:str, pth_selection:str,split:str,class_choice:list=None):    
    pth_catfile = os.path.join(pth_selection, 'classes.txt')
    classNames = util.read_txt_to_list(pth_catfile)
    
    pth_relationship = os.path.join(pth_selection, 'relationships.txt')
    util.check_file_exist(pth_relationship)
    relationNames = util.read_relationships(pth_relationship)
    
    selected_scans=set()
    if split == 'train_scans' :
        selected_scans = selected_scans.union(util.read_txt_to_list(os.path.join(pth_selection,'train_scans.txt')))
    elif split == 'validation_scans':
        selected_scans = selected_scans.union(util.read_txt_to_list(os.path.join(pth_selection,'validation_scans.txt')))
    elif split == 'test_scans':
        selected_scans = selected_scans.union(util.read_txt_to_list(os.path.join(pth_selection,'test_scans.txt')))
    else:
        raise RuntimeError('unknown split type:',split)

    with open(os.path.join(root, 'relationships_train.json'), "r") as read_file:
        data1 = json.load(read_file)
    with open(os.path.join(root, 'relationships_validation.json'), "r") as read_file:
        data2 = json.load(read_file)
    data = dict()
    data['scans'] = data1['scans'] + data2['scans']
    data['neighbors'] = {**data1['neighbors'], **data2['neighbors']}
    return  classNames, relationNames, data, selected_scans


def gen_modelnet_id(root):
    classes = []
    with open(os.path.join(root, 'train.txt'), 'r') as f:
        for line in f:
            classes.append(line.strip().split('/')[0])
    classes = np.unique(classes)
    with open(os.path.join(os.path.dirname(os.path.realpath(__file__)), '../misc/modelnet_id.txt'), 'w') as f:
        for i in range(len(classes)):
            f.write('{}\t{}\n'.format(classes[i], i))
            
            
def load_mesh(path,label_file,use_rgb,use_normal):
    result=dict()
    if label_file == 'labels.instances.align.annotated.v2.ply' or label_file == 'labels.instances.align.annotated.ply':
        if use_rgb:
            plydata = util_ply.load_rgb(path)
        else:
            plydata = trimesh.load(os.path.join(path,label_file), process=False)
            
        points = np.array(plydata.vertices.tolist())
        instances = util_ply.read_labels(plydata).flatten()
        
        if use_rgb:
            rgbs = np.array(plydata.visual.vertex_colors.tolist())[:,:3]
            points = np.concatenate((points, rgbs), axis=1)
            
        if use_normal:
            normal = plydata.vertex_normals[:,:3]
            points = np.concatenate((points, normal), axis=1)
        
        result['points']=points
        result['instances']=instances

    '''elif label_file == 'inseg.ply':
        plydata = trimesh.load(os.path.join(path,label_file), process=False)
        points = np.array(plydata.vertices.tolist())
        instances = plydata.metadata['ply_raw']['vertex']['data']['label'].flatten()
        
        if use_rgb:
            rgbs = np.array(plydata.colors)[:,:3] / 255.0
            points = np.concatenate((points, rgbs), axis=1)
        if use_normal:
            nx = plydata.metadata['ply_raw']['vertex']['data']['nx']
            ny = plydata.metadata['ply_raw']['vertex']['data']['ny']
            nz = plydata.metadata['ply_raw']['vertex']['data']['nz']
            
            normal = np.stack([ nx,ny,nz ]).squeeze().transpose()
            points = np.concatenate((points, normal), axis=1)
        result['points']=points
        result['instances']=instances
    else:
        raise NotImplementedError('')'''
    return result

class RIODatasetGraph(data.Dataset):
    def __init__(self,
                 config,
                 split='train_scans',
                 multi_rel_outputs=True,
                 shuffle_objs=True,
                 use_rgb = False,
                 use_normal = False,
                 load_cache = False,
                 sample_in_runtime=True,
                 for_eval = False,
                 max_edges = -1):
        assert split in ['train_scans', 'validation_scans','test_scans']
        self.config = config
        self.mconfig = config.dataset
        
        self.root = self.mconfig.root
        self.root_3rscan = define.DATA_PATH
        try:
            self.root_scannet = define.SCANNET_DATA_PATH
        except:
            self.root_scannet = None

        self.scans = []
        self.multi_rel_outputs = multi_rel_outputs
        self.shuffle_objs = shuffle_objs
        self.use_rgb = use_rgb
        self.use_normal = use_normal
        self.sample_in_runtime=sample_in_runtime
        self.load_cache = load_cache
        self.for_eval = for_eval
        self.max_edges=max_edges
        
        '''import resource
        rlimit = resource.getrlimit(resource.RLIMIT_NOFILE)
        resource.setrlimit(resource.RLIMIT_NOFILE, (4096, rlimit[1]))'''

        #print(self.root)            # ../data/tmp/
        #print(len(self.root))       # 41
        
        if isinstance(self.root, list):
            with open(os.path.join(self.root[0],'args.json'), 'r') as f:
                data = json.load(f)
                self.label_type = data['label_type']
                # self.isV2 = data['v2'] > 0
            classNames = None
            relationNames = None
            data = None
            selected_scans = None
            for i in range(len(self.root)):
                selection = self.mconfig.selection
                if selection == "":
                    selection = self.root[i]
                l_classNames, l_relationNames, l_data, l_selected_scans = \
                    dataset_loading_3RScan(self.root[i], selection, split)
                
                if classNames is None:
                    classNames, relationNames, data, selected_scans = \
                        l_classNames, l_relationNames, l_data, l_selected_scans
                else:
                    classNames = set(classNames).union(l_classNames)
                    relationNames= set(relationNames).union(l_relationNames)
                    data['scans'] = l_data['scans'] + data['scans']
                    data['neighbors'] = {**l_data['neighbors'], **data['neighbors']}
                    selected_scans = selected_scans.union(l_selected_scans)
            classNames = list(classNames)
            relationNames = list(relationNames)
        else:
            with open(os.path.join(self.root,'args.json'), 'r') as f:
                data = json.load(f)
                self.label_type = data['label_type']
                # self.isV2 = data['v2'] > 0
            
            if self.mconfig.selection == "":
                self.mconfig.selection = self.root
            classNames, relationNames, data, selected_scans = \
                dataset_loading_3RScan(self.root, self.mconfig.selection, split)        
        
        self.relationNames = sorted(relationNames)
        self.classNames = sorted(classNames)
        
        if not multi_rel_outputs:
            if 'none' not in self.relationNames:
                self.relationNames.append('none')
                
        wobjs, wrels, o_obj_cls, o_rel_cls = compute_weight_occurrences.compute(self.classNames, self.relationNames, data,selected_scans, False)
        self.w_cls_obj = torch.from_numpy(np.array(o_obj_cls)).float().to(self.config.DEVICE)
        self.w_cls_rel = torch.from_numpy(np.array(o_rel_cls)).float().to(self.config.DEVICE)
        self.w_cls_obj = torch.abs(1.0 / (torch.log(self.w_cls_obj)+1)) # +1 to prevent 1 /log(1) = inf
        self.w_cls_rel = torch.abs(1.0 / (torch.log(self.w_cls_rel)+1)) # +1 to prevent 1 /log(1) = inf
        
        if not multi_rel_outputs:
            self.w_cls_rel[-1] = 0.001
            
        print('=== {} classes ==='.format(len(self.classNames)))
        for i in range(len(self.classNames)):
            print('|{0:>2d} {1:>20s}'.format(i,self.classNames[i]),end='')
            if self.w_cls_obj is not None:
                print(':{0:>1.3f}|'.format(self.w_cls_obj[i]),end='')
            if (i+1) % 2 ==0:
                print('')
        print('')
        print('=== {} relationships ==='.format(len(self.relationNames)))
        for i in range(len(self.relationNames)):
            print('|{0:>2d} {1:>20s}'.format(i,self.relationNames[i]),end=' ')
            if self.w_cls_rel is not None:
                print('{0:>1.3f}|'.format(self.w_cls_rel[i]),end='')
            if (i+1) % 2 ==0:
                print('')
        print('')
        
        self.relationship_json, self.objs_json, self.scans, self.nns = self.read_relationship_json(data, selected_scans)
        print('num of data:',len(self.scans))
        assert(len(self.scans)>0)
        if sample_in_runtime:
            assert(self.nns is not None)
            
        self.dim_pts = 3
        if self.use_rgb:
            self.dim_pts += 3
        if self.use_normal:
            self.dim_pts += 3
        
        self.cache_data = dict()
        if load_cache:
            pool = mp.Pool(8)
            pool.daemon = True
            # resutls=dict()
            for scan_id in self.scans:
                scan_id_no_split = scan_id.rsplit('_',1)[0]
                if 'scene' in scan_id:
                    path = os.path.join(self.root_scannet, scan_id_no_split)
                else:
                    path = os.path.join(self.root_3rscan, scan_id_no_split)
                if scan_id_no_split not in self.cache_data:
                    self.cache_data[scan_id_no_split] = pool.apply_async(load_mesh,
                                                                          (path, self.mconfig.label_file,self.use_rgb,self.use_normal))
            pool.close()
            pool.join()
            for key, item in self.cache_data.items():
                self.cache_data[key] = item.get()

    def __getitem__(self, index):
        scan_id = self.scans[index]
        scan_id_no_split = scan_id.rsplit('_',1)[0]
        selected_instances = list(self.objs_json[scan_id].keys())
        map_instance2labelName = self.objs_json[scan_id]
        
        if self.load_cache:
            data = self.cache_data[scan_id_no_split]
        else:
            if 'scene' in scan_id:
                path = os.path.join(self.root_scannet, scan_id_no_split)
            else:
                path = os.path.join(self.root_3rscan, scan_id_no_split)
            data = load_mesh(path, self.mconfig.label_file, self.use_rgb, self.use_normal)
        points = data['points']
        instances = data['instances']

        sample_num_nn=1
        sample_num_seed=1
        if "sample_num_nn" in self.mconfig:
            sample_num_nn = self.mconfig.sample_num_nn
        if "sample_num_seed" in self.mconfig:
            sample_num_seed = self.mconfig.sample_num_seed

        obj_points, rel_points, edge_indices, instance2mask, gt_rels, gt_class = \
            util_data.data_preparation(points, instances, selected_instances, 
                         self.mconfig.num_points, self.mconfig.num_points_union,
                         # use_rgb=self.use_rgb,use_normal=self.use_normal,
                         for_train=True, instance2labelName=map_instance2labelName, 
                         classNames=self.classNames,
                         rel_json=self.relationship_json[scan_id], 
                         relationships=self.relationNames,
                         multi_rel_outputs=self.multi_rel_outputs,
                         padding=0.2,num_max_rel=self.max_edges,
                         shuffle_objs=self.shuffle_objs, nns=self.nns[scan_id_no_split],
                         sample_in_runtime=self.sample_in_runtime,
                         num_nn=sample_num_nn, num_seed=sample_num_seed,
                         use_all = self.for_eval)

        return scan_id, instance2mask, obj_points, rel_points, gt_class, gt_rels, edge_indices

    def __len__(self):
        return len(self.scans)
    
    def read_relationship_json(self, data, selected_scans:list):
        rel = dict()
        objs = dict()
        scans = list()
        nns = None
        
        if 'neighbors' in data:
            nns = data['neighbors']
        for scan in data['scans']:
            if scan["scan"] == 'fa79392f-7766-2d5c-869a-f5d6cfb62fc6':
                if self.mconfig.label_file == "labels.instances.align.annotated.v2.ply":
                    '''
                    In the 3RScanV2, the segments on the semseg file and its ply file mismatch. 
                    This causes error in loading data.
                    To verify this, run check_seg.py
                    '''
                    continue
            if scan['scan'] not in selected_scans:
                continue
                
            relationships = []
            for realationship in scan["relationships"]:
                relationships.append(realationship)
                
            objects = {}
            for k, v in scan["objects"].items():
                objects[int(k)] = v
                
            # filter scans that doesn't have the classes we care
            instances_id = list(objects.keys())
            valid_counter = 0
            for instance_id in instances_id:
                instance_labelName = objects[instance_id]
                if instance_labelName in self.classNames: # is it a class we care about?
                    valid_counter+=1
            if valid_counter < 2: # need at least two nodes
                continue

            rel[scan["scan"] + "_" + str(scan["split"])] = relationships
            scans.append(scan["scan"] + "_" + str(scan["split"]))

            
            objs[scan["scan"]+"_"+str(scan['split'])] = objects

        return rel, objs, scans, nns
    
if __name__ == '__main__':
    from config import Config

    config = Config('../config_example.json')
    config.dataset.root = '../data/tmp/'
    config.dataset.label_file = 'labels.instances.align.annotated.v2.ply' #'inseg.ply'
    config.dataset_type = 'SGPN'
    config.dataset.load_cache=False
    use_rgb= False #True
    use_normal=True
    dataset = RIODatasetGraph(config,
                              split='validation_scans',
                              load_cache=config.dataset.load_cache,
                              use_rgb=use_rgb,
                              use_normal=use_normal)

    scan_id, instance2mask, obj_points, rel_points, cat, target_rels, edge_indices = dataset.__getitem__(0)

    ''' 

    === 160 classes ===
    | 0             armchair:0.591|| 1             backpack:0.000|
    | 2                  bag:0.000|| 3                 ball:0.000|
    | 4                  bar:0.000|| 5                basin:0.000|
    | 6               basket:0.000|| 7         bath cabinet:0.000|
    | 8              bathtub:0.000|| 9                  bed:0.419|
    |10        bedside table:0.000||11                bench:0.000|
    |12                bidet:0.591||13                  bin:0.591|
    |14              blanket:0.419||15               blinds:0.000|
    |16                board:0.000||17                 book:0.000|
    |18                books:0.000||19            bookshelf:0.000|
    |20               bottle:0.000||21                  box:0.591|
    |22                bread:0.000||23               bucket:0.358|
    |24              cabinet:0.000||25               carpet:0.000|
    |26              ceiling:0.265||27                chair:0.591|
    |28             cleanser:0.000||29                clock:0.000|
    |30               closet:0.000||31              clothes:0.591|
    |32        clothes dryer:0.000||33              clutter:0.000|
    |34       coffee machine:0.000||35         coffee table:0.000|
    |36              commode:0.000||37        computer desk:0.000|
    |38                couch:0.000||39          couch table:0.000|
    |40              counter:0.000||41                  cup:0.000|
    |42             cupboard:0.000||43              curtain:0.303|
    |44              cushion:0.000||45        cutting board:0.000|
    |46           decoration:0.000||47                 desk:0.000|
    |48         dining chair:0.000||49         dining table:0.000|
    |50                 door:0.287||51            doorframe:0.591|
    |52               drawer:0.000||53                 drum:0.000|
    |54       drying machine:0.000||55        extractor fan:0.000|
    |56            fireplace:0.000||57                floor:0.287|
    |58               flower:0.000||59              flowers:0.000|
    |60               folder:0.000||61                 food:0.000|
    |62            footstool:0.000||63                frame:0.000|
    |64          fruit plate:0.000||65              garbage:0.000|
    |66          garbage bin:0.000||67                grass:0.000|
    |68           hand dryer:0.000||69               heater:0.000|
    |70                 item:0.000||71               jacket:0.000|
    |72                  jar:0.000||73               kettle:0.000|
    |74    kitchen appliance:0.000||75      kitchen cabinet:0.000|
    |76      kitchen counter:0.000||77         kitchen hood:0.000|
    |78               ladder:0.000||79                 lamp:0.419|
    |80               laptop:0.000||81       laundry basket:0.000|
    |82                light:0.591||83              machine:0.000|
    |84        magazine rack:0.000||85                 menu:0.000|
    |86            microwave:0.000||87               mirror:0.358|
    |88              monitor:0.000||89              napkins:0.000|
    |90           nightstand:0.419||91               object:0.000|
    |92              objects:0.000||93            organizer:0.000|
    |94              ottoman:0.000||95                 oven:0.000|
    |96                 pack:0.000||97                  pan:0.000|
    |98          paper towel:0.000||99               papers:0.000|
    |100                   pc:0.000||101              picture:0.591|
    |102        pile of books:0.000||103       pile of papers:0.000|
    |104               pillow:0.257||105                 pipe:0.000|
    |106                plant:0.000||107                plate:0.000|
    |108               player:0.000||109                  pot:0.000|
    |110              printer:0.000||111                 rack:0.591|
    |112             radiator:0.000||113          recycle bin:0.000|
    |114         refrigerator:0.000||115        rocking chair:0.000|
    |116                scale:0.000||117               screen:0.000|
    |118                shelf:0.303||119                 shoe:0.000|
    |120            shoe rack:0.591||121                shoes:0.000|
    |122             showcase:0.000||123               shower:0.325|
    |124       shower curtain:0.325||125         shower floor:0.000|
    |126          shower wall:0.000||127           side table:0.591|
    |128                 sink:0.325||129            soap dish:0.000|
    |130               socket:0.000||131                 sofa:0.000|
    |132           sofa chair:0.000||133                stair:0.000|
    |134                stand:0.000||135                stool:0.358|
    |136                stove:0.000||137       stuffed animal:0.000|
    |138             suitcase:0.591||139                table:0.591|
    |140           table lamp:0.000||141            telephone:0.000|
    |142              toaster:0.000||143               toilet:0.325|
    |144         toilet brush:0.000||145         toilet paper:0.000|
    |146 toilet paper dispenser:0.358||147                towel:0.000|
    |148            trash can:0.358||149             trashcan:0.000|
    |150                 tube:0.000||151                   tv:0.419|
    |152             tv stand:0.000||153                 vase:0.000|
    |154                 wall:0.200||155             wardrobe:0.591|
    |156      washing machine:0.591||157       washing powder:0.000|
    |158               window:0.000||159           windowsill:0.000|
    
    === 7 relationships ===
    | 0          attached to 0.172|| 1             build in 0.000|
    | 2         connected to 0.000|| 3           hanging on 0.231|
    | 4              part of 0.000|| 5          standing on 0.192|
    | 6         supported by 0.287|
    num of data: 12


    '''
    
    pass
